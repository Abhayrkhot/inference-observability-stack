"""Generate Prometheus recording and multi-window, multi-burn-rate alert rules from an SLO spec.

The alert tiers follow the Google SRE workbook (chapter 5, "Alerting on SLOs") for a 30-day
period: page when 2% of the budget burns in 1 hour or 5% in 6 hours, open a ticket at 10% in 3 days
or 10% in 1 day (expressed as burn-rate thresholds 14.4, 6, 3 and 1, each with a long window and a
short window that must both exceed it). ``window_scale`` shrinks every window and ``for`` duration
by the same factor so detection behaviour can be exercised in minutes; the default spec uses 1.

Spec (YAML)::

    period: 30d
    prefix: slo                 # recording-rule prefix
    alert_name: SLOErrorBudgetBurn
    window_scale: 1
    scrape_interval: 5s         # short windows must span at least 4 scrapes
    slos:
      - name: availability
        objective: 0.995
        ratio:                  # $window is replaced by each window
          errors: sum(rate(x_count{error_type!=""}[$window]))
          total: sum(rate(x_count[$window]))
      - name: ttfc
        objective: 0.99
        latency:                # good = observations <= threshold (must be a bucket bound)
          metric: gen_ai_client_operation_time_to_first_chunk_seconds
          threshold: 1.0
          selector: ''
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)(ms|s|m|h|d|w)$")
UNIT_S = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


@dataclass(frozen=True)
class Tier:
    long: str
    short: str
    burn: float
    for_: str
    severity: str


WORKBOOK_TIERS = (
    Tier("1h", "5m", 14.4, "2m", "page"),
    Tier("6h", "30m", 6.0, "15m", "page"),
    Tier("1d", "2h", 3.0, "1h", "ticket"),
    Tier("3d", "6h", 1.0, "1h", "ticket"),
)


def parse_duration(d: str) -> float:
    m = DURATION_RE.match(d.strip())
    if not m:
        raise ValueError(f"bad duration {d!r}")
    return float(m.group(1)) * UNIT_S[m.group(2)]


def fmt_duration(seconds: float) -> str:
    """Shortest exact Prometheus duration for a whole number of seconds (rounded)."""
    s = max(1, round(seconds))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if s % size == 0:
            return f"{s // size}{unit}"
    return f"{s}s"


def le_matcher(threshold: float) -> str:
    """Regex (as PromQL string content) matching how clients render a bucket bound: 1.0 -> "1.0" or "1".

    The dot is escaped for the regex and the backslash again for the PromQL string literal.
    """
    forms = {repr(float(threshold)), f"{threshold:g}"}
    return "|".join(sorted(f.replace(".", "\\\\.") for f in forms))


def latency_error_expr(metric: str, threshold: float, selector: str, window: str) -> str:
    sel = f"{selector}, " if selector else ""
    good = f'sum(rate({metric}_bucket{{{sel}le=~"{le_matcher(threshold)}"}}[{window}]))'
    total = f"sum(rate({metric}_count{{{selector}}}[{window}]))"
    return f"1 - ({good} / {total})"


def ratio_error_expr(errors: str, total: str, window: str) -> str:
    e = errors.replace("$window", window)
    t = total.replace("$window", window)
    # A missing error series means zero errors, not "no data".
    return f"(({e}) or (({t}) * 0)) / ({t})"


def check_threshold(metric: str, threshold: float) -> None:
    """Fail early if a latency threshold is not a bucket bound of the proxy's histograms."""
    from inferobs.proxy.metrics import LAYOUTS

    family = {
        "gen_ai_client_operation_time_to_first_chunk_seconds": "ttft",
        "inferobs_time_per_output_token_seconds": "tpot",
        "gen_ai_client_operation_time_per_output_chunk_seconds": "tpot",
        "gen_ai_client_operation_duration_seconds": "duration",
    }.get(metric)
    if family is None:
        return
    for layout, fams in LAYOUTS.items():
        if not any(abs(b - threshold) < 1e-9 for b in fams[family]):
            raise ValueError(
                f"threshold {threshold} for {metric} is not a bucket bound in the {layout!r} layout"
            )


def generate(spec: dict) -> dict:
    scale = float(spec.get("window_scale", 1))
    prefix = spec.get("prefix", "slo")
    alert_name = spec.get("alert_name", "SLOErrorBudgetBurn")
    period = spec.get("period", "30d")
    scrape = parse_duration(spec.get("scrape_interval", "5s"))
    extra_labels = spec.get("labels", {}) or {}
    tiers = [Tier(**t) for t in spec["tiers"]] if spec.get("tiers") else list(WORKBOOK_TIERS)

    def scaled(d: str) -> str:
        return fmt_duration(parse_duration(d) * scale)

    windows: list[str] = []
    for t in tiers:
        for w in (t.short, t.long):
            sw = scaled(w)
            if parse_duration(sw) < 4 * scrape:
                raise ValueError(
                    f"window {w} scaled to {sw} spans fewer than 4 scrapes ({spec.get('scrape_interval')})"
                )
            if sw not in windows:
                windows.append(sw)
    windows.sort(key=parse_duration)
    period_s = scaled(period)
    base_w = windows[0]

    # One group per SLO with its alerts after its recording rules: rules in a group run in order, so
    # every alert sees this evaluation's SLI values. Separate groups run in no fixed order, which
    # made alerts read the previous evaluation's values (one interval late) depending on scheduling.
    groups = []
    for slo in spec["slos"]:
        name = slo["name"]
        obj = float(slo["objective"])
        if not 0 < obj < 1:
            raise ValueError(f"{name}: objective must be in (0, 1)")
        budget = 1 - obj
        labels = {"slo": name, **extra_labels}
        rules = [{"record": f"{prefix}:objective:ratio", "expr": f"vector({obj})", "labels": labels}]
        for w in windows:
            if "latency" in slo:
                lat = slo["latency"]
                check_threshold(lat["metric"], float(lat["threshold"]))
                expr = latency_error_expr(lat["metric"], float(lat["threshold"]), lat.get("selector", ""), w)
            elif "ratio" in slo:
                expr = ratio_error_expr(slo["ratio"]["errors"], slo["ratio"]["total"], w)
            else:
                raise ValueError(f"{name}: needs a 'latency' or 'ratio' SLI")
            rules.append({"record": f"{prefix}:sli_error:ratio_rate{w}", "expr": expr, "labels": labels})
        # The period window is averaged from the shortest recorded window (as Sloth does) so a 30-day
        # range never has to be computed from raw histogram buckets.
        rules.append(
            {
                "record": f"{prefix}:sli_error:ratio_rate{period_s}",
                "expr": f'avg_over_time({prefix}:sli_error:ratio_rate{base_w}{{slo="{name}"}}[{period_s}])',
                "labels": labels,
            }
        )
        for t in tiers:
            lw, sw = scaled(t.long), scaled(t.short)
            thr = round(t.burn * budget, 10)
            rules.append(
                {
                    "alert": alert_name,
                    "expr": (
                        f'{prefix}:sli_error:ratio_rate{lw}{{slo="{name}"}} > {thr}\n'
                        f'and\n{prefix}:sli_error:ratio_rate{sw}{{slo="{name}"}} > {thr}'
                    ),
                    "for": scaled(t.for_),
                    "labels": {
                        "severity": t.severity,
                        "slo": name,
                        "long_window": lw,
                        "short_window": sw,
                        "burn_rate": f"{t.burn:g}",
                        **extra_labels,
                    },
                    "annotations": {
                        "summary": f"{name} SLO is burning its error budget at {t.burn:g}x or faster",
                        "description": (
                            f"{slo.get('description', name)} Error ratio over {lw} is "
                            "{{ $value | humanizePercentage }} against a budget of "
                            f"{budget:.4%} ({obj:.2%} objective over {period})."
                        ),
                    },
                }
            )
        groups.append({"name": f"{prefix}-{name}", "rules": rules})
    return {"groups": groups}


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data) -> bool:  # no &id001 anchors in generated rules
        return True


def _str_presenter(dumper, data):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _str_presenter)


def render(spec_path: str | Path) -> str:
    spec = yaml.safe_load(Path(spec_path).read_text())
    header = (
        f"# GENERATED by `inferobs slo gen --spec {spec_path}`. Do not edit by hand.\n"
        f"# window_scale={spec.get('window_scale', 1)}\n"
    )
    return header + yaml.dump(generate(spec), Dumper=_Dumper, sort_keys=False, width=200)
