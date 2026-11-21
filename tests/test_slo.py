from pathlib import Path

import pytest
import yaml

from inferobs.proxy.metrics import LAYOUTS
from inferobs.slo.gen import fmt_duration, generate, le_matcher, parse_duration, render

ROOT = Path(__file__).resolve().parent.parent

SPEC = {
    "period": "30d",
    "slos": [
        {
            "name": "avail",
            "objective": 0.995,
            "ratio": {"errors": 'sum(rate(x_count{e!=""}[$window]))', "total": "sum(rate(x_count[$window]))"},
        },
        {
            "name": "ttfc",
            "objective": 0.99,
            "latency": {"metric": "gen_ai_client_operation_time_to_first_chunk_seconds", "threshold": 1.0},
        },
    ],
}


def test_durations_round_trip():
    assert parse_duration("1h") == 3600 and parse_duration("2.5m") == 150
    assert [fmt_duration(s) for s in (25, 150, 300, 7200, 86400 * 3)] == ["25s", "150s", "5m", "2h", "3d"]


def test_le_matcher_is_a_valid_promql_string_regex():
    assert le_matcher(1.0) == "1|1\\\\.0"  # renders as le=~"1|1\\.0" in the rule file
    assert le_matcher(0.05) == "0\\\\.05"


def test_workbook_tiers_and_thresholds():
    out = generate(SPEC)
    alerts = out["groups"][-1]["rules"]
    assert len(alerts) == 8  # 4 tiers x 2 SLOs
    fast = next(a for a in alerts if a["labels"]["slo"] == "avail" and a["labels"]["long_window"] == "1h")
    assert fast["labels"]["short_window"] == "5m" and fast["labels"]["severity"] == "page"
    assert "> 0.072" in fast["expr"]  # 14.4 x 0.5% budget
    assert fast["for"] == "2m"
    records = {r["record"] for g in out["groups"][:-1] for r in g["rules"]}
    assert {
        "slo:sli_error:ratio_rate5m",
        "slo:sli_error:ratio_rate3d",
        "slo:sli_error:ratio_rate30d",
        "slo:objective:ratio",
    } <= records


def test_window_scale_compresses_windows_and_for():
    out = generate({**SPEC, "window_scale": 1 / 12, "prefix": "slo_fast"})
    fast = next(a for a in out["groups"][-1]["rules"] if a["labels"]["burn_rate"] == "14.4")
    assert (fast["labels"]["long_window"], fast["labels"]["short_window"], fast["for"]) == (
        "5m",
        "25s",
        "10s",
    )
    assert "slo_fast:sli_error:ratio_rate5m" in fast["expr"]


def test_rejects_too_short_windows_and_bad_thresholds():
    with pytest.raises(ValueError, match="fewer than 4 scrapes"):
        generate({**SPEC, "window_scale": 1 / 120})
    bad = {
        **SPEC,
        "slos": [
            {
                "name": "t",
                "objective": 0.99,
                "latency": {
                    "metric": "gen_ai_client_operation_time_to_first_chunk_seconds",
                    "threshold": 0.7,
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="not a bucket bound"):
        generate(bad)


def test_ratio_sli_reads_zero_without_error_series():
    out = generate(SPEC)
    rec = next(r for r in out["groups"][0]["rules"] if r["record"] == "slo:sli_error:ratio_rate5m")
    assert " or ((sum(rate(x_count[5m]))) * 0)" in rec["expr"]


def test_round_slo_thresholds_are_bucket_bounds_in_every_layout():
    for layout in LAYOUTS.values():
        assert 1.0 in layout["ttft"] and 0.05 in layout["tpot"]


@pytest.mark.parametrize(
    "spec,out",
    [
        ("slo/slo.yaml", "deploy/prometheus/rules/slo.yaml"),
        ("slo/slo-fast.yaml", "deploy/lab-rules/slo-fast.yaml"),
    ],
)
def test_committed_rules_are_current(spec, out):
    assert (ROOT / out).read_text() == render(spec), f"regenerate {out} (make rules / make lab-rules)"
    yaml.safe_load((ROOT / out).read_text())
