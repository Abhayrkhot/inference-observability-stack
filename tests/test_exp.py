from inferobs.exp.faults import EXPECTED, SILENT, alert_key
from inferobs.exp.overhead import summarize


def test_alert_key_scores_page_tiers_and_symptoms_only():
    fast = {"alertname": "SLOErrorBudgetBurnFast", "slo": "ttfc"}
    assert alert_key({**fast, "long_window": "5m"}) == "SLOErrorBudgetBurnFast:ttfc"
    assert alert_key({**fast, "long_window": "30m"}) == "SLOErrorBudgetBurnFast:ttfc@30m"
    assert alert_key({**fast, "long_window": "2h"}) is None  # ticket tier
    assert alert_key({"alertname": "SLOErrorBudgetBurn", "slo": "ttfc", "long_window": "1h"}) is None
    assert alert_key({"alertname": "EngineQueueBacklog", "model_name": "m"}) == "EngineQueueBacklog"


def test_negative_controls_are_not_also_expected():
    for scenario, silent in SILENT.items():
        assert not set(silent) & set(EXPECTED[scenario])


def test_overhead_summary_deltas_against_direct():
    def row(target, rep, ttft, tps, cpu=None):
        return {
            "target": target,
            "concurrency": 8,
            "repeat": rep,
            "errors": 0,
            "output_tok_per_s": tps,
            "ttft_p50_s": ttft,
            "ttft_p99_s": ttft * 2,
            "tpot_p50_s": 0.01,
            "itl_p99_s": 0.02,
            "e2e_p50_s": 1.0,
            "e2e_p99_s": 2.0,
            "proxy_cpu_util": cpu,
            "proxy_cpu_ms_per_req": cpu,
            "proxy_cpu_us_per_token": cpu,
        }

    rows = [row("direct", r, 0.100, 1000) for r in range(3)]
    rows += [row("proxy", r, 0.101 + 0.001 * r, 990, 0.2) for r in range(3)]
    out = {r["target"]: r for r in summarize(rows)}
    assert round(out["proxy"]["ttft_p50_delta_ms"], 3) == 2.0  # median 0.102 vs 0.100
    assert round(out["proxy"]["throughput_delta_pct"], 3) == -1.0
    assert "ttft_p50_delta_ms" not in out["direct"]
