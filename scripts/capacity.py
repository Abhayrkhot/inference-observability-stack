#!/usr/bin/env python3
"""Rough engine capacity probe, used to place the saturation ramp and the steady-load rate.

capacity.py --model M --out results/T/capacity.json   # closed loop at 64 workers, 45 s
capacity.py --rates-from results/T/capacity.json      # ramp rates as fractions of capacity
capacity.py --steady-from results/T/capacity.json     # rate for agreement/faults (meets the SLOs)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FRACTIONS = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--url", default="http://127.0.0.1:28000")
    ap.add_argument("--out")
    ap.add_argument("--rates-from")
    ap.add_argument("--steady-from")
    ap.add_argument("--prompt", default="lognormal:256:0.6:16:1024")
    ap.add_argument("--output", default="lognormal:128:0.6:8:512")
    a = ap.parse_args()
    if a.out:
        from inferobs.exp.common import run_loadgen, runs_dir

        rd = runs_dir("capacity")
        s = run_loadgen(
            a.url, a.model, rd, "probe", concurrency=64, duration=45, prompt=a.prompt, output=a.output
        )
        cap = {
            "closed_loop_c64_req_per_s": s["req_per_s"],
            "output_tok_per_s": s["output_tok_per_s"],
            "ttft_p50_s": s["ttft_s"]["p50"],
            "errors": s["errors"],
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(cap, indent=2) + "\n")
        print(json.dumps(cap))
    elif a.rates_from:
        c = json.loads(Path(a.rates_from).read_text())["closed_loop_c64_req_per_s"]
        print(",".join(f"{max(0.2, round(c * f, 1)):g}" for f in FRACTIONS))
    elif a.steady_from:
        cap_path = Path(a.steady_from)
        c = json.loads(cap_path.read_text())["closed_loop_c64_req_per_s"]
        rate = round(c * 0.4, 1)
        sat = cap_path.with_name("saturation.csv")
        if sat.exists():  # half of the highest offered rate that still met both SLOs
            ok = [float(r["offered_rps"]) for r in csv.DictReader(sat.open()) if r["meets_slo"] == "True"]
            if ok:
                rate = round(max(ok) * 0.5, 1)
        print(f"{max(rate, 0.5):g}")


if __name__ == "__main__":
    main()
