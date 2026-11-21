#!/usr/bin/env python3
"""Print a results CSV as an aligned table: scripts/peek.py FILE [COL=VALUE ...] [--cols a,b,c]."""

import csv
import sys

args = sys.argv[1:]
path = args.pop(0)
cols = None
filters = {}
for a in args:
    if a.startswith("--cols="):
        cols = a.split("=", 1)[1].split(",")
    elif "=" in a:
        k, v = a.split("=", 1)
        filters[k] = v
with open(path) as fh:
    rows = [r for r in csv.DictReader(fh) if all(r.get(k) == v for k, v in filters.items())]
if not rows:
    sys.exit("no rows")
cols = cols or list(rows[0])


def fmt(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v or ""
    return f"{f:.4g}"


table = [cols] + [[fmt(r.get(c)) for c in cols] for r in rows]
widths = [max(len(str(row[i])) for row in table) for i in range(len(cols))]
for row in table:
    print("  ".join(str(x).ljust(w) for x, w in zip(row, widths, strict=False)))
