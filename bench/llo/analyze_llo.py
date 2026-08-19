"""Turn an instrumented trace into per-unit instruction counts + counter stats.

Usage: analyze_llo.py <trace_dir_or_xplane.pb> <impl> <out.json>

Two things come out, and they answer different questions:

  lanes    — how many instructions each unit issued (MXU0/MXU1/VALU/VLD/VST/
             XLU/EUP/SALU). Denominator is instruction count; the units split
             100% between them.
  counters — the hardware counters' "% util" samples, i.e. what fraction of
             cycles a unit was busy. Denominator is time, so these can sum
             past 100% (units run in parallel).

Caveat worth respecting: XProf's trace-viewer conversion caps at 1M events.
The `original` path issues instructions densely enough to hit that cap, which
leaves its counter lanes with only a handful of samples — `n_samples` is
reported so callers can drop under-sampled series instead of plotting noise.
"""

import collections
import glob
import gzip
import json
import os
import statistics
import sys

from xprof.convert import raw_to_tool_data

TRUNCATION_CAP = 1_000_000
MIN_SAMPLES = 5  # below this a counter series is reported but flagged


def load_events(path):
    if os.path.isdir(path):
        hits = glob.glob(os.path.join(path, "**", "*.xplane.pb"), recursive=True)
        if not hits:
            raise SystemExit(f"no *.xplane.pb under {path}")
        path = hits[0]
    out = raw_to_tool_data.xspace_to_tool_data([path], "trace_viewer^", {})
    data = out[0] if isinstance(out, tuple) else out
    if isinstance(data, bytes):
        try:
            data = gzip.decompress(data)
        except Exception:
            pass
        data = data.decode("utf-8", "replace")
    return json.loads(data).get("traceEvents", []), path


def main():
    src, impl, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    events, xplane = load_events(src)

    tids = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name" and "pid" in e and "tid" in e:
            tids[(e["pid"], e["tid"])] = e["args"]["name"]

    lanes = collections.Counter()
    top_insts = collections.defaultdict(collections.Counter)
    samples = collections.defaultdict(list)
    for e in events:
        if e.get("ph") != "X":
            continue
        lane = tids.get((e.get("pid"), e.get("tid")))
        if lane is None:
            continue
        if lane.endswith("Instructions"):
            unit = lane.split()[0]
            lanes[unit] += 1
            top_insts[unit][e.get("name", "?")] += 1
        elif lane == "_counters_":
            for k, v in (e.get("args") or {}).items():
                try:
                    samples[(e.get("name"), k)].append(float(v))
                except (TypeError, ValueError):
                    pass

    total_instr = sum(lanes.values())
    result = {
        "impl": impl,
        "xplane": os.path.basename(xplane),
        "total_events": len(events),
        "truncated": len(events) >= TRUNCATION_CAP - 1000,
        "instruction_mix": {
            u: {"count": n,
                "share_pct": round(n / total_instr * 100, 2) if total_instr else 0.0,
                "top": top_insts[u].most_common(6)}
            for u, n in lanes.most_common()
        },
        "counters": {},
    }
    for (name, metric), xs in sorted(samples.items()):
        busy = [x for x in xs if x > 0]
        result["counters"][f"{name} :: {metric}"] = {
            "n_samples": len(xs),
            "enough_samples": len(xs) >= MIN_SAMPLES,
            "mean": round(statistics.fmean(xs), 2),
            "busy_mean": round(statistics.fmean(busy), 2) if busy else None,
            "peak": round(max(xs), 2),
            "nonzero_pct": round(len(busy) / len(xs) * 100, 1),
        }

    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1)
    mxu = result["instruction_mix"].get("MXU0", {}).get("count", 0) + \
        result["instruction_mix"].get("MXU1", {}).get("count", 0)
    print(f"{impl}: {len(events)} events (truncated={result['truncated']}), "
          f"{total_instr} instructions, MXU share "
          f"{mxu / total_instr * 100 if total_instr else 0:.1f}% -> {out_path}")


if __name__ == "__main__":
    main()
