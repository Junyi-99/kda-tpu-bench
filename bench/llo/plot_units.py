"""Two-panel comparison of where the kernel spends the machine.

Usage: plot_units.py <out.png> <mix_json> [<mix_json> ...] -- <counter_json> [...]

Left panel  : instruction issue mix (denominator = instructions issued)
Right panel : hardware counter utilization, bar = mean, cap = peak sample

The two panels come from different captures on purpose: the instruction mix is
read from a large shape where all three implementations sit in the same
1M-event window (shares stay comparable), and the counters from a small shape
where sampling is dense enough to be meaningful. Counter series with too few
samples are drawn as "n/a" rather than as zero — a truncated trace is missing
data, not an idle unit.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

COLOR = {"original": "#2a78d6", "mxu_port": "#eb6834", "optimized": "#1baf7a"}
LABEL = {"original": "original (all switches off)",
         "mxu_port": "mxu_port (+safe_gate)",
         "optimized": "optimized (all 5 switches)"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
MIX_UNITS = ["MXU", "VALU", "VLD", "VST", "XLU", "EUP", "SALU"]
CTR_UNITS = [("MXU", "MXU\n(busy)"), ("Vector ALU", "Vector\nALU"),
             ("Scalar ALU", "Scalar\nALU"), ("Vector Load", "Vector\nLoad"),
             ("Vector Store", "Vector\nStore"), ("XLU", "XLU")]


def read_mix(paths):
    out = {}
    for p in paths:
        d = json.load(open(p))
        mix = d["instruction_mix"]
        get = lambda u: mix.get(u, {}).get("share_pct", 0.0)
        out[d["impl"]] = {"MXU": round(get("MXU0") + get("MXU1"), 1),
                          **{u: get(u) for u in MIX_UNITS if u != "MXU"}}
    return out


def read_counters(paths):
    mean, peak = {}, {}
    for p in paths:
        d = json.load(open(p))
        c = d["counters"]
        m, q = [], []
        for key, _ in CTR_UNITS:
            row = c.get(f"{key} :: % util")
            if not row or not row["enough_samples"]:
                m.append(np.nan); q.append(np.nan); continue
            m.append(row["busy_mean"] if key == "MXU" and row["busy_mean"] else row["mean"])
            q.append(row["peak"])
        mean[d["impl"]] = m
        peak[d["impl"]] = q
    return mean, peak


def draw(ax, units, data, peaks, title, ylab, ylim=None):
    x = np.arange(len(units))
    w = 0.26
    for i, impl in enumerate(COLOR):
        if impl not in data:
            continue
        pos = x + (i - 1) * w
        vals = np.array(data[impl], dtype=float)
        ax.bar(pos, np.nan_to_num(vals), w * 0.92, color=COLOR[impl], label=LABEL[impl], zorder=3)
        pk = np.array(peaks[impl], dtype=float) if peaks else None
        for j, (xi, v) in enumerate(zip(pos, vals)):
            if np.isnan(v):
                ax.text(xi, 1.5, "n/a", ha="center", va="bottom", fontsize=7.5,
                        color=MUTED, rotation=90, zorder=4)
                continue
            if pk is not None and not np.isnan(pk[j]) and pk[j] > v:
                ax.vlines(xi, v, pk[j], color=COLOR[impl], lw=1.2, alpha=0.55, zorder=4)
                ax.hlines(pk[j], xi - w * 0.34, xi + w * 0.34, color=COLOR[impl], lw=2.0, zorder=5)
                ax.text(xi, pk[j] + 1.0, f"{round(pk[j], 1):g}", ha="center", va="bottom",
                        fontsize=6.5, color=COLOR[impl], zorder=7)
            if v >= 12:
                ax.text(xi, v - 1.2, f"{round(v, 1):g}", ha="center", va="top", fontsize=7,
                        color="white", fontweight="bold", zorder=7)
            else:
                ax.text(xi, v + 0.8, f"{round(v, 1):g}", ha="center", va="bottom", fontsize=7,
                        color=INK, zorder=7)
    ax.set_xticks(x)
    ax.set_xticklabels(units, fontsize=9)
    ax.set_ylabel(ylab, fontsize=10, color=MUTED)
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9c4")
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    out_png = sys.argv[1]
    rest = sys.argv[2:]
    sep = rest.index("--")
    mix = read_mix(rest[:sep])
    ctr_mean, ctr_peak = read_counters(rest[sep + 1:])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    draw(axes[0], MIX_UNITS, {k: [v[u] for u in MIX_UNITS] for k, v in mix.items()},
         None, "Instruction issue mix", "% of issued instructions")
    draw(axes[1], [lbl for _, lbl in CTR_UNITS], ctr_mean, ctr_peak,
         "Hardware counter utilization", "% util", ylim=(0, 100))
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper right")
    axes[1].legend(handles=[Line2D([0], [0], color=MUTED, lw=2.0, label="peak (max sample)")],
                   fontsize=8.5, frameon=False, loc="upper right")
    fig.suptitle("KDA prefill kernel: where the machine spends its issue slots and units",
                 fontsize=12, color=INK, y=1.0)
    fig.text(0.5, -0.03,
             "bar = mean over samples, cap = peak sample; n/a = trace hit XProf's 1M-event cap, "
             "leaving too few counter samples to report.",
             ha="center", fontsize=8.5, color=MUTED, style="italic")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", out_png)


if __name__ == "__main__":
    main()
