#!/usr/bin/env python3
"""Plots (GFLOP/s, light and dark SVG) and Markdown tables of the results.

For each directory results/A-<order>_B-<order> (see run-benchmarks.sh):
gflops-light.svg, gflops-dark.svg and table.md (time and GFLOP/s of the
Mojo DGEMM).
"""

import pathlib

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

# file -> (label, colour light, colour dark, line style).  The colour is
# the implementation, the micro kernel is told apart by the height of the
# curves (SME2 several times NEON) and labelled in the plot.
SERIES = {
    "mojo-sme2.txt": ("Mojo", "#2a78d6", "#3987e5", "-"),
    "ulmblas-sme2.txt": ("ulmBLAS (C++)", "#eb6834", "#d95926", "--"),
    "accelerate.txt": ("Apple Accelerate", "#1baf7a", "#199e70", ":"),
    "mojo-neon.txt": ("Mojo", "#2a78d6", "#3987e5", "-"),
    "ulmblas-neon.txt": ("ulmBLAS (C++)", "#eb6834", "#d95926", "--"),
}

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  grid="#e4e3df", col=1),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7",
                 grid="#3a3936", col=2),
}


def read(path):
    """Benchmark output: title (first '#' line), {n: (seconds, GFLOP/s)}."""
    title, rows = None, {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            if title is None:
                title = line[1:].strip()
            continue
        f = line.split()
        if f:
            if f[3] != "ok":
                raise SystemExit(f"{path}: check failed for n = {f[0]}")
            rows[int(f[0])] = (float(f[1]), float(f[2]))
    return title, rows


def plot(d, theme, ymax):
    t = THEMES[theme]
    title, _ = read(d / "mojo-neon.txt")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    labelled = set()
    for name, style in SERIES.items():
        if not (d / name).exists():
            continue
        _, rows = read(d / name)
        n = sorted(rows)
        label = style[0] if style[0] not in labelled else None
        labelled.add(style[0])
        ax.plot(n, [rows[x][1] for x in n], style[3], color=style[t["col"]],
                linewidth=2, label=label)

    ax.annotate("SME2 micro kernels", xy=(4000, 0.9 * ymax), ha="right",
                color=t["ink2"], fontsize=10)
    ax.annotate("NEON micro kernels", xy=(4000, 0.16 * ymax), ha="right",
                color=t["ink2"], fontsize=10)
    ax.set_xlim(0, 4050)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("m = n = k", color=t["ink2"])
    ax.set_ylabel("GFLOP/s", color=t["ink2"])
    ax.set_title(title + "\none thread, Apple M5", color=t["ink"],
                 loc="left", fontsize=11)
    ax.grid(True, axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"])
    leg = ax.legend(loc="center right", frameon=False)
    for text in leg.get_texts():
        text.set_color(t["ink"])
    fig.tight_layout()
    fig.savefig(d / f"gflops-{theme}.svg", facecolor=t["surface"])
    plt.close(fig)


def table(d):
    _, neon = read(d / "mojo-neon.txt")
    _, sme = read(d / "mojo-sme2.txt")
    lines = [
        "| m = n = k | NEON: time (s) | NEON: GFLOP/s "
        "| SME2: time (s) | SME2: GFLOP/s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n in sorted(neon):
        tn, gn = neon[n]
        ts, gs = sme[n]
        lines.append(f"| {n} | {tn:.3e} | {gn:.1f} | {ts:.3e} | {gs:.1f} |")
    (d / "table.md").write_text("\n".join(lines) + "\n")


def main():
    dirs = sorted(p for p in RESULTS.iterdir() if p.is_dir())
    # one y scale for all plots
    top = max(r[1] for d in dirs for f in d.glob("*.txt")
              for r in read(f)[1].values())
    ymax = 100 * (int(top * 1.1) // 100 + 1)
    for d in dirs:
        for theme in THEMES:
            plot(d, theme, ymax)
        table(d)


if __name__ == "__main__":
    main()
