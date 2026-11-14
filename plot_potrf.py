#!/usr/bin/env python3
"""Plots (GFLOP/s, light and dark SVG) and Markdown tables of the DPOTRF
results in results-potrf/<order>/ (see run-potrf-benchmarks.sh)."""

import pathlib

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results-potrf"
ORDERS = {"col": "column major", "row": "row major"}

# file -> (label, colour light, colour dark, line style)
SERIES = {
    "mojo.txt": ("Mojo potrf, Mojo BLAS (SME2)", "#2a78d6", "#3987e5", "-"),
    "mojo-accelerate-blas.txt": ("Mojo potrf, BLAS of Accelerate",
                                 "#2a78d6", "#3987e5", "--"),
    "accelerate.txt": ("Accelerate dpotrf", "#1baf7a", "#199e70", ":"),
    "../numax.txt": ("numax (float64, row major)", "#9b51e0", "#b07cf0", "-"),
}

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  grid="#e4e3df", col=1),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7",
                 grid="#3a3936", col=2),
}


def read(path):
    """{n: GFLOP/s} of bench-potrf or of the numax benchmark."""
    rows = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if not f or line.startswith("#") or not f[0].isdigit():
            continue
        if f[3] != "ok":
            raise SystemExit(f"{path}: check failed for n = {f[0]}")
        rows[int(f[0])] = float(f[2])
    return rows


def plot(order, theme, ymax):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for name, (label, c1, c2, ls) in SERIES.items():
        p = RESULTS / order / name
        if not p.exists():
            continue
        rows = read(p)
        n = sorted(rows)
        ax.plot(n, [rows[x] for x in n], ls, color=c1 if t["col"] == 1
                else c2, linewidth=2, label=label)
    ax.set_xlim(0, 4050)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("n", color=t["ink2"])
    ax.set_ylabel("GFLOP/s", color=t["ink2"])
    ax.set_title(f"DPOTRF, lower, {ORDERS[order]}\nApple M5", color=t["ink"],
                 loc="left", fontsize=11)
    ax.grid(True, axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"])
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                    ncol=2, frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t["ink"])
    fig.tight_layout()
    fig.savefig(RESULTS / order / f"gflops-{theme}.svg",
                facecolor=t["surface"])
    plt.close(fig)


def table(order, sizes):
    cols = [(label, name) for name, (label, *_) in SERIES.items()
            if (RESULTS / order / name).exists()]
    data = {name: read(RESULTS / order / name) for _, name in cols}
    lines = ["| n | " + " | ".join(h for h, _ in cols) + " |",
             "|---:|" + "---:|" * len(cols)]
    for n in sizes:
        cells = [f"{data[f][n]:.0f}" if n in data[f] else "" for _, f in cols]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    (RESULTS / order / "table.md").write_text("\n".join(lines) + "\n")


def main():
    files = [p for o in ORDERS for p in (RESULTS / o).glob("*.txt")]
    files += [RESULTS / "numax.txt"] if (RESULTS / "numax.txt").exists() else []
    top = max(g for f in files for g in read(f).values())
    ymax = 100 * (int(top * 1.05) // 100 + 1)
    for order in ORDERS:
        if not (RESULTS / order).is_dir():
            continue
        for theme in THEMES:
            plot(order, theme, ymax)
        ns = sorted(read(RESULTS / order / "mojo.txt"))
        table(order, [n for n in (100, 250, 500, 750, 1000, 1500, 2000, 3000,
                                  4000) if n in ns])


if __name__ == "__main__":
    main()
