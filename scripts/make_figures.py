"""Build report figures from reports/tables/ (run after run_analysis.py and run_mlb_comparison.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swing import data  # noqa: E402

TAB, FIG = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def model_ladder():
    t = pd.read_csv(TAB / "model_table.csv")
    t = t[t["model"] != "B0 mean"].reset_index(drop=True)
    y = np.arange(len(t))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hlines(y, t["r2_ci_low"], t["r2_ci_high"], color=BLUE, lw=2)
    ax.plot(t["r2_grouped"], y, "o", ms=8, color=BLUE, label="New hitters (grouped CV)")
    ax.plot(t["r2_random"], y, "o", ms=8, mfc=SURFACE, mec=ORANGE, mew=2, label="Random swing split (leaky)")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, t["model"])
    ax.set_xlabel("Out-of-sample R² for maximum bat speed")
    ax.set_title("Bat speed explained by each block of information")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2, frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_model_ladder.png", dpi=200)
    plt.close(fig)


def variance_split(res):
    labels = {data.OUTCOME: "Bat speed", data.EV: "Exit velocity"}
    fig, ax = plt.subplots(figsize=(7.6, 2.2))
    for i, key in enumerate(labels):
        q = res["q1_icc"][key]
        b, w = q["between_sd"] ** 2, q["within_sd"] ** 2
        share = b / (b + w)
        yi = 1 - i
        ax.barh(yi, share, color=BLUE, height=0.5)
        ax.barh(yi, 1 - share, left=share, color=ORANGE, height=0.5, edgecolor=SURFACE, linewidth=2)
        ax.text(share / 2, yi, f"Between hitters {share:.0%}", ha="center", va="center", color="white",
                fontweight="bold", fontsize=9)
        ax.text(1.01, yi, f"Within {1 - share:.0%}", ha="left", va="center", color=INK, fontsize=9)
    ax.set_yticks([1, 0], list(labels.values()))
    ax.set_xlim(0, 1.18)
    ax.set_xticks([])
    ax.grid(False)
    ax.set_title("Where the variation lives (ICC from random-intercept models)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_variance_split.png", dpi=200)
    plt.close(fig)


def ev_slopes():
    t = pd.read_csv(TAB / "q3a_ev_mixed.csv").set_index("term")
    rows = [("Bat speed, between hitters", f"{data.OUTCOME}_between", BLUE),
            ("Bat speed, within a hitter", f"{data.OUTCOME}_within", ORANGE)]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for i, (_, term, c) in enumerate(rows):
        r = t.loc[term]
        ax.hlines(i, r["ci_low"], r["ci_high"], color=c, lw=2)
        ax.plot(r["estimate"], i, "o", ms=8, color=c)
        ax.text(r["estimate"], i + 0.22, f"{r['estimate']:.2f} mph EV per mph", color=INK2, ha="center")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks([0, 1], [r[0] for r in rows])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Exit velocity gained per +1 mph bat speed (adjusted)")
    ax.set_title("How much bat speed turns into exit velocity")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_ev_slopes.png", dpi=200)
    plt.close(fig)


def collision_q():
    t = pd.read_csv(TAB / "q4_collision_q_lab_vs_mlb.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for k, (setting, c, lab) in enumerate((("lab", ORANGE, "Lab (machine, ~59 mph)"),
                                           ("mlb", BLUE, "MLB 2025 balls in play"))):
        s = t[t.setting == setting].sort_values("tau")
        x = np.arange(len(s)) + (k - 0.5) * 0.18
        ax.vlines(x, s["ci_low"], s["ci_high"], color=c, lw=2)
        ax.plot(x, s["q"], "o", ms=8, color=c, label=lab)
    ax.axhspan(0.2, 0.25, color=GRID, zorder=0, label="Typical squared-up q (Nathan 2003)")
    ax.set_xticks(np.arange(3), ["median (τ=0.50)", "τ=0.75", "near squared-up (τ=0.90)"])
    ax.set_ylabel("Collision efficiency q")
    ax.set_title("Collision efficiency in the lab and in MLB")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_collision_q.png", dpi=200)
    plt.close(fig)


def physics_scatter():
    lab = data.load().dropna(subset=[data.EV, "pitch_speed_mph"])
    qt = pd.read_csv(TAB / "q4_collision_q_lab_vs_mlb.csv").set_index(["setting", "tau"])
    sc = pd.read_parquet(ROOT / "data" / "raw" / "statcast_2025.parquet",
                         columns=["description", "events", "bat_speed", "launch_speed", "release_speed"])
    sc = sc[(sc.description == "hit_into_play") & ~sc.events.fillna("").str.contains("bunt")].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    panels = [(axes[0], "lab", lab["sweet_spot_speed_mph"] + lab["pitch_speed_mph"],
               lab[data.EV] - lab["sweet_spot_speed_mph"], "Lab swings (n={:,})"),
              (axes[1], "mlb", sc["bat_speed"] + 0.92 * sc["release_speed"],
               sc["launch_speed"] - sc["bat_speed"], "MLB 2025 balls in play (n={:,})")]
    for ax, setting, x, y, title in panels:
        if setting == "lab":
            ax.plot(x, y, "o", ms=3.5, color=ORANGE, alpha=0.5, mec="none")
        else:
            ax.hexbin(x, y, gridsize=60, cmap="Blues", mincnt=5, extent=(80, 200, -60, 60), linewidths=0)
        xs = np.linspace(x.min(), x.max(), 50)
        for tau, ls in ((0.9, "-"), (0.5, "--")):
            ax.plot(xs, qt.loc[(setting, tau), "q"] * xs, ls, color=INK, lw=1.6, label=f"τ = {tau}")
        ax.set_title(title.format(len(x)), fontsize=10)
        ax.set_xlabel("Bat speed + pitch speed (mph)")
        ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    axes[0].set_ylabel("Exit velocity − bat speed (mph)")
    axes[0].set_ylim(-60, 60)
    fig.suptitle("Slope of the line = collision efficiency q", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_physics_scatter.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    with open(TAB / "results.json") as f:
        res = json.load(f)
    model_ladder()
    variance_split(res)
    ev_slopes()
    if (TAB / "q4_collision_q_lab_vs_mlb.csv").exists():
        collision_q()
        physics_scatter()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
