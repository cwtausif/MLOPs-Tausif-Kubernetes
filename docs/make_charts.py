"""Build the report charts from real run outputs (data/, models/, docs/evidence/)."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
BLUE, ORANGE, GREEN, RED = "#2563eb", "#ea580c", "#16a34a", "#dc2626"

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
    }
)


def forecast_chart():
    clean = json.loads((ROOT / "data/clean.json").read_text())["data"]
    df = pd.DataFrame(clean)[["date", "close"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.tail(75)
    m = json.loads((ROOT / "models/metrics.json").read_text())
    p = json.loads((ROOT / "models/predictions.json").read_text())

    test_dates = pd.date_range(m["test_start"], m["test_end"])
    f_dates = pd.to_datetime([x["date"] for x in p["predictions"]])
    f_vals = [x["close"] for x in p["predictions"]]

    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=200)
    ax.plot(df["date"], df["close"], color=INK, lw=1.4, label="Actual close")
    ax.plot(
        test_dates,
        m["test_predicted"],
        color=ORANGE,
        lw=1.6,
        ls="--",
        marker="o",
        ms=3,
        label=f"Back-test (last 7 days) — MAPE {m['mape']:.2f}%",
    )
    ax.plot(
        [df["date"].iloc[-1], *f_dates],
        [df["close"].iloc[-1], *f_vals],
        color=BLUE,
        lw=1.8,
        marker="o",
        ms=3,
        label="7-day forecast (served by /predict)",
    )
    ax.axvspan(test_dates[0], test_dates[-1], color=ORANGE, alpha=0.07, lw=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title(
        f"BTC-USD close: history, back-test and forecast (model {m['version']})",
        loc="left",
        fontsize=10,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "forecast_chart.png")
    plt.close(fig)


def hpa_chart():
    rows = list(csv.DictReader(open(ROOT / "docs/evidence/hpa_timeline.csv")))
    t = [int(r["t"]) / 60 for r in rows]
    rep = [int(r["replicas"] or 0) for r in rows]
    ready = [int(r.get("ready") or 0) for r in rows]
    cpu = [float(r["cpu_pct"]) if r["cpu_pct"] else None for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=200)
    ax.axvspan(12 / 60, 252 / 60, color=RED, alpha=0.06, lw=0)
    ax.text(2.2, 0.4, "Locust: 150 users, 4 min", color=RED, fontsize=8, ha="center")
    ax.step(t, rep, where="post", color=BLUE, lw=2, label="HPA replicas")
    ax.step(t, ready, where="post", color=GREEN, lw=1.2, ls="--", label="Ready pods")
    ax.set_ylim(0, 9.5)
    ax.set_ylabel("pods")
    ax.set_xlabel("minutes since test start")
    ax2 = ax.twinx()
    ax2.plot(
        [x for x, c in zip(t, cpu) if c is not None],
        [c for c in cpu if c is not None],
        color=ORANGE,
        lw=1.4,
        marker="o",
        ms=2.5,
        label="Avg CPU (% of 100m request)",
    )
    ax2.axhline(60, color=ORANGE, lw=0.8, ls=":")
    ax2.text(t[-1], 66, "target 60%", color=ORANGE, fontsize=7, ha="right")
    ax2.set_ylim(0, 420)
    ax2.set_ylabel("CPU % of request", color=ORANGE)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.set_title(
        "Horizontal Pod Autoscaler under load (kubectl get hpa, sampled every 10 s)",
        loc="left",
        fontsize=10,
        color=INK,
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "hpa_chart.png")
    plt.close(fig)


if __name__ == "__main__":
    forecast_chart()
    hpa_chart()
    print("charts written to", OUT)
