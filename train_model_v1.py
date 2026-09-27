#!/usr/bin/env python3
"""
train_model_v1.py - first walk-forward model: can we predict which quarters will NOT beat?

Walk-forward validation (no shuffling across time):
  for each test year Y in 2019..2025: train on every report before 1 Jan Y, predict year Y.
  Results are pooled over all test years and also shown per year.

Models compared (all output a probability of not beating):
  base_overall   the training period's not-beat rate                    (naive baseline)
  base_sector    the training period's not-beat rate in the same sector (the bar to beat)
  logistic       logistic regression; gaps filled with TRAINING medians, extremes capped at
                 TRAINING 1st/99th percentiles, features standardised
  gbm            gradient-boosted trees (handles missing values itself)

Metrics
  Brier score    mean squared error of the probability (lower is better)
  Brier skill    1 - Brier / Brier(base_sector)  (> 0 means better than the sector baseline)
  ROC AUC        ranking quality (0.5 = random)
  PR AUC         precision-recall for not-beat quarters (baseline = not-beat share, ~21%)
  Top-decile     not-beat rate among the 10% of quarters the model finds riskiest

Inputs : data/processed/features_v2.csv
Outputs: reports/model_v1_results.txt, data/processed/model_v1_predictions.csv,
         reports/figures/model_v1_*.png, reports/figures/signal_*.png

Setup once (inside the .venv):  pip install scikit-learn matplotlib
Run from the project folder:    python train_model_v1.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
TEST_YEARS = range(2019, 2026)
SEED = 0

# palette (validated reference palette): blue = beat / model, orange = not beat, grays = ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

EXCLUDE = {"company_key", "sector", "report_date", "report_time", "eps_estimate",
           "not_beat", "outcome", "year"}


def load():
    df = pd.read_csv(PROC / "features_v2.csv", parse_dates=["report_date"])
    df["year"] = df["report_date"].dt.year
    feats = [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    for s in sorted(df["sector"].unique()):
        col = "sector_" + s.lower().replace(" ", "_")
        df[col] = (df["sector"] == s).astype(float)
        feats.append(col)
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    return df, feats


def fit_logistic(tr, te, feats):
    X, Z = tr[feats].copy(), te[feats].copy()
    lo, hi = X.quantile(0.01), X.quantile(0.99)
    med = X.median()
    X, Z = X.clip(lo, hi, axis=1).fillna(med), Z.clip(lo, hi, axis=1).fillna(med)
    mu, sd = X.mean(), X.std().replace(0, 1)
    X, Z = ((X - mu) / sd).fillna(0), ((Z - mu) / sd).fillna(0)
    m = LogisticRegression(C=0.05, max_iter=2000)
    m.fit(X, tr["not_beat"])
    return m.predict_proba(Z)[:, 1], pd.Series(m.coef_[0], index=feats)


def fit_gbm(tr, te, feats):
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=250,
                                       min_samples_leaf=60, l2_regularization=1.0,
                                       random_state=SEED)
    m.fit(tr[feats], tr["not_beat"])
    return m.predict_proba(te[feats])[:, 1]


def metrics(y, p, ref):
    b = brier_score_loss(y, p)
    top = p >= np.quantile(p, 0.9)
    return {"brier": b, "brier_skill": 1 - b / brier_score_loss(y, ref),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "top_decile_rate": y[top].mean()}


def main():
    df, feats = load()
    preds, coefs = [], []
    for yr in TEST_YEARS:
        tr, te = df[df["year"] < yr], df[df["year"] == yr]
        out = te[["company_key", "sector", "report_date", "not_beat", "outcome"]].copy()
        out["base_overall"] = tr["not_beat"].mean()
        out["base_sector"] = te["sector"].map(tr.groupby("sector")["not_beat"].mean())
        out["logistic"], coef = fit_logistic(tr, te, feats)
        out["gbm"] = fit_gbm(tr, te, feats)
        out["test_year"] = yr
        preds.append(out)
        coefs.append(coef.rename(yr))
        print(f"  {yr}: trained on {len(tr):,} reports, tested on {len(te):,}")
    P = pd.concat(preds)
    y, ref = P["not_beat"].values, P["base_sector"].values
    models = ["base_overall", "base_sector", "logistic", "gbm"]

    line = "=" * 78
    lines = [line, f"MODEL V1 - walk-forward {min(TEST_YEARS)}-{max(TEST_YEARS)}, "
             f"{len(P):,} test reports, not-beat share {y.mean():.1%}", line, "",
             f"{'model':14} {'Brier':>8} {'skill vs sector':>16} {'ROC AUC':>8} {'PR AUC':>7} "
             f"{'top-decile not-beat':>20}"]
    for m in models:
        r = metrics(y, P[m].values, ref)
        lines.append(f"{m:14} {r['brier']:>8.4f} {r['brier_skill']:>+15.1%} {r['roc_auc']:>8.3f} "
                     f"{r['pr_auc']:>7.3f} {r['top_decile_rate']:>19.1%}")
    lines += ["", "Brier score by test year (lower is better):",
              f"{'year':6}" + "".join(f"{m:>14}" for m in models)]
    for yr, g in P.groupby("test_year"):
        lines.append(f"{yr:<6}" + "".join(f"{brier_score_loss(g['not_beat'], g[m]):>14.4f}" for m in models))
    for m in ("logistic", "gbm"):
        beats = sum(brier_score_loss(g["not_beat"], g[m]) < brier_score_loss(g["not_beat"], g["base_sector"])
                    for _, g in P.groupby("test_year"))
        lines.append(f"{m} beats the sector baseline in {beats} of {len(TEST_YEARS)} test years.")

    C = pd.concat(coefs, axis=1).mean(axis=1).sort_values()
    lines += ["", "Logistic regression - average standardised coefficient (+ = more likely NOT to beat):"]
    for n, v in pd.concat([C.head(6), C.tail(6)]).items():
        lines.append(f"  {v:+.3f}  {n}")

    # calibration: predicted vs actual by decile (logistic regression, the better model)
    P["decile"] = pd.qcut(P["logistic"].rank(method="first"), 10, labels=False) + 1
    cal = P.groupby("decile").agg(predicted=("logistic", "mean"), actual=("not_beat", "mean"), n=("not_beat", "size"))
    lines += ["", "Calibration (logistic): predicted vs actual not-beat rate by risk decile",
              cal.round(3).to_string()]
    text = "\n".join(lines)
    print("\n" + text)

    REPORTS.mkdir(exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "model_v1_results.txt").write_text(text)
    P.drop(columns=["decile"]).to_csv(PROC / "model_v1_predictions.csv", index=False,
                                      date_format="%Y-%m-%d")
    charts(df, P, cal, models)
    print(f"\nSaved reports/model_v1_results.txt, data/processed/model_v1_predictions.csv and charts in {FIGS}")


# ----------------------------------------------------------------------------- charts
def style(ax, title, subtitle=None):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=22)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2)


def charts(df, P, cal, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. signal: not-beat rate by quintile for the six strongest features
    base = df["not_beat"].mean()
    show = [("beat_rate_shrunk", "Beat rate (blended with sector)"),
            ("mean_surprise_ps_4q", "Average past surprise"),
            ("dist_52w_high", "Price vs 52-week high"),
            ("op_margin_chg_yoy", "Operating-margin change vs last year"),
            ("inventory_minus_sales_growth", "Inventory growth minus sales growth"),
            ("reporting_lag_days", "Days from quarter end to report")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharey=True)
    for ax, (col, label) in zip(axes.flat, show):
        q = pd.qcut(df[col].rank(method="first"), 5, labels=False)
        r = df.groupby(q)["not_beat"].mean() * 100
        ax.bar(range(1, 6), r.values, color=BLUE, width=0.62)
        ax.axhline(base * 100, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        for x, v in zip(range(1, 6), r.values):
            ax.text(x, v + 0.8, f"{v:.0f}%", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(range(1, 6), ["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0, labelsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        ax.set_title(label, fontsize=10, color=INK, loc="left")
    axes[0, 0].set_ylabel("Share of quarters not beating (%)", color=INK2, fontsize=9)
    axes[1, 0].set_ylabel("Share of quarters not beating (%)", color=INK2, fontsize=9)
    fig.suptitle("Which quarters fail to beat? Not-beat rate by feature quintile", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.925, f"5,380 reports, 2018-2025. Dashed line = overall rate ({base:.0%}).",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIGS / "signal_quintiles.png", dpi=160, facecolor="white")
    plt.close(fig)

    # 2. Brier skill vs the sector baseline, by year
    fig, ax = plt.subplots(figsize=(8, 4))
    yrs = sorted(P["test_year"].unique())
    w = 0.36
    for i, (m, col, name) in enumerate([("logistic", AQUA, "Logistic regression"), ("gbm", BLUE, "Gradient-boosted trees")]):
        sk = [100 * (1 - brier_score_loss(g["not_beat"], g[m]) / brier_score_loss(g["not_beat"], g["base_sector"]))
              for _, g in P.groupby("test_year")]
        ax.bar(np.arange(len(yrs)) + (i - 0.5) * w, sk, width=w - 0.04, color=col, label=name)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(range(len(yrs)), yrs)
    ax.set_ylabel("Brier skill vs sector baseline (%)", color=INK2, fontsize=9)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.18)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    style(ax, "Does the model beat the sector base rate?", "Above 0 = better probabilities than the sector baseline, out of sample each year")
    fig.tight_layout()
    fig.savefig(FIGS / "model_v1_skill_by_year.png", dpi=160, facecolor="white")
    plt.close(fig)

    # 3. risk deciles: actual not-beat rate by predicted-risk decile (GBM)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cal.index, cal["actual"] * 100, color=BLUE, width=0.62, label="Actual")
    ax.plot(cal.index, cal["predicted"] * 100, color=ORANGE, marker="o", markersize=5,
            linewidth=2, label="Predicted")
    ax.axhline(P["not_beat"].mean() * 100, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_xticks(cal.index, [f"D{i}" for i in cal.index])
    ax.set_xlabel("Predicted risk decile (D1 = safest, D10 = riskiest)", color=INK2, fontsize=9)
    ax.set_ylabel("Share not beating (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax, "The riskiest decile fails to beat far more often",
          "Logistic regression, walk-forward 2019-2025; dashed line = average")
    fig.tight_layout()
    fig.savefig(FIGS / "model_v1_risk_deciles.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
