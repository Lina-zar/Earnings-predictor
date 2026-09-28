#!/usr/bin/env python3
"""
part3_analysis.py - Part 3: why does the model work, where does it fail, and does it
matter for the stock price?

Everything is walk-forward (test years 2019-2025, each trained only on earlier years),
using the logistic regression from train_model_v1.py.

  1. Feature groups   - Brier skill with each group removed, and with each group alone
  2. Single features  - how much worse the model gets when one feature is dropped
  3. Where it works   - skill by sector, report timing, company size and track record
  4. Robustness       - ties counted as beats, no fundamentals, later years only, and a
                        fair test of a smaller model (features chosen on 2019-21, tested 2022-25)
  5. Money test       - 2-day stock reaction (vs S&P 500) by predicted-risk decile:
                        do the quarters the model flags actually fall?

Inputs : data/processed/features_v2.csv, feature_dictionary.csv, reactions.csv
         (build_reactions.py makes reactions.csv)
Outputs: reports/part3_results.txt, reports/figures/part3_*.png
Run    : python part3_analysis.py        (needs scikit-learn and matplotlib)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

import train_model_v1 as m1

ROOT = Path(__file__).resolve().parent
PROC, REPORTS = ROOT / "data" / "processed", ROOT / "reports"
FIGS = REPORTS / "figures"
YEARS = list(range(2019, 2026))
GROUP_NAMES = {"A": "Track record", "B": "Estimate demand", "C": "Fundamentals (SEC)",
               "D": "Market signals", "E": "Timing", "F": "Earnings season", "S": "Sector"}


def walk_forward(df, feats, target="not_beat", years=YEARS):
    out = []
    for yr in years:
        tr, te = df[df["year"] < yr], df[df["year"] == yr]
        p, _ = m1.fit_logistic(tr.assign(not_beat=tr[target]), te, feats)
        base = te["sector"].map(tr.groupby("sector")[target].mean())
        out.append(pd.DataFrame({"p": p, "base": base.values, "y": te[target].values}, index=te.index))
    return pd.concat(out)


def skill(P):
    return 1 - brier_score_loss(P["y"], P["p"]) / brier_score_loss(P["y"], P["base"])


def main():
    df, feats = m1.load()
    dic = pd.read_csv(PROC / "feature_dictionary.csv")
    group = dict(zip(dic["feature"], dic["group"]))
    group.update({f: "S" for f in feats if f.startswith("sector_") and f != "sector_prior"})
    feats = [f for f in feats if f in group]
    L = ["=" * 78, "PART 3 - MODEL ANALYSIS (logistic regression, walk-forward 2019-2025)", "=" * 78]

    full = walk_forward(df, feats)
    s_full = skill(full)
    L += [f"\nFull model: Brier skill vs sector baseline {s_full:+.1%}, "
          f"ROC AUC {roc_auc_score(full['y'], full['p']):.3f}, {len(feats)} features"]

    # 1. groups
    L += ["\n1. FEATURE GROUPS", f"{'group':22} {'n':>3} {'skill without':>14} {'loss':>7} {'skill alone':>12}"]
    grows = []
    for g, name in GROUP_NAMES.items():
        gf = [f for f in feats if group[f] == g]
        if not gf:
            continue
        without = skill(walk_forward(df, [f for f in feats if group[f] != g]))
        alone = skill(walk_forward(df, gf + ([f for f in feats if group[f] == "S"] if g != "S" else [])))
        grows.append((name, len(gf), without, s_full - without, alone))
    for name, n, w, loss, a in sorted(grows, key=lambda r: -r[3]):
        L.append(f"{name:22} {n:>3} {w:>+13.1%} {loss:>+6.1%} {a:>+11.1%}")
    L.append("  (loss = how much skill disappears without the group; 'alone' includes sector dummies)")

    # 2. single features
    L += ["\n2. SINGLE FEATURES - skill lost when the feature is dropped (top 12)"]
    frows = []
    for f in feats:
        frows.append((s_full - skill(walk_forward(df, [x for x in feats if x != f])), f))
    for loss, f in sorted(frows, reverse=True)[:12]:
        L.append(f"  {loss:+.2%}  {f:32} ({GROUP_NAMES[group[f]]})")
    useless = [f for loss, f in frows if loss <= 0]
    L.append(f"  {len(useless)} features cost nothing or help when removed: {', '.join(sorted(useless))}")

    # 3. segments
    seg = full.join(df[["sector", "report_time", "log_market_cap", "beat_rate_8q"]])
    seg["size"] = pd.qcut(seg["log_market_cap"], 3, labels=["small", "mid", "large"])
    seg["history"] = pd.cut(seg["beat_rate_8q"], [-0.01, 0.625, 0.875, 1.0],
                            labels=["beat <=5 of 8", "6-7 of 8", "8 of 8"])
    L += ["\n3. WHERE IT WORKS - Brier skill vs sector baseline by segment"]
    for col in ["sector", "report_time", "size", "history"]:
        for k, g in seg.groupby(col, observed=True):
            if len(g) >= 100:
                L.append(f"  {col:12} {str(k):24} n={len(g):>5}  skill {skill(g):>+6.1%}  "
                         f"not-beat {g['y'].mean():.0%}")

    # 4. robustness
    L += ["\n4. ROBUSTNESS"]
    df["miss_only"] = (df["outcome"] == "miss").astype(int)
    L.append(f"  target = miss only (ties count as beats): skill {skill(walk_forward(df, feats, 'miss_only')):+.1%}")
    L.append(f"  no fundamentals (group C removed):          skill {skill(walk_forward(df, [f for f in feats if group[f] != 'C'])):+.1%}")
    late = full[df.loc[full.index, "year"] >= 2021]
    L.append(f"  test years 2021-2025 only:                  skill {skill(late):+.1%}")
    y20 = full[df.loc[full.index, "year"] == 2020]
    L.append(f"  2020 alone (COVID year):                    skill {skill(y20):+.1%}")

    # 4b. does a smaller model help? fair test: choose features on 2019-2021, test on 2022-2025
    sel, test = [2019, 2020, 2021], [2022, 2023, 2024, 2025]
    b0 = skill(walk_forward(df, feats, years=sel))
    keep = [f for f in feats if b0 - skill(walk_forward(df, [x for x in feats if x != f], years=sel)) > 0]
    L.append(f"  compact model ({len(keep)} features chosen on 2019-2021), tested 2022-2025: "
             f"skill {skill(walk_forward(df, keep, years=test)):+.1%} vs full model "
             f"{skill(walk_forward(df, feats, years=test)):+.1%}")

    # 5. money test
    r = pd.read_csv(PROC / "reactions.csv", parse_dates=["report_date"])
    mt = full.join(df[["company_key", "report_date", "outcome", "year"]]).merge(
        r[["company_key", "report_date", "car_0_1"]], on=["company_key", "report_date"], how="inner")
    mt["decile"] = mt.groupby("year")["p"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    dec = mt.groupby("decile").agg(risk=("p", "mean"), not_beat=("y", "mean"),
                                   car=("car_0_1", "mean"), n=("y", "size"))
    L += ["\n5. MONEY TEST - 2-day reaction vs S&P 500 by predicted-risk decile (deciles set within each year)",
          (dec.assign(risk=dec["risk"].map("{:.0%}".format), not_beat=dec["not_beat"].map("{:.0%}".format),
                      car=(dec["car"] * 100).map("{:+.2f}%".format))).to_string()]
    top, rest = mt[mt["decile"] == 10]["car_0_1"], mt[mt["decile"] < 10]["car_0_1"]
    bot = mt[mt["decile"] == 1]["car_0_1"]
    diff = top.mean() - bot.mean()
    se = np.sqrt(top.var() / len(top) + bot.var() / len(bot))
    L.append(f"\n  riskiest 10% vs safest 10%: {top.mean()*100:+.2f}% vs {bot.mean()*100:+.2f}% "
             f"(difference {diff*100:+.2f} pts, t = {diff/se:.1f})")
    yearly = mt.groupby("year").apply(lambda g: g[g["decile"] == 10]["car_0_1"].mean()
                                      - g[g["decile"] == 1]["car_0_1"].mean(), include_groups=False)
    L.append(f"  riskiest minus safest decile, by year: "
             + ", ".join(f"{y}: {v*100:+.1f}" for y, v in yearly.items()))
    L.append(f"  riskiest decile below safest in {int((yearly < 0).sum())} of {len(yearly)} years")
    # split by what actually happened; thirds (not tenths) so every group has enough reports
    mt["third"] = mt.groupby("year")["p"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) + 1)
    L.append("\n  Reaction by actual outcome, model's safest third vs riskiest third:")
    for oc in ["beat", "in line", "miss"]:
        a = mt[(mt["third"] == 1) & (mt["outcome"] == oc)]["car_0_1"]
        b = mt[(mt["third"] == 3) & (mt["outcome"] == oc)]["car_0_1"]
        t = (b.mean() - a.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
        L.append(f"    {oc:8} safest {a.mean()*100:+.2f}% (n={len(a):>4})   riskiest {b.mean()*100:+.2f}% "
                 f"(n={len(b):>4})   difference {(b.mean()-a.mean())*100:+.2f} pts, t = {t:.1f}")

    text = "\n".join(L)
    print(text)
    REPORTS.mkdir(exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "part3_results.txt").write_text(text)
    charts(grows, s_full, dec, mt)
    print("\nSaved reports/part3_results.txt and reports/figures/part3_*.png")


def charts(grows, s_full, dec, mt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # group contribution
    g = sorted(grows, key=lambda r: r[3])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([r[0] for r in g], [r[3] * 100 for r in g], color=m1.BLUE, height=0.6)
    for i, r in enumerate(g):
        ax.text(r[3] * 100 + (0.05 if r[3] >= 0 else -0.05), i, f"{r[3]*100:+.1f}",
                va="center", ha="left" if r[3] >= 0 else "right", fontsize=9, color=m1.INK2)
    ax.axvline(0, color=m1.INK, linewidth=1)
    ax.set_xlim(min(-1.0, min(r[3] for r in g) * 100 - 0.6), max(r[3] for r in g) * 100 + 0.6)
    ax.set_xlabel("Skill lost when the group is removed (percentage points)", color=m1.INK2, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(m1.GRID)
    ax.tick_params(colors=m1.INK2, length=0, labelsize=9)
    ax.xaxis.grid(True, color=m1.GRID); ax.set_axisbelow(True)
    ax.set_title("What drives the model", loc="left", fontsize=12, fontweight="bold", color=m1.INK, pad=22)
    ax.text(0, 1.02, f"Full model skill {s_full:+.1%} vs sector baseline; walk-forward 2019-2025",
            transform=ax.transAxes, fontsize=9, color=m1.INK2)
    fig.tight_layout(); fig.savefig(FIGS / "part3_feature_groups.png", dpi=160, facecolor="white")
    plt.close(fig)

    # money test: reaction by outcome, safest vs riskiest decile
    cats = [("beat", "Company beat"), ("in line", "Exactly in line"), ("miss", "Company missed")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.36
    for k, (dsel, col, name) in enumerate([(1, m1.BLUE, "Model's safest third"), (3, m1.ORANGE, "Model's riskiest third")]):
        vals = [mt[(mt["third"] == dsel) & (mt["outcome"] == oc)]["car_0_1"].mean() * 100 for oc, _ in cats]
        xs = np.arange(len(cats)) + (k - 0.5) * w
        ax.bar(xs, vals, width=w - 0.04, color=col, label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + (0.12 if v >= 0 else -0.12), f"{v:+.1f}%", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9, color=m1.INK2)
    ax.axhline(0, color=m1.INK, linewidth=1)
    ax.set_xticks(range(len(cats)), [c[1] for c in cats])
    ax.set_ylabel("Avg 2-day return vs S&P 500 (%)", color=m1.INK2, fontsize=9)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.5, hi + 0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    m1.style(ax, "The market already prices in the risk",
             "When a company the model rated risky still beats, its stock rises ~1 point more (2019-2025)")
    fig.tight_layout(); fig.savefig(FIGS / "part3_reaction_by_risk.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
