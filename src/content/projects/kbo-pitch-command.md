---
title: Pitch command prediction across a season boundary
summary: Offline gains and leaderboard gains came apart. Rather than guess which was real, I measured the transfer rate of each kind of change and discounted accordingly.
context: LG Aimers 9th / DACON competition
period: 2026.08 – 2026.09
role: Experiment design and implementation
stack: Python, CatBoost, NumPy, pandas
tags: [Tabular, Distribution Shift, Calibration]
date: 2026-09-02
---

## Setup

For a single pitch, predict the probability that the pitcher hits his spot — using only information available *before* the ball is thrown. Scored by Brier Skill Score × 1e5 against a constant-0.5 baseline.

Train on 2019–2024 seasons. Predict **2025**. That sentence is the whole problem: train and evaluation sit on opposite sides of a season boundary, and a structure that fits beautifully inside a season may not survive crossing one.

Final: **1146.91**, 97th of 2,403 participants.

## The gap that defined the project

Early on, offline validation and the leaderboard stopped agreeing. An improvement worth +90 on the 2024 fold moved the leaderboard by −6.

The usual response is to trust one and distrust the other. Instead I spent four submissions measuring how much each *kind* of change carries across the boundary.

| What was changed | Offline → leaderboard transfer |
|---|---|
| Within-season cell structure — frozen tables keyed on pitcher × handedness, pitcher × count | **0% or negative** |
| Structure and regime — recoding the futures-league flag, a team-level regime indicator | ~90% |
| Learning objective, ensembling, calibration | ~90% |

After that, offline gains were read through the axis they came from rather than at face value. That single habit is what the project is actually about.

## Why cell structure doesn't transfer

The tempting quantity, when hunting for group effects the model missed, is

```
gain = 4e5 × Σ wᵍ · bᵍ²
```

the score you would recover by perfectly removing each group's mean residual. It is easy to compute and almost entirely noise, because the groups were chosen using the same fold's labels that then score them.

Measuring three things per axis instead of one makes the problem visible.

| Axis | Within-season split-half | Across-season | Same-fold bound | Actually transfers |
|---|---:|---:|---:|---:|
| Pitcher team × batter team | **0.686** | **−0.059** | 157.2 | **0.0** |
| Pitcher team × month | 0.626 | −0.171 | 93.3 | **0.0** |
| Pitcher ID | 0.410 | 0.026 | 37.2 | **0.0** |

The first column is a positive control: the effect is real and reproducible *inside* a season. The second says it does not survive the boundary. A 157-point bound next to a −0.059 across-season correlation is a measurement of noise, not an opportunity.

The scale check is worth stating plainly. At n ≈ 2,500 per cell the standard error of a mean residual is √(0.25/2500) = 0.010. A filter of "same sign in both folds and |bias| ≥ 0.005 in each" passes **about 19% of pure noise** — 19 hits out of 100 cells, all spurious.

## What did transfer

The largest single gain, +33.7 offline at roughly 90% transfer, came from one team indicator.

Not because that team's outcomes were unusual — because its *composition* was. It played 38.1% futures-league games where other teams played 3–10%, an order of magnitude apart, and futures-league success rate collapsed from 0.70 to 0.47 in April 2023. A "this team" × "post-change regime" interaction separated that contamination from everything else.

Running the same logic across the data found nothing more at team level, but one more at pitcher level: pitchers with a high share of futures-league appearances carry a two-regime average in their history features, so the model systematically overpredicts them.

| Futures share | 0 | 0–5% | 5–20% | 20–50% | 50–80% | >80% |
|---|---:|---:|---:|---:|---:|---:|
| 2023 fold residual | −0.0015 | +0.0039 | +0.0065 | −0.0020 | −0.0100 | −0.0172 |
| 2024 fold residual | +0.0027 | +0.0041 | +0.0026 | −0.0033 | −0.0115 | −0.0075 |

The decisive split: restricted to regular-season rows the across-season correlation is 0.233 and even flips sign; restricted to futures-league rows it is **0.804**. It was never a regular-season signal. All 12 variants of the correction were positive on both folds.

## Knowing when to stop

Before searching for more ensemble weight, I asked how much was left to find. Taking 18 stored candidates and solving for the best non-negative convex combination *with access to the fold labels* — an oracle, not a deployable model — gives an upper bound on reweighting.

```
oracle 1032.7   ·   current 1031.6   ·   headroom +1.1   ·   Frank-Wolfe gap 3.6e-08
```

Cheating with the answers is worth 1.1 points. The ensemble and reweighting axis was closed, and anything further had to come from new row-level signal. This turned out to be the most useful diagnostic in the project — not because it improved the score, but because it stopped work that would not have.

## Calibration without the leaderboard

The organisers flagged repeated submissions that differ only in a calibration constant as leaderboard probing. So the target-mean constant had to be derived from training data alone.

Regular-league and futures-league rates move differently enough that a blended season average is the wrong object. Regular declines monotonically, −0.006 to −0.023 per season (sd 0.008). Futures swings −0.10 / +0.12 / +0.00 / **−0.236** / −0.014. Blending inherits the noise of the 11.8% minority.

Extrapolating the two separately and remixing at the previous season's row composition, A/B'd under an identical nested procedure: **RMSE 0.01748 → 0.01213**.

## What I take from it

Two things.

First, an offline number is not a scalar — it carries the axis it came from, and the same magnitude means different things depending on that axis. Discounting by measured transfer rate was more valuable than any individual model change.

Second, the strongest result in the project was a *negative* one. The convex oracle said the ensemble axis was exhausted, which was worth more than the +1.1 it identified, because it redirected the remaining time.

---

Code and full experiment log: [lg-aimers-9-kbo-pitch-command](https://github.com/chb2066/lg-aimers-9-kbo-pitch-command). Competition data is not redistributed.
