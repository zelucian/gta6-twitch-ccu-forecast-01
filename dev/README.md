# Exploration and LOO harness

Run `python3 dev/loo.py` from the repo root (or `python3 -m dev.loo`).
An optional positional argument selects another case file. Python standard
library only; no network; the default run takes about 0.6 seconds here.

`common.py` exposes loading, half-open release-day hour aggregation, pre-release
features, deterministic interpolated quantiles, and log-pinball scoring.
`loo.py` exposes `baseline(case)`, `pooled(case)`, `pseudo_case(comparables, index)`,
and `evaluate(comparables, forecaster)` for experimenting with other models.
The CLI prints each fold, equal-case and observed-hour-weighted aggregate losses,
and exploratory data summaries. Missing actual hours are skipped; missing
predictions for observed target hours raise an error.

## p00 results

| Forecaster | Mean LOO loss |
| --- | ---: |
| House baseline (including starter rounding) | 0.929650 |
| Pooled log-ratio quantiles | 0.713858 |

Pooled reduces loss by 23.21%. There are 37 scored folds and 851 scored target
hours, with all 24 release-day hours in each scored game. Of the 38 supplied
comparables, the GTA VI pre-release category row is explicitly not a launch and
has no hour 1: its fold cannot be constructed and is reported as skipped, with
no imputation. It remains among the other 37 comparables in each scored fold,
but supplies no paired hour-1 ratios. Thus pooled uses 36 usable training games
per fold. Aggregate case and hour weighting give identical results here.

For the 37 games with hour 1 and observed pre-history, ln(v1 / pre24 median)
has min / p05 / p25 / median / p75 / p95 / max:
`-0.827 / -0.242 / 0.142 / 0.702 / 2.130 / 3.570 / 5.186`.
Missing pre-release bins are omitted, never filled with zeros; logarithms floor
counts at one. The CLI reports pre-history coverage explicitly.

Median ln(vh / v1) is -0.238 at hour 6, -0.022 at hour 12, 0.668 at
hour 18, and 0.712 at hour 24. The profiles typically dip and then recover,
with substantial variation across games. Pearson correlation of log(v1) with
mean late-day log growth (hours 18–24) is -0.934; correlation of the pre-release
jump with that growth is only 0.152. These are exploratory associations, and
the strong first correlation shares v1 in both variables; they do not establish
an out-of-sample benefit from conditioning on those features.

Validation performed: synthetic checks of interpolated quantiles, flooring,
asymmetric loss, missing-target scoring and hour boundaries; history cutoff
and held-out exclusion checks for every scored fold; exact agreement with the
kit starter on all fold baseline forecasts; repeat-run equality for pooled.


## Task B: conditioned runtime forecaster

`../main.py` is self-contained and uses only the standard library. It prepares
release-day hourly means, requires a comparable's own hour 1 for paired log
ratios, and never imputes a release hour 1 for the GTA VI pre-release row.
Target features use only supplied known/history data. Pre24 is the median of
observed hourly means in the 24 hours before day_start; the recent feature is
the median in the last three hours before checkpoint. Missing features leave
the corresponding kernel factor neutral. The recent feature is computed but
not used by the selected model.

Weights are Gaussian: exp(-0.5 * sum((feature difference / bandwidth)^2)),
with a 1e-12 floor for numerical support. Quantiles interpolate weighted CDF
centers rescaled to the first and last centers; equal weights reproduce the
Task A linear empirical quantile exactly. Smoothing is one simultaneous pass
in log space: 70% own hour plus 30% mean of available adjacent forecast hours
(one neighbor at hours 2 and 24). Convex smoothing preserves quantile ordering.
Shrinkage experiments blend weighted and pooled log quantiles.

**Chosen:** ln(v1) bandwidth 1.0, smoothing 0.30, no jump weighting or shrinkage.
The p00 gain is 43.93%, comfortably above the required 1.5% threshold. The
worst absolute fold decreases from 2.698410 (Hollow Knight: Silksong) to
1.823764 (Black Myth: Wukong). Largest individual regression is Resident Evil
Requiem: 0.764580 pooled -> 1.070965 chosen (+0.306385). Thus the gain is not
uniform, but there is no new extreme fold loss. No additional hyperparameter
search was performed beyond the table below.

### p00 equal-fold mean log-pinball loss (37 folds)

| Variant | Loss | Gain vs pooled | Worst fold loss | Largest fold regression |
| --- | ---: | ---: | ---: | ---: |
| pooled | 0.713858 | 0.00% | 2.698410 | +0.000000 |
| w_jump_0.5 | 0.741868 | -3.92% | 2.749595 | +1.083294 |
| w_jump_1 | 0.696964 | 2.37% | 2.411395 | +0.308276 |
| w_jump_2 | 0.723057 | -1.29% | 2.629512 | +0.165756 |
| w_v1_0.5 | 0.447099 | 37.37% | 1.900761 | +0.532036 |
| w_v1_1 | 0.401788 | 43.72% | 1.829353 | +0.306909 |
| w_v1_2 | 0.453847 | 36.42% | 2.014922 | +0.142051 |
| w_v1_3 | 0.539021 | 24.49% | 2.254210 | +0.110382 |
| w_both_1_1 | 0.413776 | 42.04% | 1.494154 | +0.259620 |
| w_both_1_2 | 0.402894 | 43.56% | 1.686424 | +0.272701 |
| w_both_2_1 | 0.451713 | 36.72% | 1.807840 | +0.125788 |
| w_both_2_2 | 0.447054 | 37.37% | 1.944785 | +0.129328 |
| smooth_0.15 | 0.400861 | 43.85% | 1.826559 | +0.306641 |
| smooth_0.3 | 0.400267 | 43.93% | 1.823764 | +0.306385 |
| shrink_0.15 | 0.423281 | 40.71% | 1.872442 | +0.153787 |
| shrink_0.3 | 0.460840 | 35.44% | 1.946471 | +0.093159 |
| shrink_0.5 | 0.525546 | 26.38% | 2.073126 | +0.062849 |

All weighted v1, both-feature, smooth and shrink variants have Black Myth:
Wukong as the worst absolute fold. Jump bandwidth 0.5 has Silksong, bandwidth
1 has ARC Raiders, bandwidth 2 has Silksong. Full names of largest-regression
folds are in `tuning-results.json`.

### Other practice comparable sets

| Input | Scored folds | Pooled loss | Chosen loss | Gain |
| --- | ---: | ---: | ---: | ---: |
| p00 | 37 | 0.713858 | 0.400267 | 43.93% |
| p01 | 37 | 0.713858 | 0.400267 | 43.93% |
| p02 | 37 | 0.713858 | 0.400267 | 43.93% |
| p03 | 37 | 0.713858 | 0.400267 | 43.93% |

The four supplied comparable arrays are structurally identical (checked by
parsed-object equality). These separate runs therefore confirm consistency,
not independent generalization. Each skips the same GTA VI pre-release fold.

### Validation and reproduction

Run `python3 dev/tune.py` for the fixed variant grid and
`python3 dev/validate.py` for cross-set LOO and entry validation. Results are
saved in `tuning-results.json` and `validation-results.json`.

The actual entry ran on all four KIT/input files into temporary directories,
using `python3 -P -s main.py`. Every output has only the top-level `hours` key,
exactly 23 rows in order 2..24, five finite ascending positive quantiles at
most 1e9, and fits within 65,536 bytes. Two full runs were byte-identical.

| Case | Seconds including process startup | Bytes |
| --- | ---: | ---: |
| p00 | 0.0997 | 1456 |
| p01 | 0.1005 | 1510 |
| p02 | 0.0988 | 1550 |
| p03 | 0.0977 | 1509 |

The combined four-case run took 0.251 seconds. A synthetic zero-hour-1,
empty-comparables case passed the same contract checks. Additional missing
hour-1 / missing-history and sparse-comparable checks passed. Missing hour 1
uses the latest supplied history value, then p1=1. An hour without pairs uses
unweighted pooled ratios across all available paired hours; no pairs anywhere
uses the house baseline. Counts are floored at 1 for logarithms; final values
are capped at 1e9 and floored at 0.01 after half-up rounding protection.
The kit was not modified.
