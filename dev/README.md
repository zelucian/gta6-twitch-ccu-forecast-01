# GTA VI Twitch launch forecast: entry and development tools

This repository is the make.fun `gta6-twitch-ccu-forecast-01` entry, predicting quantiles of GTA VI's launch-day Twitch viewers for hours 2 through 24. Forecasts are evaluated by mean log-pinball loss, with smaller values preferred. `main.py` is the executable competition entry; `dev/` contains local evaluation and diagnostic tools, not runtime dependencies of the entry.

## Layout

- `main.py` reads input cases, prepares comparable launches, and produces the fixed v2 ensemble. Its `predict(case)` interface returns five viewer quantiles per forecast hour, while `CONFIG` records the production settings.
- `dev/common.py` supplies input loading, timestamp conversion, hourly aggregation, feature helpers, and quantile utilities. It implements the scoring metric as twice the pinball loss of natural-log viewer counts, averaged across observed target hours and quantile levels.
- `dev/loo.py` builds leave-one-out pseudo-cases from a case's comparables, excluding the held-out launch from training. It evaluates house-baseline and unweighted pooled-ratio reference forecasts and prints fold results and data diagnostics.
- `dev/tune.py` runs leave-one-out evaluation through `main.predict`, caching each comparable's preparation while retaining fold-specific training membership. It also generates variant summaries; its legacy executable grid still includes unsupported `jump_bw` and `shrink` arguments, so use its `run` function for the current model rather than that old grid.
- `dev/validate.py` evaluates all four kit comparable sets and checks the executable's output contract, isolated timing, repeatability, and fallback behavior. It records the run in `dev/validation-results.json`.
- `dev/validate_v2.py` applies the contract, timing, determinism, and sparse/empty-data checks to the fixed v2 ensemble, and requires p00 leave-one-out loss to be at most 0.36. It saves fold-level results and validation measurements in `dev/validation-v2.json`.
- `dev/tuning-results.json` holds earlier tuning output, rather than a fresh validation of the fixed ensemble.

## Running the tools

Run these commands from the repository root with Python 3. The entry uses only the standard library and requires no network access.

```bash
export KIT=/home/hatch/workspace/makefun/kit-gta6-twitch-ccu-forecast-01
MAKEFUN_INPUT_DIR="$KIT/input" MAKEFUN_OUTPUT_DIR=/tmp/gta6-forecasts python3 main.py
```

Each `*.in` file produces a matching `<stem>.out` file; the output directory is created when needed. Without environment overrides, the directories are `/input` and `/output`. Output JSON has only the top-level `hours` key, containing 23 ordered rows for hours 2..24. Each row has `hour` and `q`, with five finite, non-decreasing quantiles at levels 0.05, 0.25, 0.50, 0.75, and 0.95. Values stay in [0.01, 1e9], and each file must be at most 65,536 bytes.

Run the reference-model LOO harness, optionally supplying another case path:

```bash
python3 dev/loo.py
python3 dev/loo.py "$KIT/input/p01.in"
```

Evaluate the current ensemble directly through the cached LOO runner:

```bash
python3 -c 'import main; from dev.common import load_case; from dev.tune import run; result = run(load_case()["comparables"], main.CONFIG); print(result["loss"]); print(max(result["cases"], key=lambda row: row["loss"]))'
```

Run both complete validators:

```bash
python3 dev/validate_v2.py
python3 dev/validate.py
```

The validators use the kit location defined in `dev/common.py`. They launch the entry in temporary directories and assert that isolated execution takes less than eight seconds per case.

## Current method: v2

Training uses comparable launches with an observed release-day hour 1. Hourly means provide a launch level `x = ln(v1)` and subsequent log-ratios `r_h = ln(v_h) - ln(v1)`; viewer counts are floored at 1 before taking logs. The target's first-hour print anchors the predicted viewer scale.

### Pre-release shape

The preceding 24 hours are binned into hourly means and transformed to natural logs. Subtracting the median of the observed log-means yields a level-independent, 24-component profile; unobserved hours receive zero deviation. A profile is available only with at least 12 observed hours. The target profile comes from `history_hourly`, and comparable profiles come from their hourly records before release.

### Member A: similarity-weighted quantiles

Each launch receives squared distance `(x_target - x_comp)^2 / 1.3^2`, plus the squared root-mean-square shape difference divided by `0.5^2` when both profiles exist. Its Gaussian weight is `exp(-distance / 2)`, with a floor of `1e-12`. At each hour, weighted quantiles of the available log-ratios form the initial curve. Quantile interpolation uses cumulative-weight centres stretched to endpoints 0 and 1; equal weights therefore recover ordinary linear sample quantiles.

### Member B: local regression with residual quantiles

For each hour with at least eight training observations, a weighted least-squares line predicts the log-ratio from log first-hour viewers. The Gaussian weights use bandwidth 2.0 around the target launch level and the same `1e-12` floor. The target's fitted mean is shifted by weighted residual quantiles to obtain the five predictions; zero weighted level variance gives slope zero. With fewer than eight observations, this member uses that hour's unweighted ratio quantiles instead.

### Smoothing and combination

Each member independently smooths its log-ratio curves: 70% of the original hour's row plus 30% of the mean of its adjacent rows, using only one adjacent row at the endpoints. All smoothing reads the original curves, so changes do not propagate sequentially. The member then converts back to viewer counts using the target anchor, bounds the values, and rounds half-up to two decimal places.

The final forecast is a geometric blend of the completed member forecasts: `exp(0.4 * ln(A) + 0.6 * ln(B))` for each hour and quantile. Another half-up rounding step, bounding to [0.01, 1e9], and a non-decreasing pass produce the final rows.

The fixed `CONFIG` is:

| Setting | Value | Purpose |
| --- | ---: | --- |
| `v1_bw` | 1.3 | Member A's log-level bandwidth |
| `shape_bw` | 0.5 | Member A's shape-distance bandwidth |
| `regression_bw` | 2.0 | Member B's local-fit bandwidth |
| `min_fit` | 8 | Minimum observations for an hourly regression |
| `smooth` | 0.3 | Adjacent-hour contribution within each member |
| `kernel_weight` | 0.4 | Member A's log-space ensemble weight; B receives 0.6 |

### Fallbacks

Missing shape information removes only the shape contribution to the kernel distance. An hour without ratio observations uses unweighted ratios pooled across all available training hours. If there are no usable ratios at all, both members start from the house-style log-ratio curve `z * 0.15 * sqrt(h - 1)`, where `z` is `[-1.645, -0.674, 0, 0.674, 1.645]`, before their normal finishing steps. If the target has no known hour-1 print, its latest supplied history value is used, or 1 when history is empty; the anchor is always floored at 1.

## Earlier model: v1

The original entry used a single Gaussian kernel on log first-hour viewers, without the shape term or regression member. Its bandwidth was 1.0 and its adjacent-hour smoothing weight was 0.3, yielding a leave-one-out loss of 0.400267.

## Results

The p00 evaluation gives each scored launch equal weight after averaging its observed-hour quantile losses. There are 37 scored folds among 38 comparables; the pre-release GTA VI category record is skipped because it lacks release-day hour 1.

| Model | Mean LOO loss | Worst fold loss |
| --- | ---: | ---: |
| House baseline | 0.929650 | Not reported here |
| v1: kernel on ln(v1), bandwidth 1.0, smoothing 0.3 | 0.400267 | Not reported here |
| v2: fixed geometric ensemble | 0.3584725743734883 | 1.4663080749740334 |

The v2 worst fold is **Black Myth: Wukong**. `validate_v2.py` passed its loss threshold and all contract, determinism, timing, and fallback assertions. Its measured isolated runtimes were 0.2624, 0.1993, 0.3074, and 0.2455 seconds for p00 through p03, respectively.

The cross-set validator completed with these printed losses:

| Input | Chosen loss | Printed pooled loss |
| --- | ---: | ---: |
| p00 | 0.3584725743734883 | 0.3584725743734883 |
| p01 | 0.3584725743734883 | 0.3584725743734883 |
| p02 | 0.3584725743734883 | 0.3584725743734883 |
| p03 | 0.3584725743734883 | 0.3584725743734883 |

Here, the validator's `pooled` label means `run(comparables, {})`: the current `predict` defaults already match `CONFIG`, so those columns evaluate the same ensemble. This is not the unweighted pooled reference implemented in `dev/loo.py`. The validator also printed identical chosen/reference regression-fold losses of 0.30001601642786974 on every input.

An additional independent check ran `main.py` twice over all four kit inputs into separate temporary directories. Every corresponding output matched byte for byte. These results describe comparable-launch holdouts and execution checks, not observed GTA VI launch outcomes.
