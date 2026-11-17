# gta6-twitch-ccu-forecast-01

This make.fun competition entry forecasts the Twitch **Grand Theft Auto VI**
category's hourly viewer count for hours 2–24 of GTA VI's release day (UTC),
given everything recorded up to the end of hour 1. Each hour has five predicted
quantiles: 5%, 25%, 50%, 75%, and 95%.

## Method

At runtime, the program learns from the case's own `comparables`: hourly Twitch
series for past big launches, with data from SullyGnome. For each comparable it
computes per-hour log-ratios, `ln(v_h) - ln(v_1)`, and weights them with a Gaussian
kernel on similarity in `ln(hour-1 viewers)` (bandwidth 1.0). It calculates
empirical weighted quantiles for each hour, then lightly smooths them across
neighbouring hours in log space (smoothing 0.3). With no usable comparable data,
it falls back to the house persistence baseline.

The program uses only the Python 3 standard library, runs offline, and is
deterministic.

## Validation

Leave-one-out validation over 37 usable comparables gives mean pinball loss on
the log scale of **0.400267**, compared with **0.929650** for the house persistence
baseline and **0.713858** for unweighted pooling. Runtime is approximately 0.1
seconds per case, including process startup. The harness and detailed results
are in [dev/](dev/README.md).

## Run

As in the kit starter, set the input and output directories and run:

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 -P -s main.py
```

The program reads every `*.in` JSON file in the input directory and writes a
matching `*.out` JSON file in the output directory, creating that directory if
needed. The defaults are `/input` and `/output`. Each output contains an `hours`
array with hours 2–24 and five quantiles per hour.

## License and data

Code: MIT; see [LICENSE](LICENSE).

Data attribution: [SullyGnome.com](https://sullygnome.com/), via the competition kit.
