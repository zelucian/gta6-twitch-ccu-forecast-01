import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from statistics import median, fmean


sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get('MAKEFUN_INPUT_DIR', '/input'))
OUTPUT_DIR = Path(os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
CONFIG = dict(v1_bw=1.3, shape_bw=0.5, regression_bw=2.0, min_fit=8,
              smooth=0.3, kernel_weight=0.4)


def round2(value):
    scaled = value * 100
    whole = math.floor(scaled)
    rounded = (whole + (scaled - whole >= 0.5)) / 100
    return int(rounded) if rounded.is_integer() else rounded


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def log(value):
    return math.log(max(1, value))


def bins(points, start, count=24):
    grouped = {}
    for point in points:
        offset = (timestamp(point['from']) - start).total_seconds() / 3600
        viewers = point['viewers']
        if 0 <= offset < count and math.isfinite(viewers):
            grouped.setdefault(int(offset) + 1, []).append(viewers)
    return {hour: fmean(grouped[hour]) for hour in sorted(grouped)}


def shape(points, start):
    profile = bins(points, start - timedelta(hours=24))
    if len(profile) < 12:
        return None
    logged = {hour: log(value) for hour, value in profile.items()}
    centre = median(logged.values())
    return [logged.get(hour, centre) - centre for hour in range(1, 25)]


def prepare(comparable):
    start = timestamp(comparable['day_start'])
    points = comparable['hourly']
    profile = bins(points, start)
    if 1 not in profile:
        return None
    level = log(profile[1])
    ratios = {hour: log(value) - level
              for hour, value in profile.items() if hour >= 2}
    return level, shape(points, start), ratios


def weighted_quantiles(samples):
    ordered = sorted(samples, key=lambda sample: sample[0])
    if len(ordered) == 1:
        return [ordered[0][0]] * len(LEVELS)
    centres = []
    cumulative = 0.0
    for value, weight in ordered:
        centres.append(cumulative + weight / 2)
        cumulative += weight
    span = centres[-1] - centres[0]
    positions = [(centre - centres[0]) / span for centre in centres]
    quantiles = []
    index = 0
    for tau in LEVELS:
        while index < len(positions) - 2 and positions[index + 1] < tau:
            index += 1
        fraction = (tau - positions[index]) / (positions[index + 1] - positions[index])
        lower = ordered[index][0]
        upper = ordered[index + 1][0]
        quantiles.append(lower + fraction * (upper - lower))
    return quantiles


def finish(curves, p1, smooth):
    result = {}
    level = log(p1)
    ceiling = math.log(1e9)
    for hour, row in sorted(curves.items()):
        neighbours = [curves[adjacent] for adjacent in (hour - 1, hour + 1)
                      if adjacent in curves]
        blended = [(1 - smooth) * value + smooth * fmean(
            neighbour[index] for neighbour in neighbours)
            if neighbours else value for index, value in enumerate(row)]
        result[hour] = [round2(max(0.01, math.exp(min(ceiling, level + value))))
                        for value in blended]
    return result


def predict(case, v1_bw=1.3, shape_bw=0.5, regression_bw=2.0, min_fit=8,
            smooth=0.3, kernel_weight=0.4, prepared=None):
    first = next((row['viewers'] for row in case.get('known', [])
                  if row['hour'] == 1), None)
    if first is None:
        history = case.get('history_hourly') or case.get('history') or []
        latest = max(history, key=lambda point: timestamp(
            point.get('from') or point['at'])) if history else None
        first = latest['viewers'] if latest else 1
    p1 = max(1, first)
    target = log(p1)
    target_shape = shape(case.get('history_hourly') or [], timestamp(case['day_start']))
    training = prepared if prepared is not None else (
        prepare(comparable) for comparable in case.get('comparables', []))
    samples = {hour: [] for hour in range(2, 25)}
    for comparable in training:
        if comparable is None:
            continue
        level, comp_shape, ratios = comparable
        distance = ((target - level) / v1_bw) ** 2
        if target_shape is not None and comp_shape is not None:
            distance += fmean((target_value - comp_value) ** 2
                              for target_value, comp_value in zip(target_shape, comp_shape)) / shape_bw ** 2
        weight = max(1e-12, math.exp(-distance / 2))
        local_weight = max(1e-12, math.exp(-0.5 * ((level - target) / regression_bw) ** 2))
        for hour, ratio in ratios.items():
            samples[hour].append((ratio, level, weight, local_weight))

    pooled = [(sample[0], 1) for hour_samples in samples.values() for sample in hour_samples]
    fallback_curves = {}
    kernel_curves = {}
    regression_curves = {}
    for hour, hour_samples in samples.items():
        fallback = [(sample[0], 1) for sample in hour_samples] or pooled
        fallback_curves[hour] = weighted_quantiles(fallback) if fallback else [
            score * 0.15 * math.sqrt(hour - 1) for score in Z]
        kernel_curves[hour] = weighted_quantiles([
            (sample[0], sample[2]) for sample in hour_samples
        ]) if hour_samples else fallback_curves[hour]
        if len(hour_samples) < min_fit:
            regression_curves[hour] = fallback_curves[hour]
            continue
        total_weight = sum(sample[3] for sample in hour_samples)
        xbar = sum(level * local_weight for ratio, level, weight, local_weight
                   in hour_samples) / total_weight
        ybar = sum(ratio * local_weight for ratio, level, weight, local_weight
                   in hour_samples) / total_weight
        variance = sum(local_weight * (level - xbar) ** 2
                       for ratio, level, weight, local_weight in hour_samples)
        covariance = sum(local_weight * (level - xbar) * (ratio - ybar)
                         for ratio, level, weight, local_weight in hour_samples)
        slope = covariance / variance if variance else 0
        mu = ybar + slope * (target - xbar)
        residuals = [(ratio - (ybar + slope * (level - xbar)), local_weight)
                     for ratio, level, weight, local_weight in hour_samples]
        regression_curves[hour] = [mu + value for value in weighted_quantiles(residuals)]

    kernel_forecast = finish(kernel_curves, p1, smooth)
    regression_forecast = finish(regression_curves, p1, smooth)
    result = {}
    for hour in samples:
        row = []
        previous = 0.01
        for kernel_value, regression_value in zip(kernel_forecast[hour], regression_forecast[hour]):
            value = round2(math.exp(kernel_weight * math.log(kernel_value)
                                   + (1 - kernel_weight) * math.log(regression_value)))
            previous = max(previous, min(1e9, max(0.01, value)))
            row.append(previous)
        result[hour] = row
    return result


def forecast_case(case, name='case'):
    if case.get('levels') != LEVELS:
        raise ValueError(f'{name}: levels must equal {LEVELS}')
    forecast = predict(case, **CONFIG)
    return {'hours': [{'hour': hour, 'q': forecast[hour]} for hour in range(2, 25)]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(INPUT_DIR.glob('*.in')):
        case = json.loads(source.read_text())
        result = forecast_case(case, source.stem)
        (OUTPUT_DIR / f'{source.stem}.out').write_text(
            json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
