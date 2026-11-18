"""Standard-library data helpers and the competition log-pinball metric."""

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean, median

KIT = Path('/home/hatch/workspace/makefun/kit-gta6-twitch-ccu-forecast-01')
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def load_case(path=KIT / 'input/p00.in'):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def timestamp(value):
    return datetime.fromisoformat(value[:-1] + '+00:00' if value.endswith('Z') else value)


def iso(value):
    return value.isoformat().replace('+00:00', 'Z')


def hourly_means(points, start, hours=24, field='viewers'):
    bins = defaultdict(list)
    for point in points:
        offset = (timestamp(point['from']) - start).total_seconds() / 3600
        if 0 <= offset < hours:
            bins[int(offset) + 1].append(point[field])
    return {hour: fmean(bins[hour]) for hour in sorted(bins)}


def release_hourly(comparable, field='viewers'):
    return hourly_means(comparable['hourly'], timestamp(comparable['day_start']), field=field)


def log_print(value):
    return math.log(max(1, value))


def pre_release_features(comparable):
    start = timestamp(comparable['day_start'])
    pre = hourly_means(comparable['hourly'], start - timedelta(hours=24))
    recent = hourly_means(comparable['hourly'], start - timedelta(hours=6), hours=6)
    v1 = release_hourly(comparable).get(1)
    channels1 = release_hourly(comparable, 'channels').get(1)
    pre_median = median(pre.values()) if pre else None
    return {
        'pre_hours': len(pre),
        'pre_median': pre_median,
        'pre6_median': median(recent.values()) if recent else None,
        'v1': v1,
        'channels1': channels1,
        'log_v1_pre': log_print(v1) - log_print(pre_median)
        if v1 is not None and pre_median is not None else None,
        'viewers_per_channel1': v1 / channels1
        if v1 is not None and channels1 else None,
        'platforms': comparable.get('platforms', ''),
        'unlock': comparable.get('unlock', ''),
    }


def quantile(values, tau):
    sample = sorted(values)
    if not 0 <= tau <= 1 or not sample:
        raise ValueError('Quantiles require a nonempty sample and tau in [0, 1]')
    position = tau * (len(sample) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sample[lower] * (1 - fraction) + sample[upper] * fraction


def pinball(actual, predicted, tau):
    difference = log_print(actual) - log_print(predicted)
    return 2 * max(tau * difference, (tau - 1) * difference)


def score_forecast(actuals, predictions):
    losses = []
    for hour in range(2, 25):
        if hour not in actuals:
            continue
        values = predictions[hour]
        try:
            valid = (len(values) == len(LEVELS)
                     and all(math.isfinite(value) for value in values)
                     and all(left <= right for left, right in zip(values, values[1:])))
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f'Invalid quantiles for hour {hour}')
        losses.extend(pinball(actuals[hour], value, tau) for value, tau in zip(values, LEVELS))
    if not losses:
        raise ValueError('No observed target hours')
    return fmean(losses)
