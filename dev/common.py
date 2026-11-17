"""Standard-library data helpers and the competition's log-pinball metric."""

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean, median

KIT = Path('/home/hatch/workspace/makefun/kit-gta6-twitch-ccu-forecast-01')
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def load_case(path=KIT / 'input/p00.in'):
    with open(path, encoding='utf-8') as source:
        return json.load(source)


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def iso(value):
    return value.isoformat().replace('+00:00', 'Z')


def hourly_means(points, start, hours=24, field='viewers'):
    """Arithmetic means of points in half-open hour bins; missing bins stay absent.

    Each point has equal weight, irrespective of reads. Do not round or floor
    source values here: flooring to one happens only when taking logarithms.
    """
    bins = defaultdict(list)
    for point in points:
        offset = (timestamp(point['from']) - start).total_seconds() / 3600
        if 0 <= offset < hours:
            bins[int(offset) + 1].append(point[field])
    return {hour: fmean(values) for hour, values in sorted(bins.items())}


def release_hourly(comparable, field='viewers'):
    return hourly_means(comparable['hourly'], timestamp(comparable['day_start']),
                        field=field)


def log_print(value):
    return math.log(max(1, value))


def pre_release_features(comparable):
    """Features available at hour 1; pre median ignores missing hourly bins.

    Counts expose sparse pre-history. Prose metadata is preserved, not parsed
    as an authoritative unlock timestamp (several entries are unverified).
    """
    start = timestamp(comparable['day_start'])
    pre = hourly_means(comparable['hourly'], start - timedelta(hours=24))
    recent = hourly_means(comparable['hourly'], start - timedelta(hours=6), 6)
    first = release_hourly(comparable).get(1)
    channels = release_hourly(comparable, 'channels').get(1)
    pre_median = median(pre.values()) if pre else None
    return {
        'pre_hours': len(pre), 'pre_median': pre_median,
        'pre6_median': median(recent.values()) if recent else None,
        'v1': first, 'channels1': channels,
        'log_v1_pre': (log_print(first) - log_print(pre_median)
                       if first is not None and pre_median is not None else None),
        'viewers_per_channel1': (first / channels
                                if first is not None and channels else None),
        'platforms': comparable.get('platforms', ''),
        'unlock': comparable.get('unlock', ''),
    }


def quantile(values, tau):
    """Sorted linear interpolation at tau * (n - 1)."""
    if not 0 <= tau <= 1:
        raise ValueError('quantile level must be in [0, 1]')
    ordered = sorted(values)
    if not ordered:
        raise ValueError('quantile of an empty sample')
    position = tau * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


def pinball(actual, predicted, tau):
    d = log_print(actual) - log_print(predicted)
    return 2 * max(tau * d, (tau - 1) * d)


def score_forecast(actuals, predictions):
    """Mean across observed target hours and all five levels; missing targets skipped.

    Missing predictions for observed targets are errors, never silently skipped.
    """
    losses = []
    for hour, actual in sorted(actuals.items()):
        if not 2 <= hour <= 24:
            continue
        qs = predictions[hour]
        if len(qs) != len(LEVELS) or not all(math.isfinite(q) for q in qs):
            raise ValueError(f'invalid quantiles at hour {hour}')
        if any(a > b for a, b in zip(qs, qs[1:])):
            raise ValueError(f'crossing quantiles at hour {hour}')
        losses.extend(pinball(actual, q, tau) for q, tau in zip(qs, LEVELS))
    if not losses:
        raise ValueError('no observed target hours')
    return fmean(losses)
