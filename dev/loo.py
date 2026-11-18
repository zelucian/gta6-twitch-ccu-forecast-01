"""Leave-one-out evaluation with explicit missing-data handling."""

import argparse
import math
from datetime import timedelta
from statistics import correlation, fmean

if __package__:
    from .common import (
        KIT, LEVELS, iso, load_case, log_print, pre_release_features,
        quantile, release_hourly, score_forecast, timestamp,
    )
else:
    from common import (
        KIT, LEVELS, iso, load_case, log_print, pre_release_features,
        quantile, release_hourly, score_forecast, timestamp,
    )


def baseline(case):
    p1 = max(1, next(row['viewers'] for row in case['known'] if row['hour'] == 1))
    return {
        row['hour']: [
            math.floor(p1 * math.exp(zscore * 0.15 * math.sqrt(row['hour'] - 1)) * 100 + 0.5) / 100
            for zscore in (-1.645, -0.674, 0, 0.674, 1.645)
        ]
        for row in case['forecast']
    }


def pooled(case):
    p1 = max(1, next(row['viewers'] for row in case['known'] if row['hour'] == 1))
    profiles = [release_hourly(comparable) for comparable in case['comparables']]
    predictions = {}
    for row in case['forecast']:
        hour = row['hour']
        ratios = [log_print(profile[hour]) - log_print(profile[1])
                  for profile in profiles if 1 in profile and hour in profile]
        if not ratios:
            raise ValueError(f'No training observations for hour {hour}')
        predictions[hour] = [p1 * math.exp(quantile(ratios, tau)) for tau in LEVELS]
    return predictions


def pseudo_case(comparables, index):
    comparable = comparables[index]
    actual = release_hourly(comparable)
    if 1 not in actual:
        raise ValueError(f"{comparable['name']}: missing hour 1")
    channels = release_hourly(comparable, 'channels')
    start = timestamp(comparable['day_start'])
    checkpoint = start + timedelta(hours=1)
    history = [dict(point) for point in comparable['hourly'] if timestamp(point['from']) < checkpoint]
    return {
        'schema': 'makefun/twitch-ccu-case/v1',
        'id': f'loo-{index:02d}',
        'game': {'name': comparable['name']},
        'day_start': iso(start),
        'checkpoint': iso(checkpoint),
        'levels': list(LEVELS),
        'known': [{'hour': 1, 'from': iso(start), 'until': iso(checkpoint),
                   'viewers': actual[1], 'channels': channels.get(1),
                   'reads': sum(point.get('reads', 1) for point in history
                                if timestamp(point['from']) >= start)}],
        'forecast': [{'hour': hour, 'from': iso(start + timedelta(hours=hour - 1)),
                      'until': iso(start + timedelta(hours=hour))} for hour in range(2, 25)],
        'history_hourly': history,
        'history': [{'at': point['from'], 'viewers': point['viewers'],
                     'channels': point['channels']} for point in history],
        'comparables': [other for other_index, other in enumerate(comparables) if other_index != index],
        'notes': comparable.get('unlock', ''),
        'platforms': comparable.get('platforms', ''),
        'streams': None,
        'categories_hourly': None,
    }


def evaluate(comparables, forecaster):
    cases = []
    skipped = []
    for index, comparable in enumerate(comparables):
        actual = release_hourly(comparable)
        hours = sum(hour in actual for hour in range(2, 25))
        reason = 'missing hour 1' if 1 not in actual else 'no target hours' if not hours else None
        if reason:
            skipped.append({'name': comparable['name'], 'reason': reason})
            continue
        predictions = forecaster(pseudo_case(comparables, index))
        cases.append({'name': comparable['name'], 'hours': hours,
                      'loss': score_forecast(actual, predictions)})
    return {
        'loss': fmean(row['loss'] for row in cases),
        'cases': cases,
        'skipped': skipped,
        'hour_weighted_loss': sum(row['loss'] * row['hours'] for row in cases)
        / sum(row['hours'] for row in cases),
    }


def distribution(label, values):
    sample = list(values)
    levels = (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)
    names = ('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max')
    details = ' '.join(f'{name}={quantile(sample, tau):.3f}' for name, tau in zip(names, levels)) if sample else 'empty'
    print(f'{label}: n={len(sample)} {details}')


def data_summary(comparables):
    profiles = [release_hourly(comparable) for comparable in comparables]
    features = [pre_release_features(comparable) for comparable in comparables]
    print('Exploratory data summary (not evidence of model quality)')
    print(f'comparables={len(comparables)} complete_24={sum(len(profile) == 24 for profile in profiles)} '
          f'hour_1={sum(1 in profile for profile in profiles)} '
          f'target_hours={sum(sum(hour in profile for hour in range(2, 25)) for profile in profiles)}')
    distribution('pre_hours', [feature['pre_hours'] for feature in features])
    distribution('log_v1_pre', [feature['log_v1_pre'] for feature in features if feature['log_v1_pre'] is not None])
    for hour in (6, 12, 18, 24):
        distribution(f'ln(v_{hour}/v_1)', [log_print(profile[hour]) - log_print(profile[1])
                                         for profile in profiles if 1 in profile and hour in profile])
    for name in ('log_v1_pre', 'log_v1', 'viewers_per_channel1'):
        pairs = []
        for profile, feature in zip(profiles, features):
            value = log_print(feature['v1']) if name == 'log_v1' and feature['v1'] is not None else feature.get(name)
            late = [log_print(profile[hour]) - log_print(profile[1])
                    for hour in range(18, 25) if 1 in profile and hour in profile]
            if value is not None and late:
                pairs.append((value, fmean(late)))
        try:
            coefficient = correlation([pair[0] for pair in pairs], [pair[1] for pair in pairs])
            detail = f'{coefficient:.3f}'
        except (ValueError, ZeroDivisionError):
            detail = 'undefined'
        print(f'Pearson {name} vs late log ratio: n={len(pairs)} r={detail}')
    for comparable in comparables:
        if 'not a launch' in comparable.get('unlock', '').lower():
            print(f"Caution: {comparable['name']}: {comparable['unlock']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', nargs='?', default=KIT / 'input/p00.in')
    comparables = load_case(parser.parse_args().case)['comparables']
    house = evaluate(comparables, baseline)
    pool = evaluate(comparables, pooled)
    print('name\thours\tbaseline\tpooled')
    for house_row, pool_row in zip(house['cases'], pool['cases']):
        print(f"{house_row['name']}\t{house_row['hours']}\t{house_row['loss']:.6f}\t{pool_row['loss']:.6f}")
    for name, result in (('baseline', house), ('pooled', pool)):
        print(f"{name}: mean={result['loss']:.6f} hour_weighted_mean={result['hour_weighted_loss']:.6f}")
    print(f"scored_folds={len(house['cases'])} target_hours={sum(row['hours'] for row in house['cases'])}")
    for row in house['skipped']:
        print(f"Skipped {row['name']}: {row['reason']}")
    print(f"Pooled relative loss reduction: {100 * (1 - pool['loss'] / house['loss']):.3f}%")
    data_summary(comparables)


if __name__ == '__main__':
    main()
