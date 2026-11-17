"""Run with python3 dev/loo.py [case.in], or python3 -m dev.loo.

The primary aggregate equally weights pseudo-cases, matching the competition's
mean over cases. An observed-hour-weighted mean is also reported for sparse data.
"""

import argparse
import math
from datetime import timedelta
from statistics import correlation, fmean

if __package__:
    from .common import (KIT, LEVELS, iso, load_case, log_print, pre_release_features,
                         quantile, release_hourly, score_forecast, timestamp)
else:
    from common import (KIT, LEVELS, iso, load_case, log_print, pre_release_features,
                        quantile, release_hourly, score_forecast, timestamp)


def baseline(case):
    p1 = max(1, next(p['viewers'] for p in case['known'] if p['hour'] == 1))
    # Match the starter's half-up two-decimal output rounding.
    return {row['hour']: [math.floor(p1 * math.exp(z * 0.15 *
                           math.sqrt(row['hour'] - 1)) * 100 + 0.5) / 100
                         for z in (-1.645, -0.674, 0, 0.674, 1.645)]
            for row in case['forecast']}


def pooled(case):
    p1 = max(1, next(p['viewers'] for p in case['known'] if p['hour'] == 1))
    profiles = [release_hourly(c) for c in case['comparables']]
    result = {}
    for row in case['forecast']:
        hour = row['hour']
        ratios = [log_print(p[hour]) - log_print(p[1])
                  for p in profiles if 1 in p and hour in p]
        if not ratios:
            raise ValueError(f'no training pairs at hour {hour}')
        result[hour] = [p1 * math.exp(quantile(ratios, tau)) for tau in LEVELS]
    return result


def pseudo_case(comparables, index):
    """Build fresh, without inheriting the original case's game/history/notes."""
    c = comparables[index]
    start = timestamp(c['day_start'])
    checkpoint = start + timedelta(hours=1)
    actual = release_hourly(c)
    if 1 not in actual:
        raise ValueError(f"{c['name']}: missing hour 1")
    channels = release_hourly(c, 'channels')
    history = [dict(p) for p in c['hourly'] if timestamp(p['from']) < checkpoint]
    forecast = [{'hour': h, 'from': iso(start + timedelta(hours=h - 1)),
                 'until': iso(start + timedelta(hours=h))} for h in range(2, 25)]
    return {
        'schema': 'makefun/twitch-ccu-case/v1', 'id': f'loo-{index:02d}',
        'game': {'name': c['name']}, 'day_start': c['day_start'],
        'checkpoint': iso(checkpoint), 'levels': list(LEVELS),
        'known': [{'hour': 1, 'from': iso(start), 'until': iso(checkpoint),
                   'viewers': actual[1], 'channels': channels.get(1),
                   'reads': sum(p.get('reads', 1) for p in history
                                if timestamp(p['from']) >= start)}],
        'forecast': forecast, 'history_hourly': history,
        'history': [{'at': p['from'], 'viewers': p['viewers'],
                     'channels': p['channels']} for p in history],
        'comparables': [other for j, other in enumerate(comparables) if j != index],
        'notes': c.get('unlock', ''), 'platforms': c.get('platforms', ''),
        'streams': None, 'categories_hourly': None,
    }


def evaluate(comparables, forecaster):
    """Evaluate function(case) -> {hour: five quantiles}.

    A fold without hour 1 or any targets is explicitly skipped, not imputed.
    Its row remains in other folds' training comparables, where individual
    models decide whether it supplies usable observations.
    """
    rows = []
    skipped = []
    for index, c in enumerate(comparables):
        actual = release_hourly(c)
        if 1 not in actual or not any(2 <= h <= 24 for h in actual):
            skipped.append({'name': c['name'], 'reason':
                            'missing hour 1' if 1 not in actual else 'no target hours'})
            continue
        predictions = forecaster(pseudo_case(comparables, index))
        rows.append({'name': c['name'], 'hours': sum(2 <= h <= 24 for h in actual),
                     'loss': score_forecast(actual, predictions)})
    return {'loss': fmean(r['loss'] for r in rows), 'cases': rows, 'skipped': skipped,
            'hour_weighted_loss': (sum(r['loss'] * r['hours'] for r in rows) /
                                   sum(r['hours'] for r in rows))}


def distribution(label, values):
    values = list(values)
    print(f"{label} (n={len(values)}): " + ', '.join(
        f'{name}={quantile(values, tau):.3f}' for name, tau in
        [('min', 0), ('p05', .05), ('p25', .25), ('median', .5),
         ('p75', .75), ('p95', .95), ('max', 1)]))


def data_summary(comparables):
    profiles = [release_hourly(c) for c in comparables]
    features = [pre_release_features(c) for c in comparables]
    print('\nData summary (observed bins only; log inputs floored at 1):')
    print(f'Comparables: {len(comparables)}; all 24 hours: '
          f'{sum(len(p) == 24 for p in profiles)}; hour 1 present: '
          f'{sum(1 in p for p in profiles)}; target hours: '
          f'{sum(sum(2 <= h <= 24 for h in p) for p in profiles)}')
    distribution('Pre-release coverage / 24 hours', (f['pre_hours'] for f in features))
    distribution('ln(v1 / pre24 median)', (f['log_v1_pre'] for f in features
                                           if f['log_v1_pre'] is not None))
    for h in (6, 12, 18, 24):
        distribution(f'ln(v{h} / v1)', (log_print(p[h]) - log_print(p[1])
                                        for p in profiles if 1 in p and h in p))
    # Exploratory associations, not out-of-sample evidence of model improvement.
    for name, get_x in [('log_v1_pre', lambda f: f['log_v1_pre']),
                        ('log_v1', lambda f: log_print(f['v1']) if f['v1'] is not None else None),
                        ('viewers_per_channel1', lambda f: f['viewers_per_channel1'])]:
        pairs = []
        for f, p in zip(features, profiles):
            late = [log_print(p[h]) - log_print(p[1]) for h in range(18, 25)
                    if h in p and 1 in p]
            x = get_x(f)
            if x is not None and late:
                pairs.append((x, fmean(late)))
        if len(pairs) > 1:
            print(f'Exploratory Pearson r({name}, mean ln(v18..24 / v1)) '
                  f'= {correlation(*zip(*pairs)):.3f} (n={len(pairs)})')
    for c in comparables:
        if 'not a launch' in c.get('unlock', '').lower():
            print(f"Caution: retained non-launch comparable: {c['name']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', nargs='?', default=KIT / 'input/p00.in')
    args = parser.parse_args()
    comparables = load_case(args.case)['comparables']
    results = {name: evaluate(comparables, model)
               for name, model in [('baseline', baseline), ('pooled', pooled)]}
    print('LOO log-pinball loss (lower is better; equal case weight)')
    print(f"{'Case':80s} {'hours':>5s} {'baseline':>10s} {'pooled':>10s}")
    for a, b in zip(results['baseline']['cases'], results['pooled']['cases']):
        print(f"{a['name']:80s} {a['hours']:5d} {a['loss']:10.6f} {b['loss']:10.6f}")
    for name, result in results.items():
        print(f"{name}: mean={result['loss']:.6f}; "
              f"observed-hour-weighted={result['hour_weighted_loss']:.6f}")
    print(f"Scored folds: {len(results['baseline']['cases'])}/{len(comparables)}; "
          f"scored target hours: {sum(r['hours'] for r in results['baseline']['cases'])}")
    for skipped in results['baseline']['skipped']:
        print(f"Skipped fold: {skipped['name']} ({skipped['reason']}; no imputation).")
    gain = 1 - results['pooled']['loss'] / results['baseline']['loss']
    print(f'Pooled relative loss reduction: {100 * gain:.2f}%')
    data_summary(comparables)


if __name__ == '__main__':
    main()
