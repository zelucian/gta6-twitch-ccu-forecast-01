"""Reproducible LOO grid caching only individual comparable preparations."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from dev.common import KIT, load_case
from dev.loo import evaluate


def run(comparables, config):
    prepared = {id(comparable): main.prepare(comparable) for comparable in comparables}
    return evaluate(comparables, lambda case: main.predict(
        case, **config, prepared=[prepared[id(comparable)] for comparable in case['comparables']]))


def summary(name, config, result, reference):
    worst = max(result['cases'], key=lambda row: row['loss'])
    changes = [(row['loss'] - ref['loss'], row['name'])
               for row, ref in zip(result['cases'], reference['cases'])]
    delta, game = max(changes)
    return {
        'variant': name, 'config': config, 'loss': result['loss'],
        'gain_pct': 100 * (1 - result['loss'] / reference['loss']),
        'worst_loss': worst['loss'], 'worst_game': worst['name'],
        'max_regression': delta, 'regression_game': game,
    }


def main_run():
    comparables = load_case()['comparables']
    reference = run(comparables, {})
    variants = [('pooled', {})]
    variants.extend((f'w_jump_{bandwidth}', {'jump_bw': bandwidth}) for bandwidth in (0.5, 1, 2))
    variants.extend((f'w_v1_{bandwidth}', {'v1_bw': bandwidth}) for bandwidth in (0.5, 1, 2, 3))
    variants.extend((f'w_both_{viewers}_{jump}', {'v1_bw': viewers, 'jump_bw': jump})
                    for viewers in (1, 2) for jump in (1, 2))
    results = [(name, config, run(comparables, config)) for name, config in variants]
    best = min(results, key=lambda item: item[2]['loss'])[1]
    variants2 = [(f'smooth_{strength}', {**best, 'smooth': strength}) for strength in (0.15, 0.3)]
    variants2.extend((f'shrink_{strength}', {**best, 'shrink': strength}) for strength in (0.15, 0.3, 0.5))
    results.extend((name, config, run(comparables, config)) for name, config in variants2)
    rows = [summary(name, config, result, reference) for name, config, result in results]
    Path(__file__).with_name('tuning-results.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    for row in rows:
        print(json.dumps(row))


if __name__ == '__main__':
    main_run()
