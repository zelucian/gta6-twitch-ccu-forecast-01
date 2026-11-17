"""Reproducible LOO grid; caches only individual comparable preparations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from dev.common import KIT, load_case
from dev.loo import evaluate
import json


def run(comparables, config):
    prepared = {id(c): main.prepare(c) for c in comparables}
    return evaluate(comparables, lambda case: main.predict(case, **config,
                    prepared=[prepared[id(c)] for c in case['comparables']]))


def summary(name, config, result, reference):
    worst = max(result['cases'], key=lambda r: r['loss'])
    changes = [(r['loss'] - b['loss'], r['name']) for r, b in zip(result['cases'], reference['cases'])]
    delta, game = max(changes)
    return dict(variant=name, config=config, loss=result['loss'],
                gain_pct=100*(1-result['loss']/reference['loss']),
                worst_loss=worst['loss'], worst_game=worst['name'],
                max_regression=delta, regression_game=game)


def main_run():
    comparables = load_case()['comparables']
    reference = run(comparables, {})
    variants = [('pooled', {})]
    variants += [(f'w_jump_{b}', dict(jump_bw=b)) for b in (.5, 1, 2)]
    variants += [(f'w_v1_{b}', dict(v1_bw=b)) for b in (.5, 1, 2, 3)]
    variants += [(f'w_both_{v}_{j}', dict(v1_bw=v, jump_bw=j)) for v in (1, 2) for j in (1, 2)]
    results = [(name, cfg, run(comparables, cfg)) for name, cfg in variants]
    best = min(results, key=lambda x: x[2]['loss'])
    variants2 = [(f'smooth_{s}', dict(best[1], smooth=s)) for s in (.15, .3)]
    variants2 += [(f'shrink_{s}', dict(best[1], shrink=s)) for s in (.15, .3, .5)]
    results += [(name, cfg, run(comparables, cfg)) for name, cfg in variants2]
    rows = [summary(n, c, r, reference) for n,c,r in results]
    Path('dev/tuning-results.json').write_text(json.dumps(rows, indent=2)+'\n')
    for row in rows:
        print(json.dumps(row), flush=True)

if __name__ == '__main__':
    main_run()
