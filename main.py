"""Runtime-learned, similarity-weighted release-day log-ratio quantiles."""
import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from statistics import median, fmean

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get('MAKEFUN_INPUT_DIR', '/input'))
OUTPUT_DIR = Path(os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
CONFIG = dict(v1_bw=1.0, smooth=0.3)


def round2(x: float) -> float | int:
    """Round half up to two decimals, returning whole numbers as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def log(value):
    return math.log(max(1, value))


def bins(points, start, count=24):
    groups = {}
    for p in points:
        offset = (timestamp(p['from']) - start).total_seconds() / 3600
        if 0 <= offset < count and math.isfinite(p['viewers']):
            groups.setdefault(int(offset) + 1, []).append(p['viewers'])
    return {h: fmean(groups[h]) for h in sorted(groups)}


def features(points, start, checkpoint, v1):
    pre = bins(points, start - timedelta(hours=24))
    recent = bins(points, checkpoint - timedelta(hours=3), 3)
    return (log(v1), log(v1) - log(median(list(pre.values())) ) if pre else None,
            log(v1) - log(median(list(recent.values()))) if recent else None)


def prepare(c):
    start = timestamp(c['day_start'])
    profile = bins(c.get('hourly', []), start)
    if 1 not in profile:
        return None
    return (features(c['hourly'], start, start + timedelta(hours=1), profile[1]),
            {h: log(profile[h]) - log(profile[1]) for h in sorted(profile) if h >= 2})


def weighted_quantiles(samples):
    """Interpolate weighted CDF centers, rescaled to include both endpoints.

    Equal weights exactly reproduce linear quantiles at tau * (n - 1).
    """
    samples = sorted(samples)
    centers = []
    cumulative = 0.0
    for value, weight in samples:
        centers.append((cumulative + weight / 2, value))
        cumulative += weight
    lo, hi = centers[0][0], centers[-1][0]
    result = []
    for tau in LEVELS:
        target = lo + tau * (hi - lo)
        j = 0
        while j + 1 < len(centers) and centers[j + 1][0] < target:
            j += 1
        a, x = centers[j]
        b, y = centers[min(j + 1, len(centers) - 1)]
        result.append(x if b <= a else x + (y - x) * (target - a) / (b - a))
    return result


def predict(case, v1_bw=None, jump_bw=None, recent_bw=None, smooth=0.0, shrink=0.0,
            prepared=None):
    first = next((p['viewers'] for p in case.get('known', []) if p['hour'] == 1), None)
    history = case.get('history_hourly', [])
    if first is None:
        available = history or case.get('history', [])
        first = max(available, key=lambda p: p.get('from', p.get('at', '')))['viewers'] if available else 1
    p1 = max(1, first)
    start = timestamp(case['day_start'])
    target = features(history, start, timestamp(case['checkpoint']), p1)
    training = prepared if prepared is not None else [prepare(c) for c in case.get('comparables', [])]
    samples = {h: [] for h in range(2, 25)}
    for item in training:
        if item is None:
            continue
        feature, ratios = item
        distance = 0.0
        for x, y, bw in zip(target, feature, (v1_bw, jump_bw, recent_bw)):
            if bw is not None and x is not None and y is not None:
                distance += ((x - y) / bw) ** 2
        weight = max(1e-12, math.exp(-min(700, distance / 2)))
        for h in sorted(ratios):
            samples[h].append((ratios[h], weight))
    all_samples = [s for h in range(2, 25) for s in samples[h]]
    curves = {}
    for h in range(2, 25):
        data = samples[h] or [(r, 1) for r, _ in all_samples]
        if data:
            qs = weighted_quantiles(data)
            pooled = weighted_quantiles([(r, 1) for r, _ in data])
            curves[h] = [(1 - shrink) * a + shrink * b for a, b in zip(qs, pooled)]
        else:
            curves[h] = [z * .15 * math.sqrt(h - 1) for z in Z]
    result = {}
    for h in range(2, 25):
        neighbors = [curves[k] for k in (h - 1, h + 1) if k in curves]
        qs = [(1 - smooth) * q + smooth * fmean(n[j] for n in neighbors)
              for j, q in enumerate(curves[h])]
        result[h] = [round2(max(.01, math.exp(min(math.log(1e9), log(p1) + q)))) for q in qs]
    return result


def forecast_case(case, name='case'):
    if case.get('levels', LEVELS) != LEVELS:
        raise ValueError(f'{name}: unsupported quantile levels')
    values = predict(case, **CONFIG)
    return {'hours': [{'hour': h, 'q': values[h]} for h in range(2, 25)]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob('*.in')):
        case = json.loads(path.read_text(encoding='utf-8'))
        (OUTPUT_DIR / f'{path.stem}.out').write_text(
            json.dumps(forecast_case(case, path.name), separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
