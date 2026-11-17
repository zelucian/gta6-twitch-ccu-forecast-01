"""Cross-set LOO and entry contract, timing, determinism and fallback checks."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from dev.common import KIT, load_case
from dev.tune import run, summary


def check(path):
    assert path.stat().st_size <= 65536
    obj = json.loads(path.read_text())
    assert set(obj) == {'hours'}
    assert [r['hour'] for r in obj['hours']] == list(range(2, 25))
    for row in obj['hours']:
        assert set(row) == {'hour', 'q'}
        q = row['q']
        assert len(q) == 5 and q == sorted(q)
        assert all(isinstance(x, (float, int)) and math.isfinite(x) and 0 < x <= 1e9 for x in q)


def entry(source, dest):
    env = dict(os.environ, MAKEFUN_INPUT_DIR=str(source), MAKEFUN_OUTPUT_DIR=str(dest))
    start = time.perf_counter()
    subprocess.run([sys.executable, '-P', '-s', str(Path(main.__file__))], env=env, check=True)
    return time.perf_counter() - start


def validate():
    cross = []
    for path in sorted((KIT/'input').glob('*.in')):
        cs = load_case(path)['comparables']
        pooled = run(cs, {})
        chosen = run(cs, main.CONFIG)
        row = summary(path.stem, main.CONFIG, chosen, pooled)
        row.update(pooled=pooled['loss'], folds=len(chosen['cases']), skipped=chosen['skipped'])
        regressions = [(a['loss']-b['loss'], a, b) for a,b in zip(chosen['cases'],pooled['cases'])]
        delta, a, b = max(regressions, key=lambda x:x[0])
        row['regression_chosen_loss'] = a['loss']
        row['regression_pooled_loss'] = b['loss']
        cross.append(row)
        print(json.dumps(row), flush=True)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        elapsed = entry(KIT/'input', root/'a')
        entry(KIT/'input', root/'b')
        files = sorted((root/'a').glob('*.out'))
        assert len(files) == 4
        validation = dict(all_case_seconds=elapsed, cases=[])
        for path in files:
            check(path)
            assert path.read_bytes() == (root/'b'/path.name).read_bytes()
            source = root/path.stem
            source.mkdir()
            (source/(path.stem+'.in')).write_bytes((KIT/'input'/(path.stem+'.in')).read_bytes())
            seconds = entry(source, root/'timed'/path.stem)
            assert seconds < 8
            validation['cases'].append(dict(id=path.stem, seconds=seconds, bytes=path.stat().st_size))
        synth = load_case()
        synth['comparables'] = []
        synth['known'] = [{'hour':1,'viewers':0}]
        source = root/'synthetic'
        source.mkdir()
        (source/'zero.in').write_text(json.dumps(synth))
        entry(source, root/'synthetic-out')
        check(root/'synthetic-out/zero.out')
        # Exercise missing hour 1, missing history, sparse pairs and cross-hour fallback.
        synth['known'] = []
        for history in (synth['history_hourly'], []):
            synth['history_hourly'] = history
            if not history:
                synth['history'] = []
            out = root/'missing.out'
            out.write_text(json.dumps(main.forecast_case(synth)))
            check(out)
        sparse = load_case()
        c = next(c for c in sparse['comparables'] if main.prepare(c) is not None)
        start = main.timestamp(c['day_start'])
        c['hourly'] = [p for p in c['hourly'] if 0 <= (main.timestamp(p['from'])-start).total_seconds()/3600 < 2]
        sparse['comparables'] = [c]
        out = root/'sparse.out'
        out.write_text(json.dumps(main.forecast_case(sparse)))
        check(out)
        validation.update(deterministic=True, contract=True, zero_empty=True, missing_and_sparse=True)
        print(json.dumps(validation), flush=True)
    Path('dev/validation-results.json').write_text(json.dumps(dict(cross_sets=cross, validation=validation), indent=2)+'\n')

if __name__ == '__main__':
    validate()
