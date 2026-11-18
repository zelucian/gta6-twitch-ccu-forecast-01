"""Cross-set LOO and entry contract, timing, determinism and fallback checks."""

import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import main
from dev.common import KIT, load_case, timestamp
from dev.tune import run, summary


def check(path):
    path = pathlib.Path(path)
    assert path.stat().st_size <= 65536
    output = json.loads(path.read_text(encoding='utf-8'))
    assert isinstance(output, dict) and set(output) == {'hours'}
    rows = output['hours']
    assert isinstance(rows, list)
    assert [row['hour'] for row in rows] == list(range(2, 25))
    for row in rows:
        assert set(row) == {'hour', 'q'}
        values = row['q']
        assert isinstance(values, list) and len(values) == 5
        assert all(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 1e9 for value in values)
        assert values == sorted(values)


def entry(source, dest):
    environment = dict(os.environ, MAKEFUN_INPUT_DIR=str(source), MAKEFUN_OUTPUT_DIR=str(dest))
    start = time.perf_counter()
    subprocess.run([sys.executable, '-P', '-s', str(pathlib.Path(__file__).resolve().parents[1] / 'main.py')],
                   env=environment, check=True)
    return time.perf_counter() - start


def validate():
    rows = []
    for source in sorted((KIT / 'input').glob('*.in')):
        comparables = load_case(source)['comparables']
        pooled = run(comparables, {})
        chosen = run(comparables, main.CONFIG)
        row = summary(source.stem, main.CONFIG, chosen, pooled)
        chosen_fold, pooled_fold = max(zip(chosen['cases'], pooled['cases']),
                                      key=lambda pair: (pair[0]['loss'] - pair[1]['loss'], pair[0]['name']))
        row.update(pooled_loss=pooled['loss'], folds=len(chosen['cases']), skipped=chosen['skipped'],
                   regression_chosen_loss=chosen_fold['loss'], regression_pooled_loss=pooled_fold['loss'])
        rows.append(row)
        print(json.dumps(row))
    validation = {'cases': []}
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        first = root / 'a'
        repeat = root / 'b'
        validation['all_case_seconds'] = entry(KIT / 'input', first)
        entry(KIT / 'input', repeat)
        outputs = sorted(first.glob('*.out'))
        assert len(outputs) == 4
        assert {path.name for path in repeat.iterdir()} == {path.name for path in outputs}
        for output in outputs:
            check(output)
            assert output.read_bytes() == (repeat / output.name).read_bytes()
            isolated = root / f'input-{output.stem}'
            isolated.mkdir()
            input_name = output.stem + '.in'
            (isolated / input_name).write_bytes((KIT / 'input' / input_name).read_bytes())
            timed = root / f'timed-{output.stem}'
            seconds = entry(isolated, timed)
            check(timed / output.name)
            assert output.read_bytes() == (timed / output.name).read_bytes()
            assert seconds < 8
            validation['cases'].append({'id': output.stem, 'seconds': seconds, 'bytes': output.stat().st_size})
        synthetic = load_case()
        synthetic['comparables'] = []
        synthetic['known'] = [{'hour': 1, 'viewers': 0}]
        zero = root / 'zero-input'
        zero.mkdir()
        (zero / 'zero.in').write_text(json.dumps(synthetic), encoding='utf-8')
        entry(zero, root / 'zero-output')
        check(root / 'zero-output/zero.out')
        synthetic['known'] = []
        for index, history in enumerate((synthetic['history_hourly'], [])):
            synthetic['history_hourly'] = history
            if index == 1:
                synthetic['history'] = []
            output = root / f'missing-{index}.out'
            output.write_text(json.dumps(main.forecast_case(synthetic)), encoding='utf-8')
            check(output)
        comparable = next(comparable for comparable in load_case()['comparables'] if main.prepare(comparable) is not None)
        start = timestamp(comparable['day_start'])
        comparable['hourly'] = [point for point in comparable['hourly']
                                if 0 <= (timestamp(point['from']) - start).total_seconds() < 7200]
        synthetic['comparables'] = [comparable]
        output = root / 'sparse.out'
        output.write_text(json.dumps(main.forecast_case(synthetic)), encoding='utf-8')
        check(output)
    validation.update(deterministic=True, contract=True, zero_empty=True, missing_and_sparse=True)
    print(json.dumps(validation))
    report = {'cross_sets': rows, 'validation': validation}
    pathlib.Path(__file__).with_name('validation-results.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    validate()
