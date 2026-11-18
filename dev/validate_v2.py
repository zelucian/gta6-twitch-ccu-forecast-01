"""Validate the fixed v2 ensemble using the original LOO and entry checks."""

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import main
from dev.common import KIT, load_case, timestamp
from dev.tune import run
from dev.validate import check, entry


def validate():
    comparables = load_case()['comparables']
    result = run(comparables, main.CONFIG)
    assert result['loss'] <= 0.36
    worst = max(result['cases'], key=lambda row: row['loss'])
    validation = {'cases': []}
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        first = root / 'first'
        repeat = root / 'repeat'
        validation['all_case_seconds'] = entry(KIT / 'input', first)
        entry(KIT / 'input', repeat)
        inputs = sorted((KIT / 'input').glob('*.in'))
        assert len(inputs) == 4
        expected = {source.stem + '.out' for source in inputs}
        for directory in (first, repeat):
            assert {path.name for path in directory.iterdir()} == expected
        for source in inputs:
            name = source.stem + '.out'
            output = first / name
            check(output)
            assert output.read_bytes() == (repeat / name).read_bytes()
            isolated = root / f'input-{source.stem}'
            isolated.mkdir()
            (isolated / source.name).write_bytes(source.read_bytes())
            timed = root / f'timed-{source.stem}'
            seconds = entry(isolated, timed)
            check(timed / name)
            assert output.read_bytes() == (timed / name).read_bytes()
            assert seconds < 8
            validation['cases'].append({'id': source.stem, 'seconds': seconds, 'bytes': output.stat().st_size})
        synthetic = load_case()
        synthetic['comparables'] = []
        synthetic['known'] = []
        for index, history in enumerate((synthetic['history_hourly'], [])):
            synthetic['history_hourly'] = history
            synthetic['history'] = []
            output = root / f'missing-{index}.out'
            output.write_text(json.dumps(main.forecast_case(synthetic)), encoding='utf-8')
            check(output)
            assert main.predict(synthetic) == main.predict(synthetic, **main.CONFIG)
        comparable = next(comparable for comparable in load_case()['comparables'] if main.prepare(comparable) is not None)
        start = timestamp(comparable['day_start'])
        comparable['hourly'] = [point for point in comparable['hourly']
                                if 0 <= (timestamp(point['from']) - start).total_seconds() < 7200]
        synthetic['comparables'] = [comparable]
        output = root / 'sparse.out'
        output.write_text(json.dumps(main.forecast_case(synthetic)), encoding='utf-8')
        check(output)
    validation.update(contract=True, deterministic=True, missing_and_sparse=True)
    report = {'config': main.CONFIG, 'comparable_count': len(comparables), 'old_loss': 0.400267,
              'loo': result, 'worst_fold': worst, 'validation': validation}
    pathlib.Path(__file__).with_name('validation-v2.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'mean_loss': result['loss'], 'worst_fold': worst,
                      'scored_folds': len(result['cases']), 'validation': validation}))


if __name__ == '__main__':
    validate()
