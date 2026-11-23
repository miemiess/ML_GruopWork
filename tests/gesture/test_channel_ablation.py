"""合成原始记录检查完整对比与指标可重算，不作为真实实验成绩。"""
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from emginsight.gesture.channel_ablation import feature_columns, run


def test_columns():
    np.testing.assert_array_equal(feature_columns([8, 2]), [3, 4, 5, 21, 22, 23])
    for bad in ([], [0], [9], [1, 1], [1.5]):
        with pytest.raises(ValueError):
            feature_columns(bad)


def test_end_to_end(tmp_path):
    root = tmp_path / 'raw'
    for number in range(1, 10):
        subject = root / f'{number:02d}'
        subject.mkdir(parents=True)
        rows = ['time ' + ' '.join(f'channel{i}' for i in range(1, 9)) + ' class']
        for sample in range(1400):
            label = 1 + sample // 200
            values = ' '.join(str(label * channel + number / 100) for channel in range(1, 9))
            rows.append(f'{sample} {values} {label}')
        (subject / '1_raw_data_test.txt').write_text('\n'.join(rows))
    output = tmp_path / 'output'
    summary = run(root, output)
    users = [s for group in summary['subjects'].values() for s in group]
    assert len(users) == len(set(users)) == 9
    comparison = pd.read_csv(output / 'comparison.csv')
    candidates = pd.read_csv(output / 'validation.csv')
    assert len(candidates) == 36
    assert list(comparison.channel_count) == [1, 2, 4, 8]
    predictions = pd.read_csv(output / 'test_predictions.csv', dtype={'subject': str})
    previous = set()
    for row in comparison.itertuples():
        subset = set(str(row.channels).split(','))
        assert previous <= subset
        previous = subset
        pred = predictions[predictions.channel_count == row.channel_count]
        assert set(pred.subject) == set(summary['subjects']['test'])
        assert set(pred.label) == {1, 2, 3, 4, 5, 6}
        assert row.test_macro_f1 == pytest.approx(f1_score(pred.label, pred.prediction, average='macro'))
        assert pd.read_csv(output / f'confusion_{row.channel_count}.csv', index_col=0).to_numpy().sum() == len(pred)
    assert (output / 'channel_comparison.png').is_file()
    assert json.loads((output / 'metrics.json').read_text())['data_manifest']
