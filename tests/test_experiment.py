import json

import joblib
import numpy as np
import pandas as pd

from mi_lab.experiment import remove_duplicate_signals, train_evaluate


def test_duplicate_waveforms_excluded_before_splitting():
    frame = pd.DataFrame({'Patient_id': ['a', 'b', 'c'], 'signal_hash': ['same', 'same', 'unique']})
    kept, count = remove_duplicate_signals(frame)
    assert count == 2
    assert kept.Patient_id.tolist() == ['c']


def test_end_to_end_training_uses_train_scaling_and_saves_report(tmp_path):
    rng = np.random.default_rng(7)
    y = np.tile([0, 1], 60)
    x = rng.normal(size=(120, 8))
    x[:, 0] += 3 * y
    split = np.array(['train'] * 72 + ['validation'] * 24 + ['test'] * 24)
    x[split == 'test', 1] += 100  # Held-out distribution must not affect scaler.
    frame = pd.DataFrame({'Patient_id': [f'p{i}' for i in range(120)], 'target': y,
                          'ecg_row_record': [f'{i}.dat' for i in range(120)], 'split': split})
    result = train_evaluate(frame, x, [f'f{i}' for i in range(8)],
                            tmp_path / 'artifacts', tmp_path / 'report',
                            audit={}, provenance={'source': '<test>'}, seed=42,
                            n_bootstrap=10)
    model = joblib.load(tmp_path / 'artifacts/model.joblib')['pipeline']
    np.testing.assert_allclose(model['scale'].mean_, x[split == 'train'].mean(axis=0))
    assert result['test_metrics']['n_patients'] == 24
    assert 0 <= result['test_metrics']['auroc'] <= 1
    saved = json.loads((tmp_path / 'artifacts/metrics.json').read_text())
    assert saved['selected_C'] in [0.01, 0.1, 1.0]
    assert (tmp_path / 'report/report.html').exists()
    predictions = pd.read_csv(tmp_path / 'artifacts/test_predictions.csv')
    assert set(predictions.Patient_id) == set(frame.loc[split == 'test', 'Patient_id'])
