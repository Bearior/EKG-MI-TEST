import numpy as np
import pytest
import wfdb

from mi_lab.signals import LEADS, extract_features, read_ecg


def synthetic_ecg():
    t = np.arange(5000) / 500
    base = 0.1 * np.sin(2 * np.pi * t)
    for r in np.arange(0.5, 10, 1):
        base += np.exp(-((t - r) / 0.02) ** 2)
    return np.column_stack([base * (1 + i / 12) for i in range(12)])


def test_reader_reorders_leads_and_returns_millivolts(tmp_path):
    expected = synthetic_ecg()
    order = list(range(11, -1, -1))
    wfdb.wrsamp('sample', fs=500, units=['mV'] * 12,
                sig_name=[LEADS[i] for i in order], p_signal=expected[:, order],
                fmt=['16'] * 12, write_dir=str(tmp_path))
    actual = read_ecg(tmp_path / 'sample.dat')
    assert actual.shape == (5000, 12)
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_features_are_finite_deterministic_and_invariant_to_dc_offset():
    x = synthetic_ecg()
    a, names = extract_features(x)
    b, names_b = extract_features(x + 2)
    assert len(a) == len(names) and len(names) == len(set(names))
    assert len(a) > 100
    assert names == names_b
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert np.isfinite(a).all()


def test_features_preserve_signal_amplitude_information():
    x = synthetic_ecg()
    a, _ = extract_features(x)
    b, _ = extract_features(x * 2)
    assert not np.allclose(a, b)


@pytest.mark.parametrize('x', [np.zeros((5000, 12)), np.ones((10, 2)),
                              np.full((5000, 12), np.nan)])
def test_bad_signals_are_rejected(x):
    with pytest.raises(ValueError):
        extract_features(x)
