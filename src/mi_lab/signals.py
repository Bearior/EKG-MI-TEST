"""Fixed waveform features; no diagnosis or clinical metadata enters the model."""

from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt, welch

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
FEATURE_VERSION = '1.0'


def read_ecg(path: Path):
    record = wfdb.rdrecord(str(Path(path).with_suffix('')))
    names = [name.upper() for name in record.sig_name]
    if record.fs != 500 or record.p_signal.shape != (5000, 12):
        raise ValueError('Expected exactly ten seconds, 500 Hz, twelve leads')
    if len(set(names)) != 12 or set(names) != {lead.upper() for lead in LEADS}:
        raise ValueError('Missing, duplicated or unsupported lead names')
    indices = [names.index(lead.upper()) for lead in LEADS]
    units = [record.units[i] for i in indices]
    conversions = {'mV': 1.0, 'uV': 0.001, 'µV': 0.001, 'V': 1000.0}
    if any(unit not in conversions for unit in units):
        raise ValueError(f'Unsupported physical units: {units}')
    x = record.p_signal[:, indices] * np.array([conversions[unit] for unit in units])
    _validate_signal(x)
    return x


def _validate_signal(x):
    if x.shape != (5000, 12) or not np.isfinite(x).all():
        raise ValueError('Expected finite ECG array with shape (5000, 12)')
    if np.any(np.std(x, axis=0) < 1e-9):
        raise ValueError('Flat lead detected')


def extract_features(x):
    """Distribution/spectral features and a heuristic QRS-aligned median beat.

    The beat detector is a fixed energy heuristic, not a clinical delineator.
    Bandpass 0.5--40 Hz and polyphase resampling to 100 Hz are per-record operations.
    Amplitude is retained in mV; scaling is fitted later on training patients only.
    """
    x = np.asarray(x, dtype=float)
    _validate_signal(x)
    x = x - np.median(x, axis=0)
    filtered = sosfiltfilt(butter(3, [0.5, 40], fs=500, btype='bandpass', output='sos'),
                         x, axis=0)
    low = resample_poly(filtered, 1, 5, axis=0)
    frequencies, power = welch(low, fs=100, nperseg=256, axis=0)
    features, names = [], []

    def append(name, values):
        values = np.asarray(values).ravel()
        features.extend(values.tolist())
        names.extend(f'{lead}_{name}' for lead in LEADS)

    for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        append(f'quantile_{q}', np.quantile(low, q, axis=0))
    append('rms', np.sqrt(np.mean(low**2, axis=0)))
    append('derivative_rms', np.sqrt(np.mean(np.diff(low, axis=0)**2, axis=0)))
    append('peak_to_peak', np.ptp(low, axis=0))
    total = power.sum(axis=0) + 1e-12
    for start, stop in [(0.5, 3), (3, 8), (8, 15), (15, 25), (25, 40)]:
        append(f'power_fraction_{start}_{stop}',
               power[(frequencies >= start) & (frequencies < stop)].sum(axis=0) / total)

    qrs = sosfiltfilt(butter(2, [5, 20], fs=100, btype='bandpass', output='sos'),
                     low, axis=0)
    energy = np.mean(qrs**2, axis=1)
    energy = np.convolve(energy, np.ones(5) / 5, mode='same')
    peaks, _ = find_peaks(energy, distance=30, prominence=max(np.max(energy) * 0.1, 1e-12))
    peaks = peaks[(peaks >= 20) & (peaks < len(low) - 60)]
    if len(peaks) >= 2:
        beat = np.median(np.stack([low[p-20:p+60] for p in peaks]), axis=0)
        rr = np.diff(peaks) / 100
        rhythm = [len(peaks), np.median(rr), np.std(rr), 1.0]
    else:
        beat = np.zeros((80, 12))
        rhythm = [len(peaks), 0.0, 0.0, 0.0]
    # 50-Hz median morphology gives 480 features, with fixed temporal ordering.
    for t in range(0, 80, 2):
        append(f'median_beat_{t-20:+d}0ms', beat[t])
    features.extend(rhythm)
    names.extend(['qrs_count', 'rr_median_seconds', 'rr_std_seconds', 'beat_available'])
    result = np.asarray(features)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite extracted features')
    return result, names
