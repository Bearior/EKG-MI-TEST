"""Run the prespecified, ECG-only STEMI/NSTEMI baseline."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mi_lab.cohort import build_cohort, split_cohort
from mi_lab.download import verify_dataset, verify_extracted_file
from mi_lab.evaluation import binary_metrics, bootstrap_intervals, choose_threshold
from mi_lab.signals import FEATURE_VERSION, extract_features, read_ecg


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def remove_duplicate_signals(frame):
    """Exclude all members of an exact-duplicate group before assigning splits."""
    duplicate = frame.signal_hash.duplicated(keep=False)
    return frame.loc[~duplicate].reset_index(drop=True), int(duplicate.sum())


def prepare_data(data_dir, artifact_dir):
    """Read labelled source only; the publisher's hidden test is never labelled here."""
    raw = Path(data_dir) / 'raw'
    _, source_members = verify_dataset(Path(data_dir))
    metadata_files = list(raw.rglob('train.csv'))
    if len(metadata_files) != 1:
        raise ValueError(f'Expected one train.csv under {raw}, found {len(metadata_files)}')
    verify_extracted_file(metadata_files[0], source_members[metadata_files[0].relative_to(raw).as_posix()])
    source = pd.read_csv(metadata_files[0], dtype={'Patient_id': str, 'ecg_row_record': str})
    cohort, audit = build_cohort(source)
    path_index = {}
    for path in raw.rglob('*.dat'):
        if path.name in path_index:
            raise ValueError(f'Duplicate raw filename: {path.name}')
        path_index[path.name] = path
    rows, vectors, errors = [], [], []
    names = None
    for index, (_, row) in enumerate(cohort.iterrows(), 1):
        filename = row.ecg_row_record
        if filename not in path_index:
            raise FileNotFoundError(f'Missing source waveform: {filename}')
        for path in [path_index[filename], path_index[filename].with_suffix('.hea')]:
            relative = path.relative_to(raw).as_posix()
            if relative not in source_members:
                raise ValueError(f'Waveform not present in the verified source: {path}')
            verify_extracted_file(path, source_members[relative])
        try:
            x = read_ecg(path_index[filename])
            values, current_names = extract_features(x)
        except ValueError as exc:
            errors.append({'Patient_id': row.Patient_id, 'record': filename, 'reason': str(exc)})
            continue
        if names is not None and current_names != names:
            raise RuntimeError('Feature order changed between records')
        names = current_names
        item = row.to_dict()
        # Hash physical signals after standard lead ordering, not a filename or header.
        item['signal_hash'] = hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()
        item['feature_row'] = len(vectors)
        rows.append(item)
        vectors.append(values)
        if index % 250 == 0:
            print(f'Extracted {index}/{len(cohort)} ECGs', flush=True)
    if not rows:
        raise ValueError('No valid ECGs remain')
    valid, n_duplicates = remove_duplicate_signals(pd.DataFrame(rows))
    features = np.stack(vectors)[valid.feature_row.to_numpy()]
    audit['invalid_signal_records'] = len(errors)
    audit['exact_duplicate_records_excluded'] = n_duplicates
    audit['final_eligible_patients'] = len(valid)
    save_json(Path(artifact_dir) / 'signal_exclusions.json', errors)
    np.savez_compressed(Path(artifact_dir) / 'features.npz', X=features, names=np.array(names))
    save_json(Path(artifact_dir) / 'cohort_audit.json', audit)
    return valid.drop(columns='feature_row'), features, names, audit, metadata_files[0]


def train_evaluate(frame, features, names, artifact_dir, report_dir, audit, provenance,
                   seed=42, n_bootstrap=1000):
    from mi_lab.reporting import write_report

    artifact_dir, report_dir = Path(artifact_dir), Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if len(frame) != len(features) or features.shape[1] != len(names):
        raise ValueError('Feature matrix does not match cohort or feature names')
    if not np.isfinite(features).all() or frame.Patient_id.duplicated().any():
        raise ValueError('Finite features and exactly one record per patient required')
    masks = {split: frame.split.eq(split).to_numpy() for split in ['train', 'validation', 'test']}
    if not np.all(sum(masks.values()) == 1):
        raise ValueError('Each row must have exactly one valid split')
    y = frame.target.to_numpy(dtype=int)
    for split, mask in masks.items():
        if set(y[mask]) != {0, 1}:
            raise ValueError(f'Both classes required in {split}')
    train, validation, test = (masks[key] for key in ['train', 'validation', 'test'])
    selection, candidates = [], []
    for c in [0.01, 0.1, 1.0]:
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight='balanced', max_iter=5000, random_state=seed, solver='lbfgs'))
        # Explicit names make the saved pipeline easy to inspect.
        pipeline.steps = [('scale', pipeline.steps[0][1]), ('classifier', pipeline.steps[1][1])]
        pipeline.fit(features[train], y[train])
        p_val = pipeline.predict_proba(features[validation])[:, 1]
        auc = float(roc_auc_score(y[validation], p_val))
        selection.append({'C': c, 'validation_auroc': auc})
        candidates.append((pipeline, p_val))
    best = int(np.argmax([item['validation_auroc'] for item in selection]))
    model, p_val = candidates[best]
    threshold = choose_threshold(y[validation], p_val)
    # Test is evaluated after selecting both model and threshold. No refit on validation.
    p_test = model.predict_proba(features[test])[:, 1]
    dummy = DummyClassifier(strategy='prior').fit(features[train], y[train])
    p_dummy = dummy.predict_proba(features[test])[:, 1]
    counts = {key: {'n_patients': int(mask.sum()), 'n_stemi': int(y[mask].sum()),
                    'n_nstemi': int(mask.sum() - y[mask].sum())} for key, mask in masks.items()}
    result = {
        'title': 'ECG-only STEMI versus NSTEMI baseline',
        'cohort_audit': audit,
        'split_counts': counts,
        'feature_count': len(names),
        'selection': selection,
        'selected_C': selection[best]['C'],
        'threshold': threshold,
        'validation_metrics': binary_metrics(y[validation], p_val, threshold),
        'test_metrics': binary_metrics(y[test], p_test, threshold),
        'dummy_metrics': binary_metrics(y[test], p_dummy, 0.5),
        'confidence_intervals': bootstrap_intervals(y[test], p_test, threshold,
                                                    n_bootstrap=n_bootstrap, seed=seed),
        'config': {'seed': seed, 'n_bootstrap': n_bootstrap, 'feature_version': FEATURE_VERSION,
                   'target': 'STEMI=1, NSTEMI=0', 'one_ecg_per_patient': True,
                   'selection_metric': 'validation AUROC',
                   'threshold_rule': 'validation maximum Youden J; nearest 0.5 breaks ties'},
        'provenance': provenance,
        'test_curve': {'y_true': y[test].tolist(), 'probability': p_test.tolist()},
    }
    frame.to_csv(artifact_dir / 'split_manifest.csv', index=False)
    predictions = frame.loc[test, ['Patient_id', 'ecg_row_record', 'target']].copy()
    predictions['stemi_probability'] = p_test
    predictions['predicted_stemi'] = (p_test >= threshold).astype(int)
    predictions.to_csv(artifact_dir / 'test_predictions.csv', index=False)
    joblib.dump({'pipeline': model, 'threshold': threshold, 'feature_names': names,
                 'feature_version': FEATURE_VERSION, 'config': result['config']},
                artifact_dir / 'model.joblib')
    pd.DataFrame({'feature': names, 'standardized_coefficient':
                  model['classifier'].coef_[0]}).to_csv(artifact_dir / 'coefficients.csv', index=False)
    save_json(artifact_dir / 'metrics.json', result)
    # Aggregate summary can be tracked; record-level probabilities remain ignored.
    save_json(report_dir / 'summary.json', {k: v for k, v in result.items() if k != 'test_curve'})
    write_report(result, report_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/baseline'))
    parser.add_argument('--report-dir', type=Path, default=Path('reports/baseline'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    if args.download:
        from mi_lab.download import download_dataset
        download_dataset(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print('Auditing source cohort and extracting waveform-only features...', flush=True)
    frame, features, names, audit, metadata_path = prepare_data(args.data_dir, args.output_dir)
    split = split_cohort(frame, seed=args.seed)
    # Never rely on split_cohort preserving row order.
    index = {patient: i for i, patient in enumerate(frame.Patient_id)}
    features = features[[index[patient] for patient in split.Patient_id]]
    manifest_path = args.data_dir / 'source_manifest.json'
    source_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    provenance = {
        'dataset_doi': '10.6084/m9.figshare.29925314',
        'paper_doi': '10.1038/s41597-026-07278-0',
        'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        'source_manifest': source_manifest,
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'scipy', 'scikit-learn', 'wfdb', 'matplotlib']},
    }
    print(f'Training on {len(split)} eligible patients with {len(names)} features...', flush=True)
    result = train_evaluate(split, features, names, args.output_dir, args.report_dir,
                            audit, provenance, seed=args.seed, n_bootstrap=args.bootstrap)
    print(json.dumps(result['test_metrics'], indent=2), flush=True)
    print(f'Report: {(args.report_dir / "report.html").resolve()}', flush=True)


if __name__ == '__main__':
    main()
