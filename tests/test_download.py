import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from mi_lab import download


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, contents in entries.items():
            archive.writestr(name, contents)
    return output.getvalue()


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    archive.write_bytes(_zip_bytes({"../escape.txt": b"bad"}))

    with pytest.raises(ValueError, match="unsafe archive member"):
        download.safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_unexpected_release_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(download, '_fetch_metadata', lambda: {'version': 2, 'files': []})
    with pytest.raises(ValueError, match='release version'):
        download.download_dataset(tmp_path / 'data')


def test_checksum_mismatch_is_detected(tmp_path):
    archive = tmp_path / 'archive.zip'
    archive.write_bytes(b'corrupt')
    assert not download._file_is_current(archive, 7, hashlib.md5(b'correct').hexdigest())


def test_verification_requires_manifest(tmp_path):
    with pytest.raises(ValueError, match='manifest'):
        download.verify_dataset(tmp_path)


def test_verification_rejects_altered_extracted_waveform(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    archive = tmp_path / 'source.zip'
    archive.write_bytes(_zip_bytes({'raw/test.dat': b'original'}))
    with zipfile.ZipFile(archive) as zipped:
        info = zipped.getinfo('raw/test.dat')
    path = raw / 'test.dat'
    path.write_bytes(b'altered!')
    with pytest.raises(ValueError, match='Extracted source file'):
        download.verify_extracted_file(path, info)


def test_download_dataset_verifies_and_reuses_current_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_bytes = _zip_bytes({"metadata.csv": b"patient_id,label\n1,STEMI\n"})
    raw_bytes = _zip_bytes({"raw/1.csv": b"0,1,2\n"})
    calls: list[str] = []
    files = []
    for file_id, name, payload in ((1, "CSV.zip", csv_bytes), (2, "ECG_row_data.zip", raw_bytes)):
        files.append({"id": file_id, "name": name, "size": len(payload),
                      "computed_md5": hashlib.md5(payload).hexdigest(),
                      "download_url": f"https://example.test/{name}"})
    metadata = {"files": files, "doi": "doi", "version": 1, "title": "title",
                "published_date": "today", "license": {"name": "CC0"}}

    def fake_urlopen(url: str):
        calls.append(url)
        if url == download.ARTICLE_API_URL:
            return io.BytesIO(json.dumps(metadata).encode())
        return io.BytesIO(csv_bytes if url.endswith("/1") else raw_bytes)

    monkeypatch.setattr(download, "urlopen", fake_urlopen)
    download.download_dataset(tmp_path / "data")
    download.download_dataset(tmp_path / "data")

    assert calls.count(download.ARTICLE_API_URL) == 2
    assert calls.count("https://api.figshare.com/v2/file/download/1") == 1
    assert calls.count("https://api.figshare.com/v2/file/download/2") == 1
    manifest = json.loads((tmp_path / "data" / "source_manifest.json").read_text())
    assert manifest["files"][0]["sha256"] == hashlib.sha256(csv_bytes).hexdigest()
    assert (tmp_path / "data" / "raw" / "metadata.csv").exists()
