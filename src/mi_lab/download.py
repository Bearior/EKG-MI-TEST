"""Reproducibly fetch the Acute Coronary Syndrome ECG Figshare release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import urlopen

ARTICLE_ID = 29925314
ARTICLE_VERSION = 1
ARTICLE_API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/versions/{ARTICLE_VERSION}"
FILE_API_URL = "https://api.figshare.com/v2/file/download/{file_id}"
REQUIRED_FILES = {"CSV.zip", "ECG_row_data.zip"}
CHUNK_SIZE = 1024 * 1024
PINNED_SHA256 = {
    'CSV.zip': '675a57e90eea1175259301293812ec0a23c73f83ec7fb352a8309f05843d0005',
    'ECG_row_data.zip': '9a5a1bf1655b28d09de152bd4bf443c491cef4a1adff108b40938ee90e6693c4',
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_metadata() -> dict[str, Any]:
    with urlopen(ARTICLE_API_URL) as response:  # nosec B310: fixed HTTPS endpoint
        return json.load(response)


def _safe_member_path(destination: Path, member: zipfile.ZipInfo) -> Path:
    relative = PurePosixPath(member.filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive member: {member.filename!r}")
    if stat.S_ISLNK(member.external_attr >> 16):
        raise ValueError(f"archive member is a symbolic link: {member.filename!r}")
    target = destination.joinpath(*relative.parts)
    if os.path.commonpath((str(destination.resolve()), str(target.resolve()))) != str(destination.resolve()):
        raise ValueError(f"unsafe archive member: {member.filename!r}")
    return target


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a ZIP without permitting path traversal or symlink members."""
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        targets = [_safe_member_path(destination, member) for member in members]
        for member, target in zip(members, targets, strict=True):
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary, urlopen(url) as response:  # nosec B310: URL is Figshare metadata
            for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                temporary.write(chunk)
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _file_is_current(path: Path, size: int, md5: str) -> bool:
    return path.is_file() and path.stat().st_size == size and _md5(path) == md5


def verify_dataset(data_dir: Path):
    """Verify the locally pinned archives and return their member integrity index."""
    manifest_path = data_dir / 'source_manifest.json'
    if not manifest_path.exists():
        raise ValueError('Missing source manifest; run python -m mi_lab.download first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('article_id') != ARTICLE_ID or manifest.get('version') != ARTICLE_VERSION:
        raise ValueError('Source manifest does not identify the pinned release')
    files = {item['name']: item for item in manifest.get('files', [])}
    if set(files) != REQUIRED_FILES:
        raise ValueError('Source manifest is missing required archives')
    members = {}
    for name, expected_sha in PINNED_SHA256.items():
        archive = data_dir / 'downloads' / name
        item = files[name]
        if (item.get('sha256') != expected_sha or not archive.exists()
                or archive.stat().st_size != item.get('size') or _sha256(archive) != expected_sha):
            raise ValueError(f'Unverified source archive {name}; run the downloader to restore it')
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                if not member.is_dir():
                    if member.filename in members:
                        raise ValueError('Duplicate archive member names across sources')
                    members[member.filename] = member
    return manifest, members


def verify_extracted_file(path: Path, member: zipfile.ZipInfo):
    """Check a consumed raw file against the checksum in the SHA-verified archive."""
    if (not path.exists() or path.stat().st_size != member.file_size
            or zlib.crc32(path.read_bytes()) != member.CRC):
        raise ValueError(f'Extracted source file changed: {path}; rerun the downloader')


def download_dataset(data_dir: Path) -> None:
    """Download, verify, and extract the CSV and raw waveform archives.

    Version 1 metadata is fetched on every run so cache reuse is tied to the
    fixed release, byte size, and upstream MD5 checksum.
    """
    metadata = _fetch_metadata()
    if metadata.get('version') != ARTICLE_VERSION:
        raise ValueError(f'Unexpected release version; expected {ARTICLE_VERSION}')
    files = {item["name"]: item for item in metadata["files"]}
    missing = REQUIRED_FILES - files.keys()
    if missing:
        raise ValueError(f"Figshare article is missing required files: {sorted(missing)}")

    downloads = data_dir / "downloads"
    raw = data_dir / "raw"
    manifest_files: list[dict[str, Any]] = []
    for name in sorted(REQUIRED_FILES):
        item = files[name]
        archive = downloads / name
        size = int(item["size"])
        upstream_md5 = item["computed_md5"]
        if not _file_is_current(archive, size, upstream_md5):
            _download(FILE_API_URL.format(file_id=item["id"]), archive)
        if not _file_is_current(archive, size, upstream_md5):
            raise ValueError(f"checksum verification failed for {archive}")
        safe_extract(archive, raw)
        manifest_files.append({
            "id": item["id"], "name": name, "size": size,
            "md5": upstream_md5, "sha256": _sha256(archive),
            "download_url": FILE_API_URL.format(file_id=item["id"]),
        })

    manifest = {
        "source": "Figshare", "article_id": ARTICLE_ID,
        "doi": metadata["doi"], "version": metadata["version"],
        "title": metadata["title"], "published_date": metadata["published_date"],
        "license": metadata["license"], "files": manifest_files,
    }
    (data_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    download_dataset(args.data_dir)


if __name__ == "__main__":
    main()
