#!/usr/bin/env python3
"""Download the study datasets into local raw-data folders.

This script is the canonical, reviewable acquisition path for the project.
It downloads public source data into ``data/raw/`` and intentionally keeps
those files out of Git. Raw data should be treated as immutable once acquired.

The script does not hydrate Twitter/X objects and does not require API keys.
For FakeHealth, the optional Zenodo archive contains follower/following ID
network files, not full social-media post contents.

Usage examples:

    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --fakehealth-zenodo
    python scripts/download_datasets.py --coaid --fakehealth-repo

By default, existing files/directories are preserved. Use ``--force`` only
when you intentionally want to replace a local raw-data copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


COAID_REPO = "https://github.com/cuilimeng/CoAID.git"
COAID_COMMIT = "d238224346781255e1e7e6ed8bc410c2b2e6329e"
COAID_DIR = RAW_DIR / "coaid"

FAKEHEALTH_REPO = "https://github.com/EnyanDai/FakeHealth.git"
FAKEHEALTH_COMMIT = "ec9379de8f8f13af8c436dd6dd9bfaddacd2df30"
FAKEHEALTH_DIR = RAW_DIR / "fakehealth"

FAKEHEALTH_ZENODO_RECORD = "https://zenodo.org/records/3862989"
FAKEHEALTH_ZENODO_API = "https://zenodo.org/api/records/3862989"
FAKEHEALTH_ZENODO_FILE_URL = (
    "https://zenodo.org/api/records/3862989/files/FakeHealth.zip/content"
)
FAKEHEALTH_ZENODO_FILE = RAW_DIR / "fakehealth_zenodo" / "FakeHealth.zip"
FAKEHEALTH_ZENODO_EXTRACT_DIR = RAW_DIR / "fakehealth_zenodo" / "extracted"
FAKEHEALTH_ZENODO_MD5 = "1dd710f663694096ba604144ad7e4930"
FAKEHEALTH_ZENODO_SIZE = 2_205_934_463


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    status: str
    details: str


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_git_available() -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required to clone CoAID and FakeHealth.")


def remove_existing(path: Path, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise FileExistsError(
            f"{path} already exists. Re-run with --force to replace it."
        )
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def clone_at_commit(
    repo_url: str, commit: str, target_dir: Path, force: bool
) -> DownloadResult:
    ensure_git_available()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() and not force:
        return DownloadResult(target_dir, "skipped", "already exists")
    remove_existing(target_dir, force=force)
    run(["git", "clone", repo_url, str(target_dir)])
    run(["git", "checkout", commit], cwd=target_dir)
    return DownloadResult(target_dir, "downloaded", f"checked out {commit}")


def fetch_url(url: str, destination: Path, force: bool) -> DownloadResult:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return DownloadResult(destination, "skipped", "already exists")
    remove_existing(destination, force=force)

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    remove_existing(tmp_path, force=True)

    print(f"Downloading {url}")
    print(f"Saving to {destination}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp_path.rename(destination)
    return DownloadResult(destination, "downloaded", "download complete")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_fakehealth_zenodo_zip(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    digest = md5(path)
    errors: list[str] = []
    if size != FAKEHEALTH_ZENODO_SIZE:
        errors.append(f"size {size} != expected {FAKEHEALTH_ZENODO_SIZE}")
    if digest != FAKEHEALTH_ZENODO_MD5:
        errors.append(f"md5 {digest} != expected {FAKEHEALTH_ZENODO_MD5}")
    if errors:
        raise RuntimeError("FakeHealth Zenodo verification failed: " + "; ".join(errors))
    print(f"Verified {path}: size={size}, md5={digest}")


def extract_zip(zip_path: Path, extract_dir: Path, force: bool) -> DownloadResult:
    if extract_dir.exists() and not force:
        return DownloadResult(extract_dir, "skipped", "already extracted")
    remove_existing(extract_dir, force=force)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return DownloadResult(extract_dir, "extracted", "zip extracted")


def write_acquisition_metadata() -> None:
    metadata_path = RAW_DIR / "dataset_sources.json"
    payload = {
        "coaid": {
            "source": COAID_REPO,
            "commit": COAID_COMMIT,
            "local_path": str(COAID_DIR.relative_to(PROJECT_ROOT)),
        },
        "fakehealth_repository": {
            "source": FAKEHEALTH_REPO,
            "commit": FAKEHEALTH_COMMIT,
            "local_path": str(FAKEHEALTH_DIR.relative_to(PROJECT_ROOT)),
        },
        "fakehealth_zenodo_user_network": {
            "record": FAKEHEALTH_ZENODO_RECORD,
            "api_record": FAKEHEALTH_ZENODO_API,
            "file_url": FAKEHEALTH_ZENODO_FILE_URL,
            "expected_size_bytes": FAKEHEALTH_ZENODO_SIZE,
            "expected_md5": FAKEHEALTH_ZENODO_MD5,
            "local_zip": str(FAKEHEALTH_ZENODO_FILE.relative_to(PROJECT_ROOT)),
            "local_extract_dir": str(
                FAKEHEALTH_ZENODO_EXTRACT_DIR.relative_to(PROJECT_ROOT)
            ),
            "contents_note": (
                "Optional FakeHealth user-network follower/following identifier "
                "archive; not hydrated Twitter/X post content."
            ),
        },
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {metadata_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="download all datasets")
    parser.add_argument("--coaid", action="store_true", help="download CoAID repository")
    parser.add_argument(
        "--fakehealth-repo", action="store_true", help="download FakeHealth repository"
    )
    parser.add_argument(
        "--fakehealth-zenodo",
        action="store_true",
        help="download optional FakeHealth Zenodo user-network archive",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="extract the optional FakeHealth Zenodo zip after verification",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing local raw-data files/directories",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    selected = args.all or args.coaid or args.fakehealth_repo or args.fakehealth_zenodo
    if not selected:
        print("No dataset selected. Use --all or one of the dataset flags.", file=sys.stderr)
        return 2

    results: list[DownloadResult] = []
    if args.all or args.coaid:
        results.append(clone_at_commit(COAID_REPO, COAID_COMMIT, COAID_DIR, args.force))
    if args.all or args.fakehealth_repo:
        results.append(
            clone_at_commit(
                FAKEHEALTH_REPO, FAKEHEALTH_COMMIT, FAKEHEALTH_DIR, args.force
            )
        )
    if args.all or args.fakehealth_zenodo:
        results.append(
            fetch_url(FAKEHEALTH_ZENODO_FILE_URL, FAKEHEALTH_ZENODO_FILE, args.force)
        )
        verify_fakehealth_zenodo_zip(FAKEHEALTH_ZENODO_FILE)
        if args.extract:
            results.append(
                extract_zip(
                    FAKEHEALTH_ZENODO_FILE,
                    FAKEHEALTH_ZENODO_EXTRACT_DIR,
                    args.force,
                )
            )

    write_acquisition_metadata()
    print("\nSummary")
    for result in results:
        print(f"- {result.status}: {result.path} ({result.details})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
