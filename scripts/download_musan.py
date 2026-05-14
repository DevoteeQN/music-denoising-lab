from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


OPENSLR_URL = "https://www.openslr.org/resources/17/musan.tar.gz"


class DownloadProgress(tqdm):
    def update_to(self, block_num=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(block_num * block_size - self.n)


def download_openslr(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "musan.tar.gz"
    if archive.exists():
        print(f"Archive already exists: {archive}")
    else:
        with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc="musan.tar.gz") as progress:
            urllib.request.urlretrieve(OPENSLR_URL, archive, reporthook=progress.update_to)
    return archive


def extract_archive(archive: Path, out_dir: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(out_dir)
    print(f"Extracted to: {out_dir}")


def download_hf_mirror(out_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before using --source hf-mirror.") from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(
        repo_id="huseinzol05/musan-mirror",
        filename="musan.tar.gz",
        repo_type="dataset",
        local_dir=out_dir / "musan_hf",
    )
    return Path(archive)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MUSAN from OpenSLR or a Hugging Face mirror.")
    parser.add_argument("--source", choices=["openslr", "hf-mirror"], default="openslr")
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--no_extract", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.source == "openslr":
        archive = download_openslr(out_dir)
    else:
        archive = download_hf_mirror(out_dir)
    if not args.no_extract:
        extract_archive(archive, out_dir)


if __name__ == "__main__":
    main()
