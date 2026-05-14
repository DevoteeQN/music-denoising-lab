from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_utils import (
    ensure_dir,
    list_audio_files,
    load_random_segment,
    mix_at_snr,
    parse_snr_list,
    write_audio,
)


def collect_musan_files(musan_dir: Path) -> tuple[list[Path], list[Path]]:
    music_dir = musan_dir / "music"
    noise_dir = musan_dir / "noise"
    music_files = list_audio_files(music_dir)
    noise_files = list_audio_files(noise_dir)
    return music_files, noise_files


def synthesize_split(
    split: str,
    count: int,
    music_files: list[Path],
    noise_files: list[Path],
    out_dir: Path,
    sample_rate: int,
    segment_seconds: float,
    snr_list: list[float],
    rng: np.random.Generator,
) -> list[dict[str, str | int | float]]:
    split_noisy = ensure_dir(out_dir / split / "noisy")
    split_clean = ensure_dir(out_dir / split / "clean")
    segment_samples = int(round(sample_rate * segment_seconds))
    rows: list[dict[str, str | int | float]] = []

    for idx in tqdm(range(count), desc=f"Preparing {split}"):
        clean_path = music_files[int(rng.integers(0, len(music_files)))]
        noise_path = noise_files[int(rng.integers(0, len(noise_files)))]
        snr_db = float(snr_list[int(rng.integers(0, len(snr_list)))])
        clean = load_random_segment(clean_path, sample_rate, segment_samples, rng)
        noise = load_random_segment(noise_path, sample_rate, segment_samples, rng)
        noisy, clean = mix_at_snr(clean, noise, snr_db=snr_db)

        file_name = f"{split}_{idx:05d}.wav"
        noisy_out = split_noisy / file_name
        clean_out = split_clean / file_name
        write_audio(noisy_out, noisy, sample_rate)
        write_audio(clean_out, clean, sample_rate)
        rows.append(
            {
                "split": split,
                "index": idx,
                "noisy_path": str(noisy_out),
                "clean_path": str(clean_out),
                "source_music": str(clean_path),
                "source_noise": str(noise_path),
                "snr_db": snr_db,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paired noisy/clean data from MUSAN music and noise.")
    parser.add_argument("--musan_dir", default="data/raw/musan")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=100)
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--segment_seconds", type=float, default=4.0)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--snr_list", default="-5,0,5,10")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    musan_dir = Path(args.musan_dir)
    out_dir = Path(args.out_dir)
    music_files, noise_files = collect_musan_files(musan_dir)
    if not music_files or not noise_files:
        raise SystemExit(
            f"No MUSAN music/noise audio found under {musan_dir}. "
            "Download MUSAN or run `python scripts/make_demo_assets.py` for a smoke test."
        )

    ensure_dir(out_dir)
    rng = np.random.default_rng(args.seed)
    snr_list = parse_snr_list(args.snr_list)
    all_rows: list[dict[str, str | int | float]] = []
    for split, count in [
        ("train", args.train_samples),
        ("val", args.val_samples),
        ("test", args.test_samples),
    ]:
        all_rows.extend(
            synthesize_split(
                split=split,
                count=count,
                music_files=music_files,
                noise_files=noise_files,
                out_dir=out_dir,
                sample_rate=args.sample_rate,
                segment_seconds=args.segment_seconds,
                snr_list=snr_list,
                rng=rng,
            )
        )

    metadata_path = out_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "index",
                "noisy_path",
                "clean_path",
                "source_music",
                "source_noise",
                "snr_db",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
