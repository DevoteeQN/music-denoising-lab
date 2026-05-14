from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_demucs_pretrained import combine_stems, run_demucs
from src.audio_utils import ensure_dir, fix_length, list_audio_files, read_audio, spectral_gate_denoise
from src.metrics import snr_db, snr_improvement


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pretrained Demucs zero-shot music reconstruction.")
    parser.add_argument("--test_noisy", default="data/processed/test/noisy")
    parser.add_argument("--test_clean", default="data/processed/test/clean")
    parser.add_argument("--output_csv", default="outputs/metrics/test_metrics_demucs.csv")
    parser.add_argument("--work_dir", default="outputs/demucs_eval")
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segment", type=float, default=4.0)
    parser.add_argument("--batch_tracks", type=int, default=20)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_seconds", type=float, default=4.0)
    args = parser.parse_args()

    noisy_files = list_audio_files(args.test_noisy)
    if not noisy_files:
        raise SystemExit(f"No noisy files found in {args.test_noisy}")
    work_dir = Path(args.work_dir)
    ensure_dir(work_dir)
    enhanced_dir = ensure_dir(work_dir / "combined")

    rows = []
    for start in range(0, len(noisy_files), args.batch_tracks):
        batch = noisy_files[start : start + args.batch_tracks]
        run_demucs(batch, work_dir, model=args.model, device=args.device, segment=args.segment)
        for noisy_path in tqdm(batch, desc=f"Scoring Demucs {start}-{start + len(batch) - 1}"):
            clean_path = Path(args.test_clean) / noisy_path.name
            if not clean_path.exists():
                continue
            combined_path = enhanced_dir / noisy_path.name
            track_dir = work_dir / args.model / noisy_path.stem
            combine_stems(track_dir, combined_path, sample_rate=args.sample_rate)
            noisy, _ = read_audio(noisy_path, sample_rate=args.sample_rate, mono=True)
            clean, _ = read_audio(clean_path, sample_rate=args.sample_rate, mono=True)
            demucs, _ = read_audio(combined_path, sample_rate=args.sample_rate, mono=True)
            segment_samples = int(round(args.sample_rate * args.segment_seconds))
            noisy = fix_length(noisy, segment_samples)
            clean = fix_length(clean, segment_samples)
            demucs = fix_length(demucs, segment_samples)
            baseline = spectral_gate_denoise(noisy, sample_rate=args.sample_rate)
            rows.append(
                {
                    "file": noisy_path.name,
                    "snr_noisy": snr_db(clean, noisy),
                    "snr_baseline": snr_db(clean, baseline),
                    "snr_demucs": snr_db(clean, demucs),
                    "snri_baseline": snr_improvement(clean, noisy, baseline),
                    "snri_demucs": snr_improvement(clean, noisy, demucs),
                }
            )

    if not rows:
        raise SystemExit("No matched test pairs were found.")
    per_sample = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "num_samples": len(per_sample),
                "mean_snr_noisy": float(per_sample["snr_noisy"].mean()),
                "mean_snr_baseline": float(per_sample["snr_baseline"].mean()),
                "mean_snr_demucs": float(per_sample["snr_demucs"].mean()),
                "mean_snri_baseline": float(per_sample["snri_baseline"].mean()),
                "mean_snri_demucs": float(per_sample["snri_demucs"].mean()),
            }
        ]
    )
    output_csv = Path(args.output_csv)
    ensure_dir(output_csv.parent)
    summary.to_csv(output_csv, index=False)
    per_sample.to_csv(output_csv.with_name(output_csv.stem + "_per_sample.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"Wrote Demucs metrics: {output_csv}")


if __name__ == "__main__":
    main()
