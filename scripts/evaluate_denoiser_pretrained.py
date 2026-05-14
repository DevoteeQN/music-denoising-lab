from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_denoiser_pretrained import enhance_audio, load_pretrained
from src.audio_utils import ensure_dir, fix_length, list_audio_files, read_audio, spectral_gate_denoise
from src.metrics import snr_db, snr_improvement


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pretrained Facebook Denoiser on synthetic test set.")
    parser.add_argument("--test_noisy", default="data/processed/test/noisy")
    parser.add_argument("--test_clean", default="data/processed/test/clean")
    parser.add_argument("--output_csv", default="outputs/metrics/test_metrics_denoiser.csv")
    parser.add_argument("--model", choices=["dns64", "dns48", "master64", "valentini_nc"], default="dns64")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_seconds", type=float, default=4.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = load_pretrained(args.model, device)
    segment_samples = int(round(args.segment_seconds * args.sample_rate))
    rows = []
    for noisy_path in tqdm(list_audio_files(args.test_noisy), desc="Evaluating pretrained Denoiser"):
        clean_path = Path(args.test_clean) / noisy_path.name
        if not clean_path.exists():
            continue
        noisy, _ = read_audio(noisy_path, sample_rate=args.sample_rate, mono=True)
        clean, _ = read_audio(clean_path, sample_rate=args.sample_rate, mono=True)
        noisy = fix_length(noisy, segment_samples)
        clean = fix_length(clean, segment_samples)
        baseline = spectral_gate_denoise(noisy, sample_rate=args.sample_rate)
        enhanced = enhance_audio(model, noisy, device, args.segment_seconds, args.sample_rate)
        rows.append(
            {
                "file": noisy_path.name,
                "snr_noisy": snr_db(clean, noisy),
                "snr_baseline": snr_db(clean, baseline),
                "snr_denoiser": snr_db(clean, enhanced),
                "snri_baseline": snr_improvement(clean, noisy, baseline),
                "snri_denoiser": snr_improvement(clean, noisy, enhanced),
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
                "mean_snr_denoiser": float(per_sample["snr_denoiser"].mean()),
                "mean_snri_baseline": float(per_sample["snri_baseline"].mean()),
                "mean_snri_denoiser": float(per_sample["snri_denoiser"].mean()),
            }
        ]
    )
    output_csv = Path(args.output_csv)
    ensure_dir(output_csv.parent)
    summary.to_csv(output_csv, index=False)
    per_sample.to_csv(output_csv.with_name(output_csv.stem + "_per_sample.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"Wrote Denoiser metrics: {output_csv}")


if __name__ == "__main__":
    main()
