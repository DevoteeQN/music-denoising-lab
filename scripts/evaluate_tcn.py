from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.infer_tcn import denoise_full_audio, load_model
from src.audio_utils import ensure_dir, fix_length, list_audio_files, read_audio, spectral_gate_denoise
from src.metrics import snr_db, snr_improvement


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline and lightweight TCN on paired test set.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_noisy", default="data/processed/test/noisy")
    parser.add_argument("--test_clean", default="data/processed/test/clean")
    parser.add_argument("--output_csv", default="outputs/metrics/test_metrics_tcn.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    sample_rate = int(cfg.get("sample_rate", 16000))
    segment_samples = int(round(float(cfg.get("segment_seconds", 4.0)) * sample_rate))

    noisy_files = list_audio_files(args.test_noisy)
    if not noisy_files:
        raise SystemExit(f"No test noisy audio found in {args.test_noisy}")

    rows = []
    for noisy_path in tqdm(noisy_files, desc="Evaluating TCN"):
        clean_path = Path(args.test_clean) / noisy_path.name
        if not clean_path.exists():
            continue
        noisy, _ = read_audio(noisy_path, sample_rate=sample_rate, mono=True)
        clean, _ = read_audio(clean_path, sample_rate=sample_rate, mono=True)
        noisy = fix_length(noisy, segment_samples)
        clean = fix_length(clean, segment_samples)
        baseline = spectral_gate_denoise(
            noisy,
            sample_rate=sample_rate,
            n_fft=int(cfg.get("n_fft", 1024)),
            hop_length=int(cfg.get("hop_length", 256)),
            win_length=int(cfg.get("win_length", 1024)),
        )
        tcn = denoise_full_audio(model, noisy, cfg, device)
        rows.append(
            {
                "file": noisy_path.name,
                "snr_noisy": snr_db(clean, noisy),
                "snr_baseline": snr_db(clean, baseline),
                "snr_tcn": snr_db(clean, tcn),
                "snri_baseline": snr_improvement(clean, noisy, baseline),
                "snri_tcn": snr_improvement(clean, noisy, tcn),
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
                "mean_snr_tcn": float(per_sample["snr_tcn"].mean()),
                "mean_snri_baseline": float(per_sample["snri_baseline"].mean()),
                "mean_snri_tcn": float(per_sample["snri_tcn"].mean()),
            }
        ]
    )

    output_csv = Path(args.output_csv)
    ensure_dir(output_csv.parent)
    summary.to_csv(output_csv, index=False)
    per_sample_path = output_csv.with_name(output_csv.stem + "_per_sample.csv")
    per_sample.to_csv(per_sample_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote TCN summary metrics: {output_csv}")
    print(f"Wrote TCN per-sample metrics: {per_sample_path}")


if __name__ == "__main__":
    main()
