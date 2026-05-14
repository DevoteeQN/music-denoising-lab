#!/usr/bin/env bash
set -e

python scripts/prepare_musan_dataset.py \
  --musan_dir data/raw/musan \
  --out_dir data/processed \
  --train_samples 1000 \
  --val_samples 100 \
  --test_samples 100 \
  --segment_seconds 4 \
  --sample_rate 16000

python scripts/spectral_gate_baseline.py \
  --input data/raw/raw.MP3 \
  --output outputs/baseline/raw_denoised_spectral_gate.wav

python scripts/train_unet.py \
  --config configs/default.yaml \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-3

python scripts/infer_unet.py \
  --checkpoint outputs/checkpoints/best_unet.pt \
  --input data/raw/raw.MP3 \
  --output outputs/unet/raw_denoised_unet.wav

python scripts/train_tcn.py \
  --config configs/default.yaml \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-3

python scripts/infer_tcn.py \
  --checkpoint outputs/checkpoints/best_tcn.pt \
  --input data/raw/raw.MP3 \
  --output outputs/tcn/raw_denoised_tcn.wav

python scripts/train_cnn.py \
  --config configs/default.yaml \
  --epochs 10 \
  --batch_size 8

python scripts/infer_cnn.py \
  --checkpoint outputs/checkpoints/best_cnn.pt \
  --input data/raw/raw.MP3 \
  --output outputs/cnn/raw_denoised_cnn.wav

python scripts/train_dit.py \
  --config configs/default.yaml \
  --epochs 30 \
  --batch_size 8

python scripts/infer_dit.py \
  --checkpoint outputs/checkpoints/best_dit.pt \
  --input data/raw/raw.MP3 \
  --output outputs/dit/raw_denoised_dit.wav

python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/best_unet.pt \
  --test_noisy data/processed/test/noisy \
  --test_clean data/processed/test/clean \
  --output_csv outputs/metrics/test_metrics.csv

python scripts/evaluate_tcn.py \
  --checkpoint outputs/checkpoints/best_tcn.pt \
  --test_noisy data/processed/test/noisy \
  --test_clean data/processed/test/clean \
  --output_csv outputs/metrics/test_metrics_tcn.csv

python scripts/evaluate_cnn.py \
  --checkpoint outputs/checkpoints/best_cnn.pt \
  --test_noisy data/processed/test/noisy \
  --test_clean data/processed/test/clean \
  --output_csv outputs/metrics/test_metrics_cnn.csv

python scripts/evaluate_dit.py \
  --checkpoint outputs/checkpoints/best_dit.pt \
  --test_noisy data/processed/test/noisy \
  --test_clean data/processed/test/clean \
  --output_csv outputs/metrics/test_metrics_dit.csv

python scripts/plot_audio.py \
  --raw data/raw/raw.MP3 \
  --baseline outputs/baseline/raw_denoised_spectral_gate.wav \
  --unet outputs/unet/raw_denoised_unet.wav \
  --tcn outputs/tcn/raw_denoised_tcn.wav \
  --cnn outputs/cnn/raw_denoised_cnn.wav \
  --dit outputs/dit/raw_denoised_dit.wav \
  --out_dir outputs/figures
