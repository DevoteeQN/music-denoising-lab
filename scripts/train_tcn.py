from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_utils import ensure_dir
from src.dataset import PairedAudioDataset
from src.model_tcn import LightweightTCNDenoiser
from src.visualize import plot_loss_curve


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stft_mag(audio: torch.Tensor, cfg: dict, window: torch.Tensor) -> torch.Tensor:
    spec = torch.stft(
        audio,
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
        win_length=int(cfg["win_length"]),
        window=window,
        return_complex=True,
    )
    return spec.abs()


def run_epoch(model, loader, optimizer, cfg, device, train: bool) -> float:
    model.train(train)
    losses: list[float] = []
    window = torch.hann_window(int(cfg["win_length"]), device=device)
    desc = "train_tcn" if train else "val_tcn"
    for noisy, clean, _ in tqdm(loader, desc=desc, leave=False):
        noisy = noisy.to(device)
        clean = clean.to(device)
        denoised = model(noisy)
        wav_loss = F.l1_loss(denoised, clean)
        denoised_mag = stft_mag(denoised, cfg, window)
        clean_mag = stft_mag(clean, cfg, window)
        mag_loss = F.l1_loss(torch.log1p(denoised_mag), torch.log1p(clean_mag))
        loss = wav_loss + 0.2 * mag_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight Conv-TasNet-style TCN denoiser.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train_noisy", default="data/processed/train/noisy")
    parser.add_argument("--train_clean", default="data/processed/train/clean")
    parser.add_argument("--val_noisy", default="data/processed/val/noisy")
    parser.add_argument("--val_clean", default="data/processed/val/clean")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--save_dir", default="outputs/checkpoints")
    parser.add_argument("--metrics_dir", default="outputs/metrics")
    parser.add_argument("--figures_dir", default="outputs/figures")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key in ["epochs", "batch_size", "lr", "num_workers"]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value

    set_seed(int(cfg.get("seed", 42)))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    train_ds = PairedAudioDataset(
        args.train_noisy,
        args.train_clean,
        sample_rate=int(cfg["sample_rate"]),
        segment_seconds=float(cfg["segment_seconds"]),
    )
    val_ds = PairedAudioDataset(
        args.val_noisy,
        args.val_clean,
        sample_rate=int(cfg["sample_rate"]),
        segment_seconds=float(cfg["segment_seconds"]),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        drop_last=False,
    )

    model = LightweightTCNDenoiser(
        channels=int(cfg.get("tcn_channels", 64)),
        kernel_size=int(cfg.get("tcn_kernel_size", 3)),
        blocks=int(cfg.get("tcn_blocks", 8)),
        encoder_kernel=int(cfg.get("tcn_encoder_kernel", 16)),
        encoder_stride=int(cfg.get("tcn_encoder_stride", 8)),
        mask_scale=float(cfg.get("mask_scale", 1.0)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    save_dir = ensure_dir(args.save_dir)
    metrics_dir = ensure_dir(args.metrics_dir)
    figures_dir = ensure_dir(args.figures_dir)
    best_loss = float("inf")
    rows = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, optimizer, cfg, device, train=False)
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, save_dir / "last_tcn.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, save_dir / "best_tcn.pt")

    log_df = pd.DataFrame(rows)
    log_path = metrics_dir / "train_tcn_log.csv"
    log_df.to_csv(log_path, index=False)
    plot_loss_curve(log_df, figures_dir / "tcn_loss_curve.png")
    print(f"Wrote TCN train log: {log_path}")
    print(f"Wrote best TCN checkpoint: {save_dir / 'best_tcn.pt'}")


if __name__ == "__main__":
    main()
