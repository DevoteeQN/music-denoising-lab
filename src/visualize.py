from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.audio_utils import ensure_dir


def _time_axis(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return np.arange(len(audio), dtype=np.float32) / float(sample_rate)


def plot_waveform(audio: np.ndarray, sample_rate: int, title: str, out_path: str | Path) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(10, 3), dpi=160)
    ax.plot(_time_axis(audio, sample_rate), audio, linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _spectrogram_db(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> np.ndarray:
    x = torch.tensor(audio, dtype=torch.float32)
    window = torch.hann_window(win_length)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    mag = spec.abs().numpy()
    db = 20.0 * np.log10(np.maximum(mag, 1e-5))
    top = float(np.max(db))
    return np.clip(db, top - 80.0, top)


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    title: str,
    out_path: str | Path,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    db = _spectrogram_db(audio, sample_rate, n_fft, hop_length, win_length)
    duration = len(audio) / float(sample_rate)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    im = ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=[0.0, duration, 0.0, sample_rate / 2000.0],
        cmap="magma",
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    fig.colorbar(im, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_waveform_comparison(
    raw: np.ndarray,
    baseline: np.ndarray,
    unet: np.ndarray,
    sample_rate: int,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    items = [("Raw noisy", raw), ("Spectral gate baseline", baseline), ("U-Net denoised", unet)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), dpi=160, sharex=True)
    for ax, (title, audio) in zip(axes, items):
        ax.plot(_time_axis(audio, sample_rate), audio, linewidth=0.6)
        ax.set_title(title)
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_spectrogram_comparison(
    raw: np.ndarray,
    baseline: np.ndarray,
    unet: np.ndarray,
    sample_rate: int,
    out_path: str | Path,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    items = [("Raw noisy", raw), ("Spectral gate baseline", baseline), ("U-Net denoised", unet)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=160, sharex=True)
    duration = max(len(raw), len(baseline), len(unet)) / float(sample_rate)
    for ax, (title, audio) in zip(axes, items):
        db = _spectrogram_db(audio, sample_rate, n_fft, hop_length, win_length)
        im = ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=[0.0, len(audio) / sample_rate, 0.0, sample_rate / 2000.0],
            cmap="magma",
        )
        ax.set_xlim(0.0, duration)
        ax.set_title(title)
        ax.set_ylabel("Frequency (kHz)")
    axes[-1].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Magnitude (dB)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_multi_waveform_comparison(
    items: list[tuple[str, np.ndarray]],
    sample_rate: int,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(len(items), 1, figsize=(11, 2.0 * len(items)), dpi=160, sharex=True)
    if len(items) == 1:
        axes = [axes]
    for ax, (title, audio) in zip(axes, items):
        ax.plot(_time_axis(audio, sample_rate), audio, linewidth=0.6)
        ax.set_title(title)
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_multi_spectrogram_comparison(
    items: list[tuple[str, np.ndarray]],
    sample_rate: int,
    out_path: str | Path,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(len(items), 1, figsize=(11, 2.7 * len(items)), dpi=160, sharex=True)
    if len(items) == 1:
        axes = [axes]
    duration = max(len(audio) for _, audio in items) / float(sample_rate)
    im = None
    for ax, (title, audio) in zip(axes, items):
        db = _spectrogram_db(audio, sample_rate, n_fft, hop_length, win_length)
        im = ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=[0.0, len(audio) / sample_rate, 0.0, sample_rate / 2000.0],
            cmap="magma",
        )
        ax.set_xlim(0.0, duration)
        ax.set_title(title)
        ax.set_ylabel("Frequency (kHz)")
    axes[-1].set_xlabel("Time (s)")
    if im is not None:
        fig.colorbar(im, ax=axes, label="Magnitude (dB)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(log_df, out_path: str | Path) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.plot(log_df["epoch"], log_df["train_loss"], marker="o", label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], marker="o", label="val")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
