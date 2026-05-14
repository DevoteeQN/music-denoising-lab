from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_utils import read_audio
from src.visualize import (
    plot_multi_spectrogram_comparison,
    plot_multi_waveform_comparison,
    plot_spectrogram,
    plot_spectrogram_comparison,
    plot_waveform,
    plot_waveform_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create waveform and spectrogram figures for report.")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--unet", required=True)
    parser.add_argument("--tcn")
    parser.add_argument("--cnn")
    parser.add_argument("--dit")
    parser.add_argument("--demucs")
    parser.add_argument("--denoiser")
    parser.add_argument("--out_dir", default="outputs/figures")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=1024)
    args = parser.parse_args()

    raw, sr = read_audio(args.raw, sample_rate=args.sample_rate, mono=True)
    baseline, _ = read_audio(args.baseline, sample_rate=sr, mono=True)
    unet, _ = read_audio(args.unet, sample_rate=sr, mono=True)
    tcn = None
    if args.tcn:
        tcn, _ = read_audio(args.tcn, sample_rate=sr, mono=True)
    cnn = None
    if args.cnn:
        cnn, _ = read_audio(args.cnn, sample_rate=sr, mono=True)
    dit = None
    if args.dit:
        dit, _ = read_audio(args.dit, sample_rate=sr, mono=True)
    demucs = None
    if args.demucs:
        demucs, _ = read_audio(args.demucs, sample_rate=sr, mono=True)
    denoiser = None
    if args.denoiser:
        denoiser, _ = read_audio(args.denoiser, sample_rate=sr, mono=True)

    out_dir = Path(args.out_dir)
    plot_waveform(raw, sr, "Raw noisy waveform", out_dir / "raw_waveform.png")
    plot_waveform(baseline, sr, "Spectral gate baseline waveform", out_dir / "baseline_waveform.png")
    plot_waveform(unet, sr, "U-Net denoised waveform", out_dir / "unet_waveform.png")
    plot_waveform_comparison(raw, baseline, unet, sr, out_dir / "waveform_comparison.png")

    plot_spectrogram(
        raw,
        sr,
        "Raw noisy spectrogram",
        out_dir / "raw_spectrogram.png",
        args.n_fft,
        args.hop_length,
        args.win_length,
    )
    plot_spectrogram(
        baseline,
        sr,
        "Spectral gate baseline spectrogram",
        out_dir / "baseline_spectrogram.png",
        args.n_fft,
        args.hop_length,
        args.win_length,
    )
    plot_spectrogram(
        unet,
        sr,
        "U-Net denoised spectrogram",
        out_dir / "unet_spectrogram.png",
        args.n_fft,
        args.hop_length,
        args.win_length,
    )
    plot_spectrogram_comparison(
        raw,
        baseline,
        unet,
        sr,
        out_dir / "spectrogram_comparison.png",
        args.n_fft,
        args.hop_length,
        args.win_length,
    )
    if tcn is not None:
        plot_waveform(tcn, sr, "Lightweight TCN denoised waveform", out_dir / "tcn_waveform.png")
        plot_spectrogram(
            tcn,
            sr,
            "Lightweight TCN denoised spectrogram",
            out_dir / "tcn_spectrogram.png",
            args.n_fft,
            args.hop_length,
            args.win_length,
        )
        items = [
            ("Raw noisy", raw),
            ("Spectral gate baseline", baseline),
            ("U-Net denoised", unet),
            ("Lightweight TCN denoised", tcn),
        ]
        plot_multi_waveform_comparison(items, sr, out_dir / "waveform_comparison_4way.png")
        plot_multi_spectrogram_comparison(
            items,
            sr,
            out_dir / "spectrogram_comparison_4way.png",
            args.n_fft,
            args.hop_length,
            args.win_length,
        )
    extra_items = [
        ("Raw noisy", raw),
        ("Spectral gate baseline", baseline),
        ("U-Net denoised", unet),
    ]
    if tcn is not None:
        extra_items.append(("Lightweight TCN denoised", tcn))
    if cnn is not None:
        plot_waveform(cnn, sr, "Residual CNN denoised waveform", out_dir / "cnn_waveform.png")
        plot_spectrogram(
            cnn,
            sr,
            "Residual CNN denoised spectrogram",
            out_dir / "cnn_spectrogram.png",
            args.n_fft,
            args.hop_length,
            args.win_length,
        )
        extra_items.append(("Residual CNN denoised", cnn))
    if dit is not None:
        plot_waveform(dit, sr, "Tiny DiT-style denoised waveform", out_dir / "dit_waveform.png")
        plot_spectrogram(
            dit,
            sr,
            "Tiny DiT-style denoised spectrogram",
            out_dir / "dit_spectrogram.png",
            args.n_fft,
            args.hop_length,
            args.win_length,
        )
        extra_items.append(("Tiny DiT-style denoised", dit))
    if demucs is not None:
        plot_waveform(demucs, sr, "Pretrained Demucs zero-shot waveform", out_dir / "demucs_waveform.png")
        plot_spectrogram(
            demucs,
            sr,
            "Pretrained Demucs zero-shot spectrogram",
            out_dir / "demucs_spectrogram.png",
            args.n_fft,
            args.hop_length,
            args.win_length,
        )
        extra_items.append(("Pretrained Demucs", demucs))
    if denoiser is not None:
        plot_waveform(denoiser, sr, "Pretrained Denoiser waveform", out_dir / "denoiser_waveform.png")
        plot_spectrogram(
            denoiser,
            sr,
            "Pretrained Denoiser spectrogram",
            out_dir / "denoiser_spectrogram.png",
            args.n_fft,
            args.hop_length,
            args.win_length,
        )
        extra_items.append(("Pretrained Denoiser", denoiser))
    if len(extra_items) > 4:
        plot_multi_waveform_comparison(extra_items, sr, out_dir / "waveform_comparison_all.png")
        plot_multi_spectrogram_comparison(
            extra_items,
            sr,
            out_dir / "spectrogram_comparison_all.png",
            args.n_fft,
            args.hop_length,
            args.win_length,
        )
    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
