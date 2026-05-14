from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_utils import peak_normalize, read_audio, write_audio_with_mp3
from src.model_dit import TinyDiTSpectrogramDenoiser


def denoise_segment(model, segment: np.ndarray, cfg: dict, device: torch.device) -> np.ndarray:
    length = len(segment)
    x = torch.tensor(segment, dtype=torch.float32, device=device).unsqueeze(0)
    window = torch.hann_window(int(cfg["win_length"]), device=device)
    spec = torch.stft(x, n_fft=int(cfg["n_fft"]), hop_length=int(cfg["hop_length"]), win_length=int(cfg["win_length"]), window=window, return_complex=True)
    mag = spec.abs()
    phase = spec / (mag + 1e-8)
    with torch.no_grad():
        mask = model(torch.log1p(mag).unsqueeze(1)).squeeze(1)
    wav = torch.istft(mask * mag * phase, n_fft=int(cfg["n_fft"]), hop_length=int(cfg["hop_length"]), win_length=int(cfg["win_length"]), window=window, length=length)
    return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)


def denoise_full_audio(model, audio: np.ndarray, cfg: dict, device: torch.device) -> np.ndarray:
    segment_samples = int(round(float(cfg["segment_seconds"]) * int(cfg["sample_rate"])))
    outputs = []
    for start in range(0, len(audio), segment_samples):
        chunk = audio[start : start + segment_samples]
        original_len = len(chunk)
        if original_len < segment_samples:
            chunk = np.pad(chunk, (0, segment_samples - original_len))
        outputs.append(denoise_segment(model, chunk, cfg, device)[:original_len])
    return peak_normalize(np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32))


def load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    model = TinyDiTSpectrogramDenoiser(
        dim=int(cfg.get("dit_dim", 96)),
        depth=int(cfg.get("dit_depth", 4)),
        heads=int(cfg.get("dit_heads", 4)),
        patch_size=int(cfg.get("dit_patch_size", 32)),
        mask_scale=float(cfg.get("mask_scale", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny DiT-style spectrogram denoising on an audio file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model, cfg = load_model(args.checkpoint, device)
    sample_rate = int(cfg.get("sample_rate", 16000))
    audio, sr = read_audio(args.input, sample_rate=sample_rate, mono=True)
    denoised = denoise_full_audio(model, audio, cfg, device)
    write_audio_with_mp3(args.output, denoised, sr)
    print(f"Wrote DiT-style output: {args.output}")
    print(f"Wrote DiT-style MP3: {Path(args.output).with_suffix('.mp3')}")


if __name__ == "__main__":
    main()
