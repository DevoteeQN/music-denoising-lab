from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_utils import peak_normalize, read_audio, write_audio_with_mp3
from src.model_tcn import LightweightTCNDenoiser


def load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    model = LightweightTCNDenoiser(
        channels=int(cfg.get("tcn_channels", 64)),
        kernel_size=int(cfg.get("tcn_kernel_size", 3)),
        blocks=int(cfg.get("tcn_blocks", 8)),
        encoder_kernel=int(cfg.get("tcn_encoder_kernel", 16)),
        encoder_stride=int(cfg.get("tcn_encoder_stride", 8)),
        mask_scale=float(cfg.get("mask_scale", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def denoise_full_audio(model, audio: np.ndarray, cfg: dict, device: torch.device) -> np.ndarray:
    segment_samples = int(round(float(cfg.get("segment_seconds", 4.0)) * int(cfg.get("sample_rate", 16000))))
    outputs = []
    for start in range(0, len(audio), segment_samples):
        chunk = audio[start : start + segment_samples]
        original_len = len(chunk)
        if original_len < segment_samples:
            chunk = np.pad(chunk, (0, segment_samples - original_len))
        x = torch.tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            denoised = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
        outputs.append(denoised[:original_len])
    return peak_normalize(np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight TCN denoising on an audio file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    sample_rate = int(cfg.get("sample_rate", 16000))
    audio, sr = read_audio(args.input, sample_rate=sample_rate, mono=True)
    denoised = denoise_full_audio(model, audio, cfg, device)
    write_audio_with_mp3(args.output, denoised, sr)
    print(f"Wrote TCN output: {args.output}")
    print(f"Wrote TCN MP3: {Path(args.output).with_suffix('.mp3')}")


if __name__ == "__main__":
    main()
