from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_utils import peak_normalize, read_audio, write_audio_with_mp3


def load_pretrained(name: str, device: torch.device):
    import denoiser.pretrained as pretrained

    choices = {
        "dns64": pretrained.dns64,
        "dns48": pretrained.dns48,
        "master64": pretrained.master64,
        "valentini_nc": pretrained.valentini_nc,
    }
    if name not in choices:
        raise ValueError(f"Unknown denoiser model {name}. Choose from {sorted(choices)}")
    model = choices[name]().to(device)
    model.eval()
    return model


def enhance_audio(model, audio: np.ndarray, device: torch.device, chunk_seconds: float, sample_rate: int) -> np.ndarray:
    chunk_samples = int(round(chunk_seconds * sample_rate))
    outputs = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        original_len = len(chunk)
        if original_len < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - original_len))
        x = torch.tensor(chunk, dtype=torch.float32, device=device).view(1, 1, -1)
        with torch.no_grad():
            y = model(x).view(-1).detach().cpu().numpy().astype(np.float32)
        outputs.append(y[:original_len])
    return peak_normalize(np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Facebook Denoiser pretrained speech enhancement model.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["dns64", "dns48", "master64", "valentini_nc"], default="dns64")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--chunk_seconds", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = load_pretrained(args.model, device)
    audio, sr = read_audio(args.input, sample_rate=args.sample_rate, mono=True)
    enhanced = enhance_audio(model, audio, device, args.chunk_seconds, sr)
    write_audio_with_mp3(args.output, enhanced, sr)
    print(f"Wrote Denoiser output: {args.output}")
    print(f"Wrote Denoiser MP3: {Path(args.output).with_suffix('.mp3')}")


if __name__ == "__main__":
    main()
