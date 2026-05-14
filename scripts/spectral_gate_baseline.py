from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_utils import read_audio, spectral_gate_denoise, write_audio_with_mp3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traditional spectral-gate denoising baseline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--noise_reduce_strength", type=float, default=0.8)
    parser.add_argument("--prop_decrease", type=float, default=0.8)
    args = parser.parse_args()

    audio, sr = read_audio(args.input, sample_rate=args.sample_rate, mono=True)
    denoised = spectral_gate_denoise(
        audio,
        sample_rate=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        noise_reduce_strength=args.noise_reduce_strength,
        prop_decrease=args.prop_decrease,
    )
    write_audio_with_mp3(args.output, denoised, sr)
    print(f"Wrote baseline output: {args.output}")
    print(f"Wrote baseline MP3: {Path(args.output).with_suffix('.mp3')}")


if __name__ == "__main__":
    main()
