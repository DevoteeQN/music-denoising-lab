from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_utils import ensure_dir, mix_at_snr, peak_normalize, write_audio


def synth_music(duration: float, sample_rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    chord_sets = [
        [220.0, 277.18, 329.63],
        [196.0, 246.94, 293.66],
        [261.63, 329.63, 392.0],
        [174.61, 220.0, 261.63],
    ]
    audio = np.zeros_like(t)
    bar_len = max(1, len(t) // len(chord_sets))
    for i, chord in enumerate(chord_sets):
        start = i * bar_len
        end = len(t) if i == len(chord_sets) - 1 else (i + 1) * bar_len
        tt = t[start:end]
        env = 0.55 + 0.45 * np.sin(2 * np.pi * 0.75 * tt + rng.uniform(0, np.pi)) ** 2
        part = np.zeros_like(tt)
        for freq in chord:
            vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * 5.0 * tt)
            part += np.sin(2 * np.pi * freq * vibrato * tt)
            part += 0.35 * np.sin(2 * np.pi * freq * 2.0 * tt)
        audio[start:end] = part * env
    audio += 0.08 * np.sin(2 * np.pi * 110.0 * t)
    audio *= 0.2
    return peak_normalize(audio.astype(np.float32), peak=0.75)


def synth_noise(duration: float, sample_rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    white = rng.normal(0.0, 1.0, n).astype(np.float32)
    kernel = np.ones(128, dtype=np.float32) / 128.0
    low = np.convolve(white, kernel, mode="same")
    hum_t = np.arange(n, dtype=np.float32) / sample_rate
    hum = 0.4 * np.sin(2 * np.pi * 60.0 * hum_t) + 0.2 * np.sin(2 * np.pi * 120.0 * hum_t)
    clicks = np.zeros(n, dtype=np.float32)
    click_positions = rng.integers(0, n, size=max(1, int(duration * 3)))
    clicks[click_positions] = rng.uniform(-1.0, 1.0, size=len(click_positions))
    clicks = np.convolve(clicks, np.exp(-np.arange(200) / 30.0).astype(np.float32), mode="same")
    noise = 0.45 * white + 0.7 * low + 0.25 * hum + 0.25 * clicks
    return peak_normalize(noise.astype(np.float32), peak=0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create small synthetic assets for smoke tests.")
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--num_files", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    music_dir = out_dir / "musan" / "music" / "demo"
    noise_dir = out_dir / "musan" / "noise" / "demo"
    ensure_dir(music_dir)
    ensure_dir(noise_dir)

    for idx in range(args.num_files):
        music_path = music_dir / f"demo_music_{idx:03d}.wav"
        noise_path = noise_dir / f"demo_noise_{idx:03d}.wav"
        if args.force or not music_path.exists():
            write_audio(music_path, synth_music(args.duration, args.sample_rate, args.seed + idx), args.sample_rate)
        if args.force or not noise_path.exists():
            write_audio(noise_path, synth_noise(args.duration, args.sample_rate, args.seed + 1000 + idx), args.sample_rate)

    raw_path = out_dir / "raw.MP3"
    if args.force or not raw_path.exists():
        clean = synth_music(args.duration, args.sample_rate, args.seed + 999)
        noise = synth_noise(args.duration, args.sample_rate, args.seed + 1999)
        noisy, _ = mix_at_snr(clean, noise, snr_db=0.0)
        try:
            write_audio(raw_path, noisy, args.sample_rate)
        except RuntimeError as exc:
            fallback = raw_path.with_suffix(".wav")
            write_audio(fallback, noisy, args.sample_rate)
            print(f"MP3 export unavailable: {exc}")
            print(f"Wrote fallback raw audio: {fallback}")

    print(f"Demo assets written under: {out_dir}")
    print("Use these only for smoke tests; replace them with course raw.MP3 and MUSAN for the report.")


if __name__ == "__main__":
    main()
