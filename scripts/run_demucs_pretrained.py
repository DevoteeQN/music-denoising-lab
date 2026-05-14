from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_utils import ensure_dir, peak_normalize, read_audio, write_audio_with_mp3


STEMS = ["bass", "drums", "other", "vocals"]


def run_demucs(
    inputs: list[Path],
    work_dir: Path,
    model: str = "htdemucs",
    device: str = "cuda",
    segment: float = 4.0,
) -> None:
    ensure_dir(work_dir)
    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str(ROOT / "data" / "raw" / ".torch_home"))
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model,
        "-d",
        device,
        "--shifts",
        "0",
        "--segment",
        str(int(round(segment))),
        "-o",
        str(work_dir),
        *[str(path) for path in inputs],
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def combine_stems(track_dir: Path, output: Path, sample_rate: int = 16000) -> None:
    signals = []
    for stem in STEMS:
        stem_path = track_dir / f"{stem}.wav"
        if stem_path.exists():
            audio, _ = read_audio(stem_path, sample_rate=sample_rate, mono=True)
            signals.append(audio)
    if not signals:
        raise FileNotFoundError(f"No Demucs stems found in {track_dir}")
    length = min(len(sig) for sig in signals)
    combined = np.sum([sig[:length] for sig in signals], axis=0)
    write_audio_with_mp3(output, peak_normalize(combined), sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pretrained Demucs and sum music stems as zero-shot denoising.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work_dir", default="outputs/demucs_work")
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segment", type=float, default=4.0)
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    input_path = Path(args.input)
    work_dir = Path(args.work_dir)
    run_demucs([input_path], work_dir, model=args.model, device=args.device, segment=args.segment)
    track_dir = work_dir / args.model / input_path.stem
    combine_stems(track_dir, Path(args.output), sample_rate=args.sample_rate)
    print(f"Wrote Demucs output: {args.output}")
    print(f"Wrote Demucs MP3: {Path(args.output).with_suffix('.mp3')}")


if __name__ == "__main__":
    main()
