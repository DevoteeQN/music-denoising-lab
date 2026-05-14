from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.audio_utils import fix_length, list_audio_files, read_audio


class PairedAudioDataset(Dataset):
    def __init__(
        self,
        noisy_dir: str | Path,
        clean_dir: str | Path,
        sample_rate: int = 16000,
        segment_seconds: float = 4.0,
    ) -> None:
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.sample_rate = sample_rate
        self.segment_samples = int(round(sample_rate * segment_seconds))

        noisy_files = list_audio_files(self.noisy_dir)
        pairs: list[tuple[Path, Path]] = []
        for noisy_path in noisy_files:
            clean_path = self.clean_dir / noisy_path.name
            if clean_path.exists():
                pairs.append((noisy_path, clean_path))

        if not pairs:
            clean_files = list_audio_files(self.clean_dir)
            pairs = list(zip(noisy_files, clean_files))

        if not pairs:
            raise RuntimeError(
                f"No paired audio files found in noisy={self.noisy_dir} clean={self.clean_dir}"
            )
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        noisy_path, clean_path = self.pairs[idx]
        noisy, _ = read_audio(noisy_path, sample_rate=self.sample_rate, mono=True)
        clean, _ = read_audio(clean_path, sample_rate=self.sample_rate, mono=True)
        noisy = fix_length(noisy, self.segment_samples)
        clean = fix_length(clean, self.segment_samples)
        return (
            torch.from_numpy(noisy).float(),
            torch.from_numpy(clean).float(),
            noisy_path.name,
        )
