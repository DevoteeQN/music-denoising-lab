from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.io import wavfile


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.float32:
        out = audio
    elif audio.dtype == np.float64:
        out = audio.astype(np.float32)
    elif np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        out = audio.astype(np.float32) / max_value
    else:
        out = audio.astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(audio, up, down, axis=0).astype(np.float32)


def _read_with_ffmpeg(path: Path, sample_rate: int, mono: bool) -> tuple[np.ndarray, int]:
    ffmpeg = get_ffmpeg_exe()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg is required to read non-WAV audio. Install ffmpeg or `pip install imageio-ffmpeg`."
        )
    channels = "1" if mono else "2"
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        channels,
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if not mono:
        audio = audio.reshape(-1, 2)
    return audio.astype(np.float32), sample_rate


def read_audio(path: str | Path, sample_rate: int = 16000, mono: bool = True) -> tuple[np.ndarray, int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".wav":
        try:
            sr, audio = wavfile.read(path)
            audio = _to_float32(audio)
            if audio.ndim == 2 and mono:
                audio = audio.mean(axis=1)
            audio = _resample(audio, sr, sample_rate)
            return np.asarray(audio, dtype=np.float32), sample_rate
        except Exception:
            pass

    audio, sr = _read_with_ffmpeg(path, sample_rate=sample_rate, mono=mono)
    return np.asarray(audio, dtype=np.float32), sr


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    ensure_dir(path.parent)
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sample_rate, (audio * 32767.0).astype(np.int16))


def _write_mp3(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    ffmpeg = get_ffmpeg_exe()
    if not ffmpeg:
        fallback = path.with_suffix(".wav")
        _write_wav(fallback, audio, sample_rate)
        raise RuntimeError(
            f"FFmpeg is required to write MP3. Wrote WAV fallback instead: {fallback}"
        )
    ensure_dir(path.parent)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_wav = Path(tmp_dir) / "audio.wav"
        _write_wav(tmp_wav, audio, sample_rate)
        cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(tmp_wav),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(path),
        ]
        subprocess.run(cmd, check=True)


def write_audio(path: str | Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        _write_mp3(path, audio, sample_rate)
    else:
        _write_wav(path, audio, sample_rate)


def write_audio_with_mp3(path: str | Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path = Path(path)
    write_audio(path, audio, sample_rate)
    if path.suffix.lower() != ".mp3":
        mp3_path = path.with_suffix(".mp3")
        write_audio(mp3_path, audio, sample_rate)


def list_audio_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    return sorted(files)


def peak_normalize(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > peak and max_abs > 0:
        audio = audio / max_abs * peak
    return audio.astype(np.float32)


def match_rms(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    ref_rms = float(np.sqrt(np.mean(np.square(reference)) + eps))
    est_rms = float(np.sqrt(np.mean(np.square(estimate)) + eps))
    if est_rms <= eps:
        return estimate.astype(np.float32)
    return peak_normalize(estimate * (ref_rms / est_rms)).astype(np.float32)


def fix_length(audio: np.ndarray, length: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == length:
        return audio
    if len(audio) > length:
        return audio[:length]
    if len(audio) == 0:
        return np.zeros(length, dtype=np.float32)
    reps = int(np.ceil(length / len(audio)))
    return np.tile(audio, reps)[:length].astype(np.float32)


def random_crop_or_pad(audio: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) > length:
        start = int(rng.integers(0, len(audio) - length + 1))
        return audio[start : start + length].astype(np.float32)
    return fix_length(audio, length)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(clean, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    p_clean = float(np.mean(clean**2) + eps)
    p_noise = float(np.mean(noise**2) + eps)
    alpha = math.sqrt(p_clean / (p_noise * (10.0 ** (snr_db / 10.0))))
    noisy = clean + alpha * noise
    max_abs = float(np.max(np.abs(noisy))) if noisy.size else 0.0
    if max_abs > 0.99:
        noisy = noisy / max_abs * 0.99
        clean = clean / max_abs * 0.99
    return noisy.astype(np.float32), clean.astype(np.float32)


def load_random_segment(path: str | Path, sample_rate: int, length: int, rng: np.random.Generator) -> np.ndarray:
    audio, _ = read_audio(path, sample_rate=sample_rate, mono=True)
    return random_crop_or_pad(audio, length=length, rng=rng)


def spectral_gate_denoise(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    noise_reduce_strength: float = 0.8,
    prop_decrease: float = 0.8,
) -> np.ndarray:
    del sample_rate
    x = torch.tensor(audio, dtype=torch.float32)
    if x.numel() < win_length:
        x = F.pad(x, (0, win_length - x.numel()))
    window = torch.hann_window(win_length)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    mag = spec.abs()
    noise_profile = torch.quantile(mag, q=0.2, dim=-1, keepdim=True)
    threshold = noise_profile * (1.0 + noise_reduce_strength)
    softness = noise_profile * 0.5 + 1e-6
    soft_mask = torch.sigmoid((mag - threshold) / softness)
    mask = (1.0 - prop_decrease) + prop_decrease * soft_mask
    enhanced_spec = spec * mask
    enhanced = torch.istft(
        enhanced_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=len(audio),
    )
    y = enhanced.detach().cpu().numpy().astype(np.float32)
    y = match_rms(np.asarray(audio, dtype=np.float32), y)
    return peak_normalize(y)


def parse_snr_list(values: str | Iterable[float]) -> list[float]:
    if isinstance(values, str):
        return [float(v.strip()) for v in values.split(",") if v.strip()]
    return [float(v) for v in values]
