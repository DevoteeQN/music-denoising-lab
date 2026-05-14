from __future__ import annotations

import numpy as np


def snr_db(clean: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    clean = np.asarray(clean, dtype=np.float32)
    estimate = np.asarray(estimate, dtype=np.float32)
    length = min(len(clean), len(estimate))
    clean = clean[:length]
    estimate = estimate[:length]
    noise = estimate - clean
    signal_power = float(np.sum(clean**2) + eps)
    noise_power = float(np.sum(noise**2) + eps)
    return 10.0 * np.log10(signal_power / noise_power)


def snr_improvement(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> float:
    return snr_db(clean, denoised) - snr_db(clean, noisy)
