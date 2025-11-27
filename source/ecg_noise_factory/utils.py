from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import yaml
import wfdb
from scipy.signal import resample


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def noise_filename(prefix: str, sampling_rate: int, mode: str) -> str:
    """Consistent filename pattern including mode."""
    return f"{mode}_{prefix}_{sampling_rate}.npy"


def split_channels(data: np.ndarray, mode: str) -> np.ndarray:
    """
    Split noise data by mode:
    - train: channel 0
    - test: first half of channel 1
    - eval: second half of channel 1
    - all: full data
    """
    if mode == "train":
        return data[:, 0:1]  # keep 2D shape
    elif mode == "test":
        half = data.shape[0] // 2
        return data[:half, 1:2]
    elif mode == "eval":
        half = data.shape[0] // 2
        return data[half:, 1:2]
    elif mode == "all":
        return data
    else:
        raise ValueError("Mode must be one of: train, test, eval, all.")


def download_and_prepare_all(
    data_path: Path, prefixes: List[str], modes: List[str]
) -> None:
    """
    Download 360 Hz PhysioNet noise records, resample to 100/500 Hz,
    and save all mode splits.
    """
    data_path.mkdir(parents=True, exist_ok=True)

    for prefix in prefixes:
        # Download original 360 Hz record
        record = wfdb.rdrecord(prefix, pn_dir="nstdb").p_signal

        for sr in (100, 360, 500):
            if sr == 360:
                rec = record
            else:
                rec = resample(record, int(record.shape[0] * sr / 360), axis=0)

            # Save splits
            for mode in modes:
                split = split_channels(rec, mode)
                np.save(data_path / noise_filename(prefix, sr, mode), split)


def load_or_generate_noise(
    data_path: Path, prefix: str, sampling_rate: int, mode: str
) -> np.ndarray:
    """
    Load noise data for a given prefix, sampling_rate, and mode.
    If missing, generate all files by downloading 360 Hz and resampling.
    """
    file_path = data_path / noise_filename(prefix, sampling_rate, mode)
    if not file_path.exists():
        download_and_prepare_all(
            data_path, ["bw", "ma", "em"], ["train", "test", "eval", "all"]
        )
    return np.load(file_path)


def generate_sine_noise(
    length: int,
    sampling_rate: int,
    min_freq: float,
    max_freq: float,
    num_frequencies: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate sine wave noise by summing multiple sine waves with random frequencies and phases.

    Args:
        length: Length of the noise signal in samples
        sampling_rate: Sampling rate in Hz
        min_freq: Minimum frequency in Hz
        max_freq: Maximum frequency in Hz
        num_frequencies: Number of sine waves to sum
        rng: NumPy random number generator

    Returns:
        Noise array of shape (length,) with dtype float32
    """
    noise = np.zeros(length, dtype=np.float32)
    t = np.arange(length) / sampling_rate

    for _ in range(num_frequencies):
        freq = rng.uniform(min_freq, max_freq)
        phase = rng.uniform(0, 2 * np.pi)
        noise += np.sin(2 * np.pi * freq * t + phase)

    return noise


def generate_spike_noise(
    length: int,
    amplitude: float,
    spike_duration: int,
    min_spikes: int,
    max_spikes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate spike noise by placing random spikes at random locations in the signal.

    Args:
        length: Length of the noise signal in samples
        amplitude: Amplitude of each spike
        spike_duration: Duration of each spike in samples
        min_spikes: Minimum number of spikes
        max_spikes: Maximum number of spikes
        rng: NumPy random number generator

    Returns:
        Noise array of shape (length,) with dtype float32
    """
    noise = np.zeros(length, dtype=np.float32)
    num_spikes = rng.integers(min_spikes, max_spikes + 1)

    if num_spikes > 0:
        spike_positions = rng.choice(length, size=num_spikes, replace=False)

        for pos in spike_positions:
            end_pos = min(pos + spike_duration, length)
            noise[pos:end_pos] = amplitude

    return noise


def generate_trunc_noise(
    signal: np.ndarray,
    min_length: int,
    max_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Truncate the signal at a random location by zeroing out a segment of random length.

    Args:
        signal: Input signal array
        min_length: Minimum truncation length in samples
        max_length: Maximum truncation length in samples
        rng: NumPy random number generator

    Returns:
        Truncated signal array (copy of input with zeroed segment)
    """
    truncated = signal.copy()
    trunc_length = rng.integers(min_length, max_length + 1)
    start_pos = rng.integers(0, len(signal) - trunc_length + 1)
    truncated[start_pos : start_pos + trunc_length] = 0

    return truncated
