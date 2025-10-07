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


def download_and_prepare_all(data_path: Path, prefixes: List[str], modes: List[str]) -> None:
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


def load_or_generate_noise(data_path: Path, prefix: str, sampling_rate: int, mode: str) -> np.ndarray:
    """
    Load noise data for a given prefix, sampling_rate, and mode.
    If missing, generate all files by downloading 360 Hz and resampling.
    """
    file_path = data_path / noise_filename(prefix, sampling_rate, mode)
    if not file_path.exists():
        download_and_prepare_all(data_path, ["bw", "ma", "em"], ["train", "test", "eval", "all"])
    return np.load(file_path)
