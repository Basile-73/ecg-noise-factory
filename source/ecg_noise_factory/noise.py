from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from .utils import load_config, load_or_generate_noise

class NoiseFactory:
    def __init__(self, data_path: str, sampling_rate: int, config_path: str, mode: str = "all", seed: Optional[int] = None) -> None:
        if sampling_rate not in (100, 360, 500):
            raise ValueError("Sampling rate not supported. Choose 100, 360, or 500 Hz.")
        if mode not in ("train", "test", "eval", "all"):
            raise ValueError("Mode must be one of: train, test, eval, all.")

        self.sampling_rate: int = sampling_rate
        self.mode: str = mode
        self.data_path: Path = Path(data_path)
        self.config: Dict[str, Any] = load_config(config_path)
        self.noise_types: List[str] = ["bw", "ma", "em", "AWGN"]
        self.rng = np.random.default_rng(seed)

        # Load or auto-generate signals
        self.bw: np.ndarray = load_or_generate_noise(self.data_path, "bw", self.sampling_rate, self.mode)
        self.ma: np.ndarray = load_or_generate_noise(self.data_path, "ma", self.sampling_rate, self.mode)
        self.em: np.ndarray = load_or_generate_noise(self.data_path, "em", self.sampling_rate, self.mode)

    def add_noise(
        self,
        x: np.ndarray,
        batch_axis: int,
        channel_axis: int,
        length_axis: int,
    ) -> np.ndarray:
        """
        Add all noise types (bw, ma, em, AWGN) to ECGs using SNR values from config.
        Shape of `x` is arbitrary; specify axes.
        """
        x = np.array(x, copy=True)
        rng = self.rng
        noise_types = list(self.config["SNR"].keys())  # e.g. ["bw", "ma", "em", "AWGN"]

        # Permute to (B, C, L, extra...)
        axes = [batch_axis, channel_axis, length_axis]
        perm = axes + [i for i in range(x.ndim) if i not in axes]
        x_perm = np.transpose(x, perm)
        B, C, L = x_perm.shape[:3]
        tail_shape = x_perm.shape[3:]

        # Flatten extra dims, only use first slice
        x_core = x_perm.reshape(B, C, L, -1)[..., 0].astype(np.float32)

        noisy = x_core.copy()

        # --- AWGN ---
        noise = rng.standard_normal((B, C, L)).astype(np.float32)
        Px = (noisy ** 2).sum(axis=2)
        snr = float(self.config["SNR"]["AWGN"])
        Pn = Px / (10.0 ** (snr / 10.0))
        Pn_prime = (noise ** 2).sum(axis=2).clip(min=1e-12)
        scale = np.sqrt(Pn / Pn_prime)[..., None]
        noisy += noise * scale

        # --- Structured noise (bw, ma, em) ---
        for ntype in ("bw", "ma", "em"):
            noise_bank = getattr(self, ntype).astype(np.float32)  # (Tn, Cn)
            Tn, Cn = noise_bank.shape
            reps = int(np.ceil(L / Tn)) if Tn < L else 1
            big = np.tile(noise_bank, (reps, 1))
            Tmax = big.shape[0]
            starts = rng.integers(0, Tmax - L + 1, size=B)
            segs = np.stack([big[s:s+L, :min(C, Cn)] for s in starts], axis=0)  # (B,L,C?)
            segs = np.transpose(segs, (0, 2, 1))  # (B,C,L)

            # Match channel count
            if segs.shape[1] < C:
                segs = np.tile(segs, (1, int(np.ceil(C / segs.shape[1])), 1))[:, :C, :]

            Px = (noisy ** 2).sum(axis=2)
            snr = float(self.config["SNR"][ntype])
            Pn = Px / (10.0 ** (snr / 10.0))
            Pn_prime = (segs ** 2).sum(axis=2).clip(min=1e-12)
            scale = np.sqrt(Pn / Pn_prime)[..., None]
            noisy += segs * scale

        # Restore shape
        out = x_perm.reshape(B, C, L, -1)
        out[..., 0] = noisy
        out = out.reshape(x_perm.shape)

        # Inverse permute
        inv = np.argsort(perm)
        return np.transpose(out, inv)
