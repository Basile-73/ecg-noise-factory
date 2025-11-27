from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from .utils import (
    load_config,
    load_or_generate_noise,
    generate_sine_noise,
    generate_spike_noise,
    generate_trunc_noise,
)


class NoiseFactory:
    def __init__(
        self,
        data_path: str,
        sampling_rate: int,
        config_path: str,
        mode: str = "all",
        seed: Optional[int] = None,
    ) -> None:
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
        self.bw: np.ndarray = load_or_generate_noise(
            self.data_path, "bw", self.sampling_rate, self.mode
        )
        self.ma: np.ndarray = load_or_generate_noise(
            self.data_path, "ma", self.sampling_rate, self.mode
        )
        self.em: np.ndarray = load_or_generate_noise(
            self.data_path, "em", self.sampling_rate, self.mode
        )

        # Check and store parameters for new noise types
        if "sine_noise" in self.config and self.config["sine_noise"].get(
            "include", True
        ):
            self.sine_noise_params = self.config["sine_noise"]
            self.noise_types.append("sine_noise")

        if "spike_noise" in self.config and self.config["spike_noise"].get(
            "include", True
        ):
            self.spike_noise_params = self.config["spike_noise"]
            self.noise_types.append("spike_noise")

        if "trunc_noise" in self.config and self.config["trunc_noise"].get(
            "include", True
        ):
            self.trunc_noise_params = self.config["trunc_noise"]
            self.noise_types.append("trunc_noise")

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

        # Permute to (B, C, L, extra...)
        axes = [batch_axis, channel_axis, length_axis]
        perm = axes + [i for i in range(x.ndim) if i not in axes]
        x_perm = np.transpose(x, perm)
        B, C, L = x_perm.shape[:3]

        # Flatten extra dims, only use first slice
        x_core = x_perm.reshape(B, C, L, -1)[..., 0].astype(np.float32)

        noisy = x_core.copy()

        # --- AWGN ---
        if "AWGN" in self.config["SNR"] and self.config["SNR"]["AWGN"] is not None:
            noise = rng.standard_normal((B, C, L)).astype(np.float32)
            Px = (noisy**2).sum(axis=2)
            snr = float(self.config["SNR"]["AWGN"])
            Pn = Px / (10.0 ** (snr / 10.0))
            Pn_prime = (noise**2).sum(axis=2).clip(min=1e-12)
            scale = np.sqrt(Pn / Pn_prime)[..., None]
            noisy += noise * scale

        # --- Sine noise ---
        if hasattr(self, "sine_noise_params"):
            params = self.sine_noise_params
            snr = float(params["snr"])
            min_freq = float(params["min_freq"])
            max_freq = float(params["max_freq"])
            num_frequencies = int(params["number_of_frequencies"])

            sine_noise = np.zeros((B, C, L), dtype=np.float32)
            for b in range(B):
                for c in range(C):
                    sine_noise[b, c, :] = generate_sine_noise(
                        L,
                        self.sampling_rate,
                        min_freq,
                        max_freq,
                        num_frequencies,
                        rng,
                    )

            Px = (noisy**2).sum(axis=2)
            Pn = Px / (10.0 ** (snr / 10.0))
            Pn_prime = (sine_noise**2).sum(axis=2).clip(min=1e-12)
            scale = np.sqrt(Pn / Pn_prime)[..., None]
            noisy += sine_noise * scale

        # --- Structured noise (bw, ma, em) ---
        for ntype in ("bw", "ma", "em"):
            if ntype not in self.config["SNR"] or self.config["SNR"][ntype] is None:
                continue

            noise_bank = getattr(self, ntype).astype(np.float32)  # (Tn, Cn)
            Tn, Cn = noise_bank.shape
            reps = int(np.ceil(L / Tn)) if Tn < L else 1
            big = np.tile(noise_bank, (reps, 1))
            Tmax = big.shape[0]
            starts = rng.integers(0, Tmax - L + 1, size=B)
            segs = np.stack(
                [big[s : s + L, : min(C, Cn)] for s in starts], axis=0
            )  # (B,L,C?)
            segs = np.transpose(segs, (0, 2, 1))  # (B,C,L)

            # Match channel count
            if segs.shape[1] < C:
                segs = np.tile(segs, (1, int(np.ceil(C / segs.shape[1])), 1))[:, :C, :]

            Px = (noisy**2).sum(axis=2)
            snr = float(self.config["SNR"][ntype])
            Pn = Px / (10.0 ** (snr / 10.0))
            Pn_prime = (segs**2).sum(axis=2).clip(min=1e-12)
            scale = np.sqrt(Pn / Pn_prime)[..., None]
            noisy += segs * scale

        # --- Spike noise ---
        if hasattr(self, "spike_noise_params"):
            params = self.spike_noise_params
            amplitude = float(params["amplitude"])
            spike_duration = int(params["spike_duration"])
            min_spikes = int(params["min_number_of_spikes"])
            max_spikes = int(params["max_number_of_spikes"])

            spike_noise = np.zeros((B, C, L), dtype=np.float32)
            for b in range(B):
                for c in range(C):
                    spike_noise[b, c, :] = generate_spike_noise(
                        L, amplitude, spike_duration, min_spikes, max_spikes, rng
                    )

            noisy += spike_noise

        # --- Truncation noise ---
        if hasattr(self, "trunc_noise_params"):
            params = self.trunc_noise_params
            min_length = int(params["min_length"])
            max_length = int(params["max_length"])

            for b in range(B):
                for c in range(C):
                    noisy[b, c, :] = generate_trunc_noise(
                        noisy[b, c, :], min_length, max_length, rng
                    )

        # Restore shape
        out = x_perm.reshape(B, C, L, -1)
        out[..., 0] = noisy
        out = out.reshape(x_perm.shape)

        # Inverse permute
        inv = np.argsort(perm)
        return np.transpose(out, inv)
