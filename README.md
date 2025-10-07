## ECG Noise Factory

Add realistic noise to ECG signals at a target SNR. The package can auto-download and prepare the MIT-BIH Noise Stress Test Database (NSTDB) noise records, then mix them into your ECG arrays with mathematically correct scaling.

### Installation

```bash
pip install -i https://test.pypi.org/simple/ ecg-noise-factory
```

Or from source (editable):

```bash
pip install -e /Users/basilemorel/code/ecg-noise-factory
```

### Required directories

- `source/ecg_noise_factory`: package code
- `configs`: YAML configs with SNR settings
- `data`: where noise `.npy` files are stored or auto-generated. This folder can start empty; it will be populated on first use.
- `dist`: built artifacts (optional if installing from source)

Only the `configs` directory and a writable `data` directory are required at runtime.

### Config file format

YAML with an `SNR` section listing per-noise-type SNR values in dB. Available keys: `bw` (baseline wander), `ma` (muscle artifact), `em` (electrode motion), and `AWGN` (white noise).

Example (`configs/default.yaml`):

```yaml
SNR:  # values in dB
  bw: 2.5   # baseline wander
  ma: 7.5   # muscle artifact
  em: 12.5  # electrode motion
  AWGN: 22.5
```

You can provide your own YAML file with any subset/superset of these keys. Missing keys are simply not applied.

### How data is downloaded and stored

On first use, if an expected noise file is missing, the package will:

1. Download the NSTDB records via PhysioNet using `wfdb` (`pn_dir="nstdb"`) for prefixes `bw`, `ma`, `em` at 360 Hz.
2. Resample to 100 Hz and 500 Hz (in addition to keeping 360 Hz).
3. Save NumPy arrays into `data/` using the pattern:

   - `data/{mode}_{prefix}_{sr}.npy`

   where:
   - `mode ∈ {train, test, eval, all}` controls which channel/slice is saved
   - `prefix ∈ {bw, ma, em}` is the noise type
   - `sr ∈ {100, 360, 500}` is the sampling rate

These files are created by the library automatically when you instantiate `NoiseFactory` if they are missing.

Dependencies used for this step: `wfdb` and `scipy` (for resampling).

### API usage

```python
import numpy as np
from ecg_noise_factory.noise import NoiseFactory

# Your ECG batch, e.g. shape (batch, channels, length)
ecg = np.random.randn(8, 1, 3000).astype(np.float32)

factory = NoiseFactory(
    data_path="/code/ecg-noise-factory/data",  # writable folder
    sampling_rate=500,                                           # 100, 360, or 500
    config_path="/code/ecg-noise-factory/configs/default.yaml",
    mode="train",                                               # train | test | eval | all
    seed=42,                                                     # optional for reproducibility
)

# Tell the library which axes are batch, channel, and length in your array
noisy = factory.add_noise(
    x=ecg,
    batch_axis=0,
    channel_axis=1,
    length_axis=2,
)
```

Axis arguments allow arbitrary input shapes. Extra dimensions beyond `(B, C, L)` are preserved.

### How noise is added (mathematical formulation)

For a clean ECG signal \(X\) and a noise signal \(N\), the library scales \(N\) so that the mixed signal achieves the requested input SNR (in dB). For each sample in a batch and channel:

1. Compute the power of the clean signal

\[ P_x = \frac{1}{n} \sum_{i=1}^{n} (X_i)^2 \]

2. Compute the expected noise power from the target SNR (in dB)

\[ P_n = \frac{P_x}{10^{\mathrm{SNR}_{\text{input}}/10}} \]

3. Compute the actual power of the raw noise segment

\[ P_n' = \sum_{i=1}^{n} (n_i)^2 \]

4. Determine the scale factor

\[ \text{scale factor} = \sqrt{\frac{P_n}{P_n'}} \]

5. Add scaled noise to the clean signal

\[ \hat{X} = X + (\text{scale factor}) \cdot N \]

This procedure is applied for each configured noise type: `AWGN`, and structured noises `bw`, `ma`, `em`. The scaling is recomputed per batch and per channel so that the resulting mixture respects the specified SNRs in your config.

### Notes

- Supported sampling rates: 100, 360, 500 Hz.
- If `data/` lacks the required `.npy` files, they will be generated automatically on first use.
- Set `mode` to choose which split you want (e.g., `train` uses channel 0; `test` and `eval` split channel 1 as implemented in `utils.py`).


