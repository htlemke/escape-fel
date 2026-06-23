# Pump-Probe Analysis Example

This example reproduces the key steps of a time-resolved X-ray scattering
experiment: loading / generating data, normalising to an intensity reference,
applying timetool correction, and computing the differential signal.

## Setting Up

```python
import numpy as np
import matplotlib.pyplot as plt
import escape
from escape.storage.example_data import make_pump_probe_scan

# Synthetic pump-probe dataset
# sig   — detector signal (events from both laser-ON and laser-OFF shots)
# i0    — incoming X-ray intensity (correlated with sig)
# pump_on — boolean per event: True = laser was fired ("pumped")
# delay — scan delay value as an Array (event-aligned)

sig, i0, pump_on, delay = make_pump_probe_scan(
    n_steps=20,
    n_events_per_step=600,
    noise=0.06,
    i0_noise=0.04,
    seed=42,
)

print(sig.scan)
# Scan over 20 steps
# Parameters delay_s
```

## Step 1 — Separate Pump-On and Pump-Off Shots

```python
# ~50 % of events per step are pumped; the rest serve as reference (off)
sig_on  = sig[~pump_on]   # laser-pumped signal
sig_off = sig[pump_on]    # unpumped reference

i0_on  = i0[~pump_on]
i0_off = i0[pump_on]

print(f"ON  events per step: {sig_on.scan.count()[:3]} …")
print(f"OFF events per step: {sig_off.scan.count()[:3]} …")
```

## Step 2 — Normalise to I0

Index alignment makes normalisation trivial — the division operator finds
common pulse IDs automatically:

```python
sig_on_norm  = sig_on  / i0_on
sig_off_norm = sig_off / i0_off
```

## Step 3 — Drift Correction

A slow drift in the source or sample can be removed by dividing each ON shot
by the average of its neighbouring OFF shots.  We use
{meth}`~escape.Array.get_index_array` to group shots into time bins of
1000 pulse IDs, then {meth}`~escape.Array.categorize` to apply that grouping:

```python
# Group pulse IDs into bins of ~1000
time_bins = sig_on_norm.get_index_array(N_index_aggregation=1000)

# Apply the same grouping to both ON and OFF
on_rebinned  = time_bins.categorize(sig_on_norm)
off_rebinned = time_bins.categorize(sig_off_norm)

# Per time-bin reference average
ref_per_bin = off_rebinned.scan.nanmean(axis=0)

# Divide ON by OFF average within each bin → drift-corrected ratio
ratio_drift = on_rebinned.scan / ref_per_bin
```

## Step 4 — Per-Step Pump/Probe Ratio

```python
# Simple version without drift correction — divide per-step ON mean by OFF mean
step_means_on  = np.asarray(sig_on_norm.scan.nanmean())
step_means_off = np.asarray(sig_off_norm.scan.nanmean())

ratio = step_means_on / step_means_off - 1.0   # relative change ΔI/I

par_vals = np.asarray(sig.scan.par_steps["delay_s"])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(par_vals * 1e12, ratio, "o-", lw=1.5)
ax.axhline(0, ls="--", color="k", alpha=0.4)
ax.set_xlabel("delay / ps")
ax.set_ylabel("ΔI/I")
ax.set_title("Pump-probe signal (no jitter correction)")
plt.tight_layout()
plt.show()
```

## Step 5 — Timetool Jitter Correction with `digitize`

Timing jitter between the pump and probe pulses broadens the apparent response.
Correcting it requires re-binning events by the timetool-measured actual delay
rather than the nominal scan parameter.

Here we simulate a timetool measurement:

```python
rng = np.random.default_rng(0)

# True time = nominal delay + random jitter (~100 fs rms)
jitter = rng.normal(0, 100e-15, len(sig))
t_jitter = escape.Array(
    data=(np.repeat(par_vals, 600) + jitter).astype(np.float64),
    index=sig.index,
    step_lengths=sig.scan.step_lengths,
    parameter=sig.scan.parameter,
    name="t_actual_s",
)

# Compute to numpy before digitizing (required by digitize)
t_jitter_np = t_jitter.compute()

# Re-bin onto 50 fs bins covering the delay range
t_bins = np.arange(par_vals.min(), par_vals.max(), 50e-15)
t_sorted = t_jitter_np.digitize(t_bins)

# Apply the same bin ordering to the normalised signal
ratio_tt = t_sorted.categorize(sig_on_norm)

# Per-bin ratio
ratio_tt_mean = np.asarray(ratio_tt.scan.nanmean())
ratio_tt_sem  = np.asarray(ratio_tt.scan.nanstd()) / np.sqrt(ratio_tt.scan.count())
t_centers = np.asarray(ratio_tt.scan.par_steps.iloc[:, 0]) * 1e12   # → ps

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(t_centers, ratio_tt_mean, yerr=ratio_tt_sem, fmt="o-", lw=1.5,
            label="timetool corrected")
ax.plot(par_vals * 1e12, ratio, "s--", alpha=0.5, label="nominal delay")
ax.axhline(1.0, ls="--", color="k", alpha=0.3)
ax.set_xlabel("delay / ps")
ax.set_ylabel("signal (normalised)")
ax.legend()
ax.set_title("Pump-probe signal with timing jitter correction")
plt.tight_layout()
plt.show()
```

## Step 6 — Storing Results

```python
import h5py

with h5py.File("pump_probe_results.h5", "w") as fh:
    sig.store(fh,            "signal")
    i0.store(fh,             "i0")
    pump_on.store(fh,        "pump_on")
    sig_on_norm.store(fh,    "signal_norm_on")
    ratio_tt.store(fh,       "signal_tt_corrected")
```
