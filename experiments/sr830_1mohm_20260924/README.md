# SR830 / nominal 1 MΩ resistor bench measurements

Measured through the local GPIB REST server on 2026-09-24, using SR830 serial **40423** (firmware 1.07). The user connected a nominal 1 MΩ resistor between **Sine Out and A/I**. No physical rewiring was performed. All auxiliary outputs were kept at zero; no DC bias was applied. Source amplitudes are RMS.

This experiment measures the combined response of the resistor, wiring, source and input amplifier. The resistor and source amplitude were not independently calibrated. A peak or cutoff in the measured ratio must not be attributed to the resistor alone.

## Results

The low-frequency current measurement is reasonable for the nominal resistor: approximately **0.986 µS**, equivalent to **1.014 MΩ** for the entire measured ratio. Current is very nearly linear with source amplitude. At 13.033 Hz, fitting RMS current against 4–500 mV RMS gives a slope of **0.9848 µS**; the individual `I/V` ratios span 0.9849–0.9900 µS. The small amplitude dependence includes source-setting accuracy and cannot be assigned solely to the resistor.

| Test | Observed behavior |
| --- | --- |
| 10⁶ V/A current input, DC coupling | Nearly flat through a few kHz; peak **2.642 µS at 38 kHz**, then strong attenuation |
| 10⁶ V/A descending −3 dB crossing | About **61.1 kHz**, relative to the low-frequency plateau; bracketed by 60 and 63.33 kHz |
| 10⁸ V/A diagnostic | Approximately **768 Hz** −3 dB crossing, bracketed by 700 and 933 Hz; 1 MΩ is below this gain's recommended source impedance |
| AC versus DC coupling | Fitted high-pass corner **0.1612 Hz**, agreeing with the nominal 0.16 Hz specification; at 0.04 Hz AC passes only 24.1% of the DC-coupled signal |
| Same wiring in voltage-input mode | Complex RC fit through 10.33 kHz: corner **1.098 kHz**, equivalent input-plus-wiring capacitance **about 160 pF** under the nominal 1 MΩ / 10 MΩ model |
| Frequency–amplitude map | 82 measured cells and two explicit skips: 500 mV at 33.33 and 38 kHz would leave insufficient range headroom |

The current-mode peak repeats across amplitudes, including a separate wider-range frequency run, so it is not explained by the initial range clipping. The peak and associated phase rotation are properties of the complete measurement path. They make high-frequency `I/V` unsuitable as an uncorrected resistor conductance in this setup. The measured plateau supports working at tens of hertz to about 1 kHz with the 10⁶ V/A input for this resistor; extrapolation to other samples or wiring requires another transfer check.

The current-input −3 dB crossings use logarithmic interpolation between measured frequencies. They are not precision-calibrated specifications or confidence-bounded fits. See `summary.json` for exact calculation outputs and the plots for the observed curves.

The explicitly fitted AC-coupling and voltage-mode corners use RC models. The voltage-mode fit quantifies loading but does not identify a unique physical capacitor or prove the mechanism behind the current-mode peak. Its equivalent capacitance is consistent with input and cable loading. Points above 10.33 kHz are shown but excluded from that model fit.

The final analysis retains **151 current settings and 11 voltage settings**, with three readings per setting. Nine completed acquisition protocols produced **702 server CSV rows**, including settling checks and retained repeats. All 702 were cross-checked against the client logs and their Dropbox mirror. The initial clipped current points and unsettled voltage point remain inspectable; plots use their corrected repeats. An additional final live check verified both read endpoints and left the instrument at 13.033 Hz, 10⁶ V/A current input, DC coupling, **4 mV RMS** and **zero AUX outputs**. The server was ready with zero active sessions afterward.

Plots: [frequency](plots/signal_vs_frequency.png), [amplitude](plots/signal_vs_amplitude.png), [map](plots/conductance_map.png), [coupling](plots/input_coupling_cutoff.png), [voltage diagnostic](plots/voltage_input_diagnostic.png).

## Software validation

The final full server suite passed **97 tests**. It covers the new driver, invalid requests, both current gains and units, register semantics, configuration rounding, range quality flags, four-device sessions, mixed samples, lifecycle cleanup, failure/recovery paths, and preservation of historical schema-1 and schema-2 files. The previously observed sweep-completion timing timeout did not recur in the final full runs; no timeout was relaxed for this work.

API **1.1.1** was deployed through the existing Windows startup task after confirming there were no active sessions. Live checks exercised discovery, typed configuration/readback, individual and grouped readings, raw identity/offset queries, heartbeat, range widening and cleanup. `verify.py` independently joins every client reading to the server CSV by timestamp, compares all lock-in numeric fields and statuses, checks local/Dropbox data equality, and verifies cleanup. Its recorded results are in `verification.json`.

## Data and reproduction

- `measure.py` controls the actual instrument exclusively through the REST API, discovers serial 40423 dynamically, maintains the session lease and verifies cleanup.
- `*.jsonl` preserve actual configurations, native readings, transient status flags, requests, server run IDs and cleanup results. An unsuccessful startup attempt is retained and excluded from analysis.
- `measurements.csv` preserves every current replicate from completed protocols, including the initially over-range readings with explicit flags. `voltage_measurements.csv` contains the voltage-input diagnostic.
- `points.csv` contains the accepted three-reading averages. Out-of-range points are excluded, and later repeats at the same settings supersede earlier points. `exclusions.json` records excluded readings.
- `analyze.py` regenerates the plots and numerical summaries from those files, using the repository's existing NumPy and Matplotlib dependencies.
- The server also independently saves CSV/event/metadata recordings locally and mirrors them to the configured Dropbox folder; each JSONL records its run identity and storage result.

Run from the repository root with the resistor wiring confirmed:

```powershell
.\.venv\Scripts\python.exe experiments\sr830_1mohm_20260924\measure.py --protocol pilot
.\.venv\Scripts\python.exe experiments\sr830_1mohm_20260924\measure.py --protocol frequency
.\.venv\Scripts\python.exe experiments\sr830_1mohm_20260924\measure.py --protocol frequency_refine
.\.venv\Scripts\python.exe experiments\sr830_1mohm_20260924\measure.py --protocol high_gain
.\.venv\Scripts\python.exe experiments\sr830_1mohm_20260924\measure.py --protocol map
.\.venv\Scripts\python.exe experiments\sr830_1mohm_20260924\measure.py --protocol map_refine
.\.venv\Scripts\python.exe experiments\sr830_1mohm_20260924\measure.py --protocol coupling
.\.venv\Scripts\python.exe experiments\sr830_1mohm_20260924\measure.py --protocol voltage_check
.\.venv\Scripts\python.exe experiments\sr830_1mohm_20260924\analyze.py
```

Each protocol reserves and releases its own session. The initial pilot used a wide 1 µA sensitivity; subsequent measurements used tighter ranges or widened them to preserve headroom. The map skips a higher amplitude if the preceding lower-amplitude result predicts more than 80% of the maximum 1 µA range. Skipped cells are shown explicitly rather than interpolated.

## Measurement definitions and settling

For this nominally linear resistor, the AC ratio `|I| / Vsource` estimates the magnitude of the small-signal conductance. It is not a DC I–V curve or a numerical bias derivative. `Vsource` is the instrument's **actual programmed amplitude readback**, not an independent measurement of the voltage directly across the resistor. The nominal prediction is 1 µS; including the manual's 50 Ω source resistance and approximately 1 kΩ current-input burden gives about 0.99895 µS for an exact 1 MΩ resistor.

The native current readings are already in amperes at both preamplifier gains; no extra gain division is applied. The current input produces approximately 180° inversion. For phase plots only, the coherent complex pair is represented as `-(X + iY)` before taking its argument. Native X, Y, magnitude and phase are preserved separately. Native magnitude/phase and X/Y are sampled about 10 µs apart and need not be algebraically identical.

The SR830 read endpoint returns the currently filtered value and does not wait. The initial protocols explicitly waited at least **15 actual time constants and three source periods**, with a minimum 1.5 seconds. AC coupling additionally waits at least 15 seconds. The client consumes a settling check, then records three reads spaced by at least three time constants and one source period. Time constants are 0.1 or 0.3 seconds with 24 dB/octave slope, low-noise reserve, no notch filtering, internal reference at harmonic 1 and synchronous filtering selected (active below about 200 Hz). These conditions and all readbacks are logged per point.

The first voltage-mode diagnostic exposed a longer transition transient in its first point. That point was excluded and the voltage diagnostic repeated. The final client uses a minimum five-second wait, a minimum 15 seconds after changing input mode, and rejects a replicate set whose magnitude spread exceeds 0.5% of its mean. Actual waits used by each historical run remain in its JSONL. All retained current sets passed the same spread check; it is a practical settling/noise check, not proof of absolute accuracy.

Repeat standard deviations describe short-term repeatability only; filtered readings can remain correlated. They are not calibration uncertainties or confidence intervals. Zero measured scatter can reflect output resolution, not perfect accuracy.

## Instrument limits and interpretation

The [SR830 manual](https://www.thinksrs.com/downloads/PDFs/Manuals/SR830m.pdf), sections 3-20 and 4-9, gives nominal current-input bandwidths of 70 kHz at 10⁶ V/A and 700 Hz at 10⁸ V/A. It recommends source impedance greater than 1 MΩ and 100 MΩ respectively. The high-gain run with this 1 MΩ resistor is therefore a **diagnostic bandwidth comparison**, not a precision-current calibration under recommended source conditions.

The AC-coupling high-pass corner is specified at 0.16 Hz. The current preamplifier itself remains DC coupled. Voltage-input mode is specified as 10 MΩ in parallel with 25 pF; wiring adds capacitance.

During the first frequency sweep, native magnitude clipped near 1.09 times full scale while the native overload register remained clear. Those readings were excluded and the affected region repeated on a wider range. The API now independently reports `sensitivity` and `range_exceeded` in both individual and grouped readings and records them in schema 3, preserving native status flags. The client additionally requires headroom rather than relying on the overload register alone.

All sessions end by verifying **4 mV RMS Sine Out and zero on AUX 1–4**. AC remains active; this is output minimization, not output-off.
