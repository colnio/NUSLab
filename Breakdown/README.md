# Crosspoint Breakdown Characterization

Walks a sample device by device through capacitance characterization and
destructive voltage stress:

```
C(f)  ->  C(V)  ->  [swap cables]  ->  RVS or CVS  ->  [swap back]  ->  next device
```

The MFIA and the Keithley 2400 share one probe station and are swapped by hand,
so the run pauses for the operator at each swap. Devices are numbered
automatically, breakdown voltages accumulate into statistics, and the program
uses those statistics to recommend the stress level for each constant-voltage
run.

## Running it

```
python -m pip install -r Breakdown/requirements.txt
python Breakdown/app_qt/main.py
```

Select **Mock** for both instruments to rehearse the whole workflow — prompts,
folder tree, plots, summary — with no hardware attached. Worth doing once before
a campaign, because the mistakes here are irreversible.

## What a run produces

```
<output>/<YYYY-MM-DD>/<sample>/
    <sample>_params_<ts>.json      full parameter snapshot, reloadable
    <sample>_summary.csv           ONE ROW PER DEVICE - analyse from this
    <sample>_<ts>.log
    5um/
        dev001/
            dev001.meta.json
            data/  CF_...csv  CV_...csv  RVS_...csv
            plots/
        dev002/
            data/  CF_...csv  CV_...csv  CVS_...csv
    20um/
        dev001/ ...
```

The device index is derived from the directory tree and scoped per
(sample, crosspoint size), so it **restarts at 1 when you change size** and
survives restarting the program mid-campaign. Set it manually or reset it from
the Setup tab.

## Operating modes

| Mode | Device 1 | Device 2 | Device 3 | ... |
|---|---|---|---|---|
| `rvs_only` | RVS | RVS | RVS | ... |
| `alternating` | RVS | CVS | RVS | ... |
| `cvs_only` | **RVS** | CVS | CVS | ... |
| `manual` | you choose | you choose | you choose | ... |

Device 1 always ramps in `cvs_only`. Alternating mode remains strict even when
an RVS device survives its ceiling: the next even-numbered device is still CVS,
and the operator must enter a custom safe voltage because no statistics-based
recommendation is available. The program never silently substitutes another
RVS. Valid RVS history and the previous CVS voltage are restored when the same
date/sample/crosspoint campaign is reopened.

## Sweep shapes

- **C(f)** — always log-spaced, always `f_min → f_max → f_min` at fixed bias.
- **C(V)** — always linear, always `0 → V_max → V_min → 0`, the full hysteresis
  loop through zero.

Every point carries a `direction` (`fwd`/`rev`) and a `segment` number, so the
branches of a loop can be separated during analysis.

## Choosing the CVS stress voltage

See [`docs/cvs-voltage-selection.md`](docs/cvs-voltage-selection.md), mirrored in
the app's Help tab. In short: `V_CVS = k × median(V_BD)` from RVS on sister
devices of the same size, `k` ≈ 0.80–0.92, and vary `k` across the batch so you
get at least three field levels to extrapolate from.

The dialog offers three ways to answer, every time:

1. the value used on the previous CVS device,
2. the statistics-derived recommendation, with its reasoning and caveats,
3. a value you type.

## Safety

- Before every cable prompt, the Keithley is driven to zero with its output
  opened. The MFIA is returned to the lab's enabled idle state: 10 mV AC,
  0 V DC bias, and 100 kHz. Cleanup runs from `finally` blocks.
- Breakdown detection uses an **absolute current threshold**, confirmed over two
  consecutive points so a single noisy sample cannot end a device. The run
  refuses to start if that threshold is at or above compliance, since breakdown
  could then never trigger.
- The window refuses to close while a run is live: a CVS hold means the SMU is
  energised on a probe.
- Stop is checked every point, and also releases a prompt that is waiting on
  screen.
- Every CVS voltage is checked against the run's hard voltage ceiling in both
  the dialog and the core engine.
- Before destructive stress, three zero-volt reads calibrate the Keithley's real
  timing. If the requested ramp or sampling cadence is not achievable, the
  operator must explicitly accept the calibrated rate.
- Invalid instrument reads and critical setting failures get one retry. A
  repeated fault safely aborts the campaign while retaining partial data and
  lifecycle metadata.

## Layout

```
core/       no Qt imports at all - the engine
  params    dataclasses, JSON save/load, validation
  naming    output paths, device index scoping
  sweeps    C(f) and C(V) runners      sweep_points  their point geometry
  stress    RVS and CVS runners        ramp          rate -> step/dwell planning
  detect    breakdown criterion        advisor       CVS level recommendation
  session   the per-device state machine
  storage   CSV writers, summary table, metadata
  events    Prompter protocol - the UI seam
  mfia      MFIA session (zhinst)      mfia_nodes    poll payload normalisation
  smu       Keithley session (pyvisa)  models        impedance model arithmetic
  mock      simulated instruments
app_qt/     PyQt5 shell - wizard, panels, dialogs, plots
```

`core/` imports no GUI toolkit and only touches `zhinst`/`pyvisa` lazily inside
`mfia.py`/`smu.py`, so the engine is testable without instrument drivers and a
different front-end could reuse it unchanged.

Reused from elsewhere in the repo rather than reimplemented:
`KeithleyGUI/keithley.py` (2400 driver, discovery, safe shutdown) and
`KeithleyGUI/ui_helpers.py` (directories, JSON, numeric parsing, shared style).
The MFIA node-access helpers are ported from `CV_MAP/cv_cf_recipe_ui.py`;
`CV_MAP/` itself is left untouched.

## Tests

```
python -m pytest Breakdown/tests                    # everything (~65 s)
python -m pytest Breakdown/tests -m "not slow"      # core only (<1 s)
```

The slow ones drive a real Qt event loop end to end with mock instruments. The
rest need neither hardware nor instrument libraries.
