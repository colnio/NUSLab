# NUSLab GPIB server

Local REST control for the Keithley 2400, 6430, and 2002, and the Stanford Research Systems SR830 lock-in. The service uses NI-VISA through PyVISA and listens only on `127.0.0.1:8765`. Interactive API documentation is available at `http://127.0.0.1:8765/docs` after startup.

## Install and start

From the repository root, run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\GPIBServer\install_autostart.ps1` once while signed in as the lab user. This machine blocks direct PowerShell script execution. The installer adds the pinned Python dependencies to the repository's `.venv`, creates a single-instance Task Scheduler job, and starts the server. Use the same PowerShell command with `start_server.ps1` to start it manually. The task restarts after a failure and starts again at sign-in.

Use the existing GUI programs only when the server is stopped. The server reserves each GPIB resource for one session at a time; another process cannot obey that reservation. On startup, the server identifies instruments with `*IDN?`, then aborts and turns off recognized SMUs. Discovery itself does not clear or reset instruments. A device that cannot be identified or shut down appears in `/v1/health` or `/v1/devices` as an error and is unavailable for sessions.

For the SR830, source cleanup means **4 mV RMS Sine Out and 0 V on AUX outputs 1–4**. The sine output remains active: this instrument has no software output-off switch. Health, cleanup responses and event records use `outputs_minimized`, not `off`. This policy applies at startup, reservation, release, rollback, lease expiry, recovery, and server shutdown, and after communication, command/hardware or recording failures. Failed cleanup verification blocks further use until the session is released and explicit recovery succeeds. Overload and reference-unlock indications are measurement flags, not command failures.

The default Dropbox root is:

```text
C:\Users\MeasurmentStand\NUS Dropbox\Ozyilmaz Group\1_MAC\1_Projects\Iurii\ResistiveSwitches\VacuumProbeData
```

Override it with `NUSLAB_GPIB_DROPBOX_ROOT` if needed. The durable local recovery root defaults to `%LOCALAPPDATA%\NUSLab\GPIBServer\recovery` and can be overridden with `NUSLAB_GPIB_RECOVERY_ROOT`. Each run is saved under `date/sample/data`; sweep plots are under `date/sample/plots`. Files are mirrored to Dropbox when available. `/v1/runs/{run_id}` reports local and Dropbox paths and sync state.

## API workflow

1. `GET /v1/devices` to discover resource names and models.
2. `POST /v1/sessions` with a sample name and a map of aliases to resources. One session can reserve up to four instruments.
3. Call typed device endpoints, grouped samples, or a voltage-list sweep. Send `POST /v1/sessions/{id}/heartbeat` at least every 30 seconds while using manual controls. The lease expires after 90 seconds; active server sweep jobs keep it alive.
4. Poll `GET /v1/jobs/{job_id}` and `GET /v1/jobs/{job_id}/readings?after=0`. A job is fully finished when `finished_at` is present. Stop a job with `POST /v1/jobs/{job_id}/cancel`.
5. `DELETE /v1/sessions/{id}` to clean up source outputs and finish the recording (SMUs off; SR830 outputs minimized).

Example session request:

```json
{
  "sample_name": "FET_16_10",
  "devices": {
    "sd": "GPIB0::12::INSTR",
    "gate": "GPIB0::16::INSTR",
    "probe": "GPIB0::15::INSTR"
  },
  "notes": {"contacts": [16, 10]}
}
```

The resource mapping above is an example from the live check on 2026-09-21; always use `/v1/devices` before an experiment. During that check the 6430 was connected to a 1 MΩ resistor and the 2002 was disconnected. These addresses do not establish a FET measurement or the current physical wiring. The server does not infer physical connections from model or GPIB address.

Typed operations, using the returned session ID and chosen alias:

```text
GET  /v1/sessions/{id}/devices/{alias}
POST /v1/sessions/{id}/devices/{alias}/configure
  {"source_mode":"voltage","nplc":0.1,"compliance":0.00001,"sense_autorange":true}
POST /v1/sessions/{id}/devices/{alias}/setpoint
  {"value":0.1}
POST /v1/sessions/{id}/devices/{alias}/output
  {"enabled":true}
POST /v1/sessions/{id}/devices/{alias}/read
  {}
POST /v1/sessions/{id}/samples
  {"devices":["sd","probe","gate"]}
POST /v1/sessions/{id}/devices/{alias}/scpi
  {"kind":"query","command":"*IDN?"}
```

For the 2002, `configure` accepts `function` (`VOLT:DC`, `VOLT:AC`, `CURR:DC`, `CURR:AC`, `RES`, `FRES`, `FREQ`, `TEMP`), optional `nplc`, and range settings where supported. A grouped sample assigns one sample ID and records an individual timestamp for each sequential GPIB read. Raw SCPI is available to a session owner and can change instrument state; typed controls query the instrument's current state instead of relying on a cached value.

## SR830 controls

Always discover before reserving; address 8 below is only the connected unit's example address. These requests use the same session/heartbeat mechanism as the Keithleys:

```text
GET /v1/devices
POST /v1/sessions
  {"sample_name":"lockin_check","devices":{"lia":"GPIB0::8::INSTR"}}
GET /v1/sessions/{id}/devices/lia
POST /v1/sessions/{id}/heartbeat
```

Device status queries actual settings. Configuration is a partial update: omitted fields retain their settings, subject to hardware coupling between sensitivity, input gain, frequency and filters. Requests are validated before any setting is written. Unknown fields and settings for another model are rejected.

Example voltage configuration (this applies excitation; choose values appropriate to the connected sample):

```text
POST /v1/sessions/{id}/devices/lia/configure
  {"reference_source":"internal","frequency_hz":17,"harmonic":1,
   "sine_amplitude_v_rms":0.01,"input_mode":"A-B","coupling":"AC",
   "sensitivity":0.001,"time_constant_s":1,"filter_slope_db_oct":24}
```

Available fields:

| Field | Values / meaning |
| --- | --- |
| `reference_source` | `internal`, `external`; include `frequency_hz` when switching from external to internal |
| `frequency_hz` | Internal-reference frequency, 0.001–102000 Hz; frequency × harmonic must be ≤102000 Hz |
| `phase_deg` | −360 through 729.99; readback reports the wrapped/rounded phase |
| `harmonic` | Integer 1–19999, subject to the frequency limit |
| `external_trigger` | `sine`, `ttl_rising`, `ttl_falling`; below 1 Hz external reference requires TTL |
| `sine_amplitude_v_rms` | 0.004–5 V RMS; hardware rounds to its 2 mV steps |
| `input_mode` | `A`, `A-B`, `current_1e6`, `current_1e8` |
| `coupling`, `grounding` | `AC` / `DC`; `float` / `ground` |
| `notch_filter` | `off`, `line`, `twice_line`, `both` |
| `sensitivity` | Discrete 1–2–5 full-scale values: 2 nV–1 V in voltage mode, 2 fA–1 µA in current mode; `current_1e8` requires ≤10 nA |
| `reserve` | `high`, `normal`, `low_noise` |
| `time_constant_s` | 1 or 3 × powers of ten, 10 µs–30000 s; explicit values above 30 s require detection frequency ≤200 Hz |
| `filter_slope_db_oct` | 6, 12, 18, 24 |
| `synchronous_filter` | Boolean selecting synchronous filtering below approximately 200 Hz; selection does not mean the filter is active above that frequency |
| `aux_outputs_v` | Mapping with keys `"1"`–`"4"` to −10.5–10.5 V; hardware rounds to 1 mV |

To select the high-gain current input, include a compatible sensitivity in the same request if necessary:

```json
{"input_mode":"current_1e8","sensitivity":1e-12,"time_constant_s":1}
```

Configuration responses contain actual settings plus an `adjustments` map of hardware-rounded or indirectly changed values. Short time constants may be increased by the instrument's gain/filter constraints. Numeric sensitivity uses the requested input mode's units, or the current mode if unchanged. This API does not silently select the other current gain to accommodate an incompatible requested sensitivity.

Auxiliary bias is a separate connector, not a DC offset on Sine Out. For example:

```text
POST /v1/sessions/{id}/devices/lia/configure
  {"aux_outputs_v":{"1":0.5}}
POST /v1/sessions/{id}/devices/lia/read
  {}
POST /v1/sessions/{id}/samples
  {"devices":["lia","sd"]}
POST /v1/sessions/{id}/devices/lia/minimize-outputs
DELETE /v1/sessions/{id}
```

The grouped example requires `sd` to be reserved in that session. Read responses expose `x`, `y`, `magnitude`, `phase_deg`, `frequency_hz`, `unit` (`V` or `A`), `input_mode`, timestamp and diagnostics. `magnitude` is the lock-in signal magnitude, **not resistance**. No sample voltage, resistance, external attenuation or excitation-current calibration is inferred. The SR830's native digital current readings are already in amperes; they are not divided by preamplifier gain again.

Reads use `SNAP? 1,2,3,4,9`. X/Y form a coherent pair; magnitude/phase are sampled approximately 10 µs apart from that pair, and reference frequency has its own update interval. The response is the current filtered result, with `settling_waited:false`; the client must allow settling after changing a source, reference or filter. Mixed-device samples are sequential, not hardware synchronized.

`status_word` and `status_flags` report `input_overload`, `filter_overload`, `output_overload`, `reference_unlocked`, `frequency_range_changed`, `time_constant_changed`, and `data_triggered`. These are latched indications since the preceding status read, not exclusively instantaneous conditions. Status/configuration reads consume these latches. `diagnostics` preserves the standard-event and error-register values and decoded command/hardware errors; power-on and front-panel activity alone are not failures. Raw queries of clear-on-read registers preserve the returned value without querying the same register a second time.

Readings also include the actual `sensitivity` (full scale in the reading's units) and `range_exceeded`. The latter independently compares X, Y and magnitude with full scale. A live resistor check found magnitude clipping above full scale without a native overload flag. If either `range_exceeded` or `overload` is true, widen the sensitivity or reduce excitation, allow settling and repeat. Native status values are preserved; the API does not silently autorange or replace magnitude with a value calculated from X/Y.

Raw native SR830 commands remain available through `/scpi`, for example `{"kind":"query","command":"*IDN?"}`. Command/hardware errors are returned in `instrument_errors` and trigger source minimization. SMU `/setpoint`, `/output` and voltage-list `/sweeps` operations are rejected for SR830. Use `/configure` for amplitudes and auxiliary voltages, and `/minimize-outputs` for cleanup. Dedicated auto functions, internal-buffer acquisition and resistance/bias-sweep workflows are outside this API version.

Command reference: [SRS SR830 manual, chapter 5](https://www.thinksrs.com/downloads/PDFs/Manuals/SR830m.pdf). Implemented with the existing PyVISA dependency; no additional runtime packages are needed.

## SMU sweeps and recording

Example hardware-timed list sweep:

```json
{
  "device": "sd",
  "voltage_points_v": [0, 0.1, 0.2, 0.1, 0],
  "nplc": 0.1,
  "source_delay_s": 0.01,
  "compliance_a": 0.00001,
  "compliance_action": "stop"
}
```

Submit to `POST /v1/sessions/{id}/sweeps`. The 2400 and 6430 use hardware-timed voltage lists. Requests are split into batches of at most 2500 points and shorter batches when `compliance_action` is `stop`, so compliance stops at a batch boundary. Output is off after each batch and when a job ends. Manual typed source controls use the instrument's current compliance setting unless the client changes it; the server imposes no additional amplitude cap.

The `.data` file is CSV with a stable column set for manual reads, grouped samples, and sweeps. `timestamp_utc` on sweep rows is the batch readback time. `instrument_time_s` is the instrument timestamp when it is valid; invalid or nonmonotonic timestamps remain in `instrument_time_raw_s` with `timestamp_invalid=true`. The `.events.jsonl` file records commands, results, errors, and lifecycle events. `.meta.json` contains instrument IDs, sample notes, job outcomes, and sync status. A sweep has linear and logarithmic IV PNG plots when valid data exists.

New recordings use schema version 3. Schema 2 appended SR830 measurement fields, individual status flags and diagnostic register values to the original columns; schema 3 additionally appends `sensitivity` and `range_exceeded`. Existing column names/order are preserved; irrelevant fields are empty on other instruments' rows. Historical recordings remain readable and are mirrored without rewriting their CSV contents.

All API failures use `{"error":{"code":"...","message":"...","details":{...}}}`. Timeouts and instrument errors cause an emergency source-cleanup attempt. Failed verification makes the device unavailable until `POST /v1/devices/recover` succeeds. That explicit recovery action sends a VISA device clear, re-identifies the device, and verifies cleanup; routine discovery does not clear the interface. A disconnected 2002 input can float; its voltage is recorded as measured rather than treated as zero.

Atomic metadata and Dropbox file replacements retry transient Windows access/sharing locks for up to 0.63 seconds. The previous destination stays intact until replacement succeeds. Persistent errors still use the normal storage-error shutdown or pending-sync path.

## Tests

Voltage lists are uploaded in groups of at most 100 values using `:SOUR:LIST:VOLT:APP` for subsequent groups, as required by both instrument manuals. This upload grouping does not interrupt a hardware sweep. The loaded list length is checked before output enable. Combined configuration requests widen compliance before widening the sense range to avoid the 6430's range/compliance ordering error.

Run `\.venv\Scripts\python.exe -m pytest GPIBServer\tests -q` from the repository root. The fake VISA tests cover startup recovery, reservations, lease expiry, both sweep models, compliance, errors, and file fallback. A conservative live smoke check was run against the three connected instruments on 2026-09-21: identification and output-off on the DUT-connected 2400, DC voltage read on the open 2002, and a 0/0.1/0 V list on the 6430 with the 1 MΩ resistor and 10 µA compliance. No DUT voltage sweep was run.

SR830 fake-VISA tests additionally cover both current gains and units, complete request validation, command ordering, rounded/automatically adjusted settings, latched diagnostics, four-device sessions, cleanup failures and recovery, and schema-1 compatibility. On 2026-09-24 the deployed API v1.1.0 passed a live integration check with SR830 serial 40423: discovery, actual configuration readback, native identity query, individual/grouped reads, recording and mirroring, explicit minimization, and session release. The existing `current_1e8` input mode and 13.033 Hz reference were retained; excitation was only minimized to 4 mV RMS and auxiliary outputs to zero. These reads test instrument/software communication, not sample resistance or calibration. The detailed local check report is `output/sr830_live_check_20260924.json`.

Subsequent user-authorized measurements with a nominal 1 MΩ resistor exercised 0.04 Hz–102 kHz and 4–500 mV RMS through the API. They verified native current scaling, exposed magnitude clipping without a native overload flag, and motivated the independent `range_exceeded` check in API v1.1.1 / recording schema 3. The final full suite passed 97 tests, including schema-1/2 preservation. Measurement scripts, raw records, validation and plots are in [the bench report](../experiments/sr830_1mohm_20260924/README.md). No biased sample measurements were performed.
