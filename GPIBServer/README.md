# NUSLab GPIB server

Local REST control for the Keithley 2400, 6430, and 2002. The service uses NI-VISA through PyVISA and listens only on `127.0.0.1:8765`. Interactive API documentation is available at `http://127.0.0.1:8765/docs` after startup.

## Install and start

From the repository root, run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\GPIBServer\install_autostart.ps1` once while signed in as the lab user. This machine blocks direct PowerShell script execution. The installer adds the pinned Python dependencies to the repository's `.venv`, creates a single-instance Task Scheduler job, and starts the server. Use the same PowerShell command with `start_server.ps1` to start it manually. The task restarts after a failure and starts again at sign-in.

Use the existing GUI programs only when the server is stopped. The server reserves each GPIB resource for one session at a time; another process cannot obey that reservation. On startup, the server identifies instruments with `*IDN?`, then aborts and turns off recognized SMUs. Discovery itself does not clear or reset instruments. A device that cannot be identified or shut down appears in `/v1/health` or `/v1/devices` as an error and is unavailable for sessions.

The default Dropbox root is:

```text
C:\Users\MeasurmentStand\NUS Dropbox\Ozyilmaz Group\1_MAC\1_Projects\Iurii\ResistiveSwitches\VacuumProbeData
```

Override it with `NUSLAB_GPIB_DROPBOX_ROOT` if needed. The durable local recovery root defaults to `%LOCALAPPDATA%\NUSLab\GPIBServer\recovery` and can be overridden with `NUSLAB_GPIB_RECOVERY_ROOT`. Each run is saved under `date/sample/data`; sweep plots are under `date/sample/plots`. Files are mirrored to Dropbox when available. `/v1/runs/{run_id}` reports local and Dropbox paths and sync state.

## API workflow

1. `GET /v1/devices` to discover resource names and models.
2. `POST /v1/sessions` with a sample name and a map of aliases to resources. One session can reserve all three instruments.
3. Call typed device endpoints, grouped samples, or a voltage-list sweep. Send `POST /v1/sessions/{id}/heartbeat` at least every 30 seconds while using manual controls. The lease expires after 90 seconds; active server sweep jobs keep it alive.
4. Poll `GET /v1/jobs/{job_id}` and `GET /v1/jobs/{job_id}/readings?after=0`. A job is fully finished when `finished_at` is present. Stop a job with `POST /v1/jobs/{job_id}/cancel`.
5. `DELETE /v1/sessions/{id}` to turn off sources and finish the recording.

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

The resource mapping above is an example from the live check on 2026-09-21; always use `/v1/devices` before an experiment. The current physical wiring has the 6430 on a 1 MΩ resistor and the 2002 disconnected, so it is **not** wired as a FET measurement. The server does not infer physical connections from model or GPIB address.

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

All API failures use `{"error":{"code":"...","message":"...","details":{...}}}`. Timeouts and instrument errors cause an emergency output-off attempt. If output-off verification fails, the device is marked unavailable until `POST /v1/devices/recover` succeeds. That explicit recovery action sends a GPIB interface clear, re-identifies the device, and verifies output off; routine discovery does not clear the interface. A disconnected 2002 input can float; its voltage is recorded as measured rather than treated as zero.

Atomic metadata and Dropbox file replacements retry transient Windows access/sharing locks for up to 0.63 seconds. The previous destination stays intact until replacement succeeds. Persistent errors still use the normal storage-error shutdown or pending-sync path.

## Tests

Run `\.venv\Scripts\python.exe -m pytest GPIBServer\tests -q` from the repository root. The fake VISA tests cover startup recovery, reservations, lease expiry, both sweep models, compliance, errors, and file fallback. A conservative live smoke check was run against the three connected instruments on 2026-09-21: identification and output-off on the DUT-connected 2400, DC voltage read on the open 2002, and a 0/0.1/0 V list on the 6430 with the 1 MΩ resistor and 10 µA compliance. No DUT voltage sweep was run.
