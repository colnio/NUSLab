"""Reproducible SR830 / nominal 1 Mohm bench check through the REST API.

Requires the user-authorized resistor wiring: Sine Out -> 1 Mohm -> A/I.
All amplitudes are RMS, no DC bias. Does not use PyVISA directly.
Use --protocol pilot first and inspect results before other protocols.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8765/v1"


def api(method, path, body=None):
    request = urllib.request.Request(BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {path}: {exc.code} {exc.read().decode()}") from exc


def protocol_points(name):
    if name == "pilot":
        return [("pilot", "current_1e6", "DC", 13.033, a) for a in (.004, .01, .02)]
    if name == "frequency":
        frequencies = [.1, .2, .5, 1, 2, 5, 13.033, 31, 73, 173, 333, 733,
                       1033, 1733, 3333, 7333, 10330, 17330, 23330, 33330,
                       43330, 53330, 63330, 70000, 80000, 90000, 102000]
        return [(name, "current_1e6", "DC", f, .1) for f in frequencies]
    if name == "map":
        frequencies = [.2, 1, 13.033, 73, 333, 1033, 3333, 10330, 33330, 70000, 102000]
        amplitudes = [.004, .01, .02, .05, .1, .2, .5]
        return [(name, "current_1e6", "DC", f, a) for f in frequencies for a in amplitudes]
    if name == "frequency_refine":
        return [("frequency", "current_1e6", "DC", f, .1) for f in
                (25000, 28000, 30000, 33330, 36000, 38000, 40000, 42000, 43330, 46000, 49000, 57000, 60000)]
    if name == "map_refine":
        return [("map", "current_1e6", "DC", 38000, a) for a in (.004, .01, .02, .05, .1, .2, .5)]
    if name == "high_gain":
        return [(name, "current_1e8", "DC", f, .004) for f in
                (13.033, 73, 173, 333, 533, 700, 933, 1333, 2333, 3333, 7333, 10330)]
    if name == "voltage_check":
        return [(name, "A", "DC", f, .1) for f in
                (13.033, 73, 173, 333, 733, 1033, 3333, 10330, 33330, 70000, 102000)]
    if name == "coupling":
        return [(name, "current_1e6", coupling, f, .1) for f in
                (.04, .08, .16, .32, .64, 1.28, 3.1, 13.033)
                for coupling in ("DC", "AC")]
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("pilot", "frequency", "frequency_refine", "map", "map_refine", "high_gain", "coupling", "voltage_check"), required=True)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / f"{args.protocol}_{stamp}.jsonl"
    session = None
    last_heartbeat = 0
    handle = path.open("x", encoding="utf-8")

    def log(kind, **values):
        handle.write(json.dumps({"kind": kind, "logged_at": datetime.now(timezone.utc).isoformat(), **values}) + "\n")
        handle.flush()

    def heartbeat():
        nonlocal last_heartbeat
        if time.monotonic() - last_heartbeat > 15:
            api("POST", f"/sessions/{session['session_id']}/heartbeat", {})
            last_heartbeat = time.monotonic()

    def wait(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            heartbeat()
            time.sleep(min(1, max(0, until - time.monotonic())))

    try:
        devices = api("GET", "/devices")["devices"]
        matches = [d for d in devices if d["model"] == "SR830" and d["serial"] == "40423"]
        if len(matches) != 1 or not matches[0]["available"]:
            raise RuntimeError("Expected available SR830 serial 40423")
        device = matches[0]
        session = api("POST", "/sessions", {"sample_name": f"SR830_1Mohm_{args.protocol}",
            "devices": {"lia": device["resource"]},
            "notes": {"wiring": "Sine Out through nominal 1 Mohm resistor to A/I", "dc_bias_v": 0,
                      "protocol": args.protocol, "resistor_calibration": "nominal only"}})
        sid = session["session_id"]
        prefix = f"/sessions/{sid}/devices/lia"
        log("session", session=session, device=device, protocol=args.protocol)
        previous_configuration = api("GET", prefix)
        log("initial_configuration", configuration=previous_configuration)
        for channel in (1, 2, 3):
            log("offset_expand", channel=channel,
                response=api("POST", prefix + "/scpi", {"kind": "query", "command": f"OEXP? {channel}"}))
        points = protocol_points(args.protocol)
        previous_g = {}
        for index, (series, mode, coupling, frequency, amplitude) in enumerate(points):
            heartbeat()
            estimated_g = previous_g.get(frequency, 1e-6)
            if mode != "A" and amplitude * estimated_g > .8e-6:
                log("point_skipped", point_index=index, series=series, frequency_hz=frequency,
                    sine_amplitude_v_rms=amplitude, reason="Previous lower-amplitude measurement predicts >80% of maximum 1 uA range")
                print(f"{args.protocol} {index+1}/{len(points)}: skipped {frequency:g} Hz / {amplitude:g} V (range headroom)", flush=True)
                continue
            tc = .1 if frequency >= 3 else .3
            # Full-scale range accommodates a nominal resistor current. At high
            # frequencies parasitics can matter; flag and abort overloads.
            ranges = [v * 10.0 ** e for e in range(-9, -5) for v in (1, 2, 5)]
            sensitivity = (.2 if mode == "A" else 1e-8 if mode == "current_1e8" else
                           min(1e-6, next(v for v in ranges if v >= 1.8 * amplitude * estimated_g)))
            if args.protocol == "frequency_refine":
                sensitivity = 1e-6
            requested = {"reference_source": "internal", "frequency_hz": frequency,
                "harmonic": 1, "phase_deg": 0, "sine_amplitude_v_rms": amplitude,
                "input_mode": mode, "coupling": coupling, "grounding": "float",
                "notch_filter": "off", "sensitivity": sensitivity, "reserve": "low_noise",
                "time_constant_s": tc, "filter_slope_db_oct": 24, "synchronous_filter": True,
                "aux_outputs_v": {str(i): 0 for i in range(1, 5)}}
            actual = api("POST", prefix + "/configure", requested)
            if actual["input_mode"] != mode or actual["unit"] != ("V" if mode == "A" else "A"):
                raise RuntimeError(f"Unexpected mode/units: {actual}")
            # 15 TC leaves <0.03% residual for four identical RC stages.
            # Allow >=3 source cycles including synchronous-filter settling.
            settling = max(15 * actual["time_constant_s"], 3 / actual["frequency_hz"], 5)
            if mode != previous_configuration["input_mode"]:
                settling = max(settling, 15)
            if coupling == "AC":
                settling = max(settling, 15)  # AC input HP transient (~1 s RC)
            log("point_configuration", point_index=index, series=series,
                requested=requested, configuration=actual, settling_s=settling)
            wait(settling)
            transient = api("POST", prefix + "/read", {})
            log("settling_check", point_index=index, reading=transient)
            while max(abs(transient[k]) for k in ("x", "y", "magnitude")) > .8 * actual["sensitivity"]:
                wider = next((v for v in ranges if actual["sensitivity"] * 1.01 < v <= 1e-6), None)
                if wider is None or mode == "current_1e8":
                    raise RuntimeError("Measured signal has insufficient full-scale headroom")
                actual = api("POST", prefix + "/configure", {"sensitivity": wider})
                settling = max(15 * actual["time_constant_s"], 3 / actual["frequency_hz"], 5)
                log("autorange", point_index=index, configuration=actual, settling_s=settling)
                wait(settling)
                transient = api("POST", prefix + "/read", {})
                log("settling_check", point_index=index, reading=transient)
            readings = []
            spacing = max(3 * actual["time_constant_s"], 1 / actual["frequency_hz"], .3)
            for replicate in range(3):
                wait(spacing)
                if replicate == 1:
                    response = api("POST", f"/sessions/{sid}/samples", {"devices": ["lia"]})
                    reading = response["readings"]["lia"]
                    operation = "samples"
                else:
                    reading = api("POST", prefix + "/read", {})
                    operation = "read"
                log("measurement", point_index=index, replicate=replicate, series=series,
                    operation=operation, configuration=actual, settling_s=settling,
                    spacing_s=spacing, reading=reading)
                if reading["overload"] or reading["reference_unlocked"]:
                    raise RuntimeError(f"Persistent overload/unlocked at {frequency} Hz, {amplitude} V: {reading}")
                if max(abs(reading[k]) for k in ("x", "y", "magnitude")) > actual["sensitivity"]:
                    raise RuntimeError("Signal exceeded full scale despite status register; widen range and repeat")
                readings.append(reading)
            conductance = statistics.mean(r["magnitude"] for r in readings) / actual["sine_amplitude_v_rms"]
            spread = (max(r["magnitude"] for r in readings) - min(r["magnitude"] for r in readings)) / statistics.mean(r["magnitude"] for r in readings)
            if spread > .005:
                raise RuntimeError(f"Replicate magnitude spread {spread:.2%} exceeds 0.5%; allow longer settling and repeat")
            previous_g[frequency] = conductance
            previous_configuration = actual
            ratio_label = f"|Vinput/Vsource|={conductance:.6f}" if mode == "A" else f"|G|={conductance*1e6:.6f} uS"
            circular_phase = math.degrees(math.atan2(sum(math.sin(math.radians(r["phase_deg"])) for r in readings),
                                                     sum(math.cos(math.radians(r["phase_deg"])) for r in readings)))
            print(f"{args.protocol} {index+1}/{len(points)}: {frequency:g} Hz, {amplitude:g} Vrms, {ratio_label}, phase={circular_phase:.3f} deg", flush=True)
            if args.protocol == "pilot" and not .5e-6 < conductance < 1.5e-6:
                raise RuntimeError("Pilot current inconsistent with nominal 1 Mohm; inspect wiring/scaling")
        log("complete", points=len(points))
    except BaseException as exc:
        log("failure", error=str(exc))
        raise
    finally:
        if session:
            try:
                log("cleanup", result=api("POST", f"/sessions/{session['session_id']}/devices/lia/minimize-outputs", {}))
            finally:
                log("release", result=api("DELETE", f"/sessions/{session['session_id']}"))
                log("recording", result=api("GET", f"/runs/{session['run_id']}"))
        handle.close()
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
