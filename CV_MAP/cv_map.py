from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import zhinst.core as zi


@dataclass
class SavedState:
    imps_enable: int
    imps_model: int
    bias_enable: int
    bias_value: float
    aux_outputselect: int
    aux_preoffset: float
    aux_scale: float
    aux_offset: float
    aux_limitlower: float
    aux_limitupper: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2D C(Vg, Bias) sweep for MFIA via LabOne Data Server"
    )
    p.add_argument("--host", default="192.168.121.162")
    p.add_argument("--port", type=int, default=8004)
    p.add_argument("--device", default="", help="Device ID like DEV7784. If empty, first MFIA found is used.")
    p.add_argument("--imps", type=int, default=0, help="Impedance module index (default 0)")
    p.add_argument("--aux", type=int, default=0, help="Aux output channel index for Vg (Aux1=0)")
    p.add_argument("--bias-start", type=float, default=0.0)
    p.add_argument("--bias-stop", type=float, default=0.0)
    p.add_argument("--bias-steps", type=int, default=1)
    p.add_argument("--vg-start", type=float, default=0.0)
    p.add_argument("--vg-stop", type=float, default=0.0)
    p.add_argument("--vg-steps", type=int, default=1)
    p.add_argument("--freq", type=float, default=None, help="Oscillator frequency in Hz (if set).")
    p.add_argument("--settle", type=float, default=0.1, help="Settling time (s) after each step")
    p.add_argument("--ramp-step", type=float, default=0.05, help="Ramp step size in volts")
    p.add_argument("--model", type=int, default=0, help="Impedance model (0=Rp||Cp). See /imps/0/model options.")
    p.add_argument("--outfile", default="cv_map.csv")
    return p.parse_args()


def linspace(start: float, stop: float, steps: int) -> List[float]:
    if steps <= 1:
        return [float(start)]
    return list(np.linspace(start, stop, steps))


def ramp_set_double(daq: zi.ziDAQServer, path: str, target: float, step: float, wait: float) -> None:
    try:
        current = daq.getDouble(path)
    except Exception:
        # if not readable as double, just set directly
        daq.setDouble(path, target)
        time.sleep(wait)
        return

    if math.isclose(current, target, rel_tol=0, abs_tol=1e-9):
        return

    delta = target - current
    n_steps = max(1, int(abs(delta) / max(step, 1e-6)))
    for v in np.linspace(current, target, n_steps + 1)[1:]:
        daq.setDouble(path, float(v))
        time.sleep(wait)


def find_mfia_device(host: str, port: int, preferred: str | None) -> Tuple[str, str]:
    disc = zi.ziDiscovery()
    devices = disc.findAll()
    if not devices:
        raise RuntimeError("No devices found by discovery.")

    if preferred:
        dev = preferred.upper()
        info = disc.get(dev)
        if info.get("devicetype") != "MFIA":
            raise RuntimeError(f"Device {dev} is not an MFIA: {info}")
        return dev, info.get("connected", "")

    for dev in devices:
        info = disc.get(dev)
        if info.get("devicetype") == "MFIA":
            return dev, info.get("connected", "")

    raise RuntimeError("No MFIA device found.")


def save_state(daq: zi.ziDAQServer, dev: str, imps: int, aux: int) -> SavedState:
    base = f"/{dev}/imps/{imps}"
    auxbase = f"/{dev}/auxouts/{aux}"
    return SavedState(
        imps_enable=daq.getInt(f"{base}/enable"),
        imps_model=daq.getInt(f"{base}/model"),
        bias_enable=daq.getInt(f"{base}/bias/enable"),
        bias_value=daq.getDouble(f"{base}/bias/value"),
        aux_outputselect=daq.getInt(f"{auxbase}/outputselect"),
        aux_preoffset=daq.getDouble(f"{auxbase}/preoffset"),
        aux_scale=daq.getDouble(f"{auxbase}/scale"),
        aux_offset=daq.getDouble(f"{auxbase}/offset"),
        aux_limitlower=daq.getDouble(f"{auxbase}/limitlower"),
        aux_limitupper=daq.getDouble(f"{auxbase}/limitupper"),
    )


def restore_state(daq: zi.ziDAQServer, dev: str, imps: int, aux: int, s: SavedState) -> None:
    base = f"/{dev}/imps/{imps}"
    auxbase = f"/{dev}/auxouts/{aux}"

    daq.setInt(f"{base}/enable", s.imps_enable)
    daq.setInt(f"{base}/model", s.imps_model)
    daq.setInt(f"{base}/bias/enable", s.bias_enable)
    daq.setDouble(f"{base}/bias/value", s.bias_value)

    daq.setInt(f"{auxbase}/outputselect", s.aux_outputselect)
    daq.setDouble(f"{auxbase}/preoffset", s.aux_preoffset)
    daq.setDouble(f"{auxbase}/scale", s.aux_scale)
    daq.setDouble(f"{auxbase}/offset", s.aux_offset)
    daq.setDouble(f"{auxbase}/limitlower", s.aux_limitlower)
    daq.setDouble(f"{auxbase}/limitupper", s.aux_limitupper)


def get_latest_sample(daq: zi.ziDAQServer, path: str, timeout: float = 1.0) -> Dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = daq.poll(0.1, 10, 0, True)
        if path in data and data[path]:
            sample = data[path]
            # reduce arrays to last value
            out = {}
            for k, v in sample.items():
                try:
                    out[k] = v[-1]
                except Exception:
                    out[k] = v
            return out
    raise TimeoutError(f"No data received for {path} within {timeout}s")


def main() -> int:
    args = parse_args()

    host = args.host
    port = args.port

    daq = zi.ziDAQServer(host, port, 6)

    dev, interface = find_mfia_device(host, port, args.device if args.device else None)
    dev = dev.lower()
    if not interface:
        interface = "PCIe"
    daq.connectDevice(dev, interface)

    imps = args.imps
    aux = args.aux

    base = f"/{dev}/imps/{imps}"
    auxbase = f"/{dev}/auxouts/{aux}"

    # limits and safety checks
    bias_values = linspace(args.bias_start, args.bias_stop, args.bias_steps)
    vg_values = linspace(args.vg_start, args.vg_stop, args.vg_steps)

    if any(abs(v) > 10.0 for v in vg_values):
        raise ValueError("Vg values exceed ?10 V (aux output limit)")

    mode = daq.getInt(f"{base}/mode")
    if mode == 0:  # 4-terminal
        if any(abs(v) > 3.0 for v in bias_values):
            raise ValueError("Bias exceeds ?3 V in 4-terminal mode")
    else:
        if any(abs(v) > 10.0 for v in bias_values):
            raise ValueError("Bias exceeds ?10 V in 2-terminal mode")

    state = save_state(daq, dev, imps, aux)

    path_sample = f"{base}/sample"

    try:
        # Configure
        daq.setInt(f"{base}/model", args.model)
        daq.setInt(f"{base}/enable", 1)
        daq.setInt(f"{base}/bias/enable", 1)
        if args.freq is not None:
            daq.setDouble(f"{base}/freq", float(args.freq))

        # Aux output manual
        daq.setInt(f"{auxbase}/outputselect", -1)  # manual
        daq.setDouble(f"{auxbase}/preoffset", 0.0)
        daq.setDouble(f"{auxbase}/scale", 1.0)
        daq.setDouble(f"{auxbase}/limitlower", min(-10.0, min(vg_values) - 0.1))
        daq.setDouble(f"{auxbase}/limitupper", max(10.0, max(vg_values) + 0.1))

        # Subscribe to sample stream
        daq.subscribe(path_sample)

        results: List[Dict] = []

        # Initial settle
        time.sleep(args.settle)

        for vg in vg_values:
            ramp_set_double(daq, f"{auxbase}/offset", vg, args.ramp_step, 0.02)
            time.sleep(args.settle)

            for bias in bias_values:
                ramp_set_double(daq, f"{base}/bias/value", bias, args.ramp_step, 0.02)
                time.sleep(args.settle)

                sample = get_latest_sample(daq, path_sample, timeout=max(1.0, args.settle * 5))
                row = {
                    "bias_V": bias,
                    "vg_V": vg,
                    "param0": sample.get("param0"),
                    "param1": sample.get("param1"),
                    "z": sample.get("z"),
                    "frequency": sample.get("frequency"),
                    "drive": sample.get("drive"),
                }
                results.append(row)
                print(f"bias={bias:+.4f} V, vg={vg:+.4f} V, param1(C?)={row['param1']}")

        # Save CSV
        with open(args.outfile, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["bias_V", "vg_V", "param0", "param1", "z", "frequency", "drive"],
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"Saved {len(results)} rows to {args.outfile}")

    finally:
        try:
            daq.unsubscribe(path_sample)
        except Exception:
            pass
        restore_state(daq, dev, imps, aux, state)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
