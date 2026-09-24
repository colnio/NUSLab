"""Verify deployed controls and leave the authorized resistor at minimum drive."""
import json
import time

from measure import ROOT, api


def main():
    report = {"health_before": api("GET", "/health")}
    assert report["health_before"]["active_sessions"] == 0
    devices = api("GET", "/devices")["devices"]
    device = next(d for d in devices if d["model"] == "SR830" and d["serial"] == "40423")
    assert device["available"]
    session = api("POST", "/sessions", {"sample_name": "SR830_1Mohm_final_check",
        "devices": {"lia": device["resource"]}})
    sid = session["session_id"]
    prefix = f"/sessions/{sid}/devices/lia"
    report.update(device=device, session=session)
    try:
        report["identity_query"] = api("POST", prefix+"/scpi", {"kind": "query", "command": "*IDN?"})
        report["configuration"] = api("POST", prefix+"/configure", {
            "reference_source": "internal", "frequency_hz": 13.033,
            "input_mode": "current_1e6", "sensitivity": 1e-8,
            "coupling": "DC", "time_constant_s": .3, "filter_slope_db_oct": 24,
            "sine_amplitude_v_rms": .004, "aux_outputs_v": {str(i): 0 for i in range(1,5)}})
        for _ in range(3):
            api("POST", f"/sessions/{sid}/heartbeat", {})
            time.sleep(5)
        report["read"] = api("POST", prefix+"/read", {})
        report["samples"] = api("POST", f"/sessions/{sid}/samples", {"devices": ["lia"]})
        for reading in (report["read"], report["samples"]["readings"]["lia"]):
            assert reading["unit"] == "A" and reading["input_mode"] == "current_1e6"
            assert not reading["range_exceeded"] and not reading["overload"] and not reading["reference_unlocked"]
            assert 3.2e-9 < reading["magnitude"] < 4.8e-9
    finally:
        try:
            report["cleanup"] = api("POST", prefix+"/minimize-outputs", {})
            report["final_configuration"] = api("GET", prefix)
        finally:
            report["release"] = api("DELETE", f"/sessions/{sid}")
            report["recording"] = api("GET", f"/runs/{session['run_id']}")
            (ROOT/"final_check.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    report["health_after"] = api("GET", "/health")
    assert report["health_after"]["active_sessions"] == 0
    assert report["health_after"]["status"] == "ready"
    final = report["final_configuration"]
    assert final["sine_amplitude_v_rms"] == .004 and all(v == 0 for v in final["aux_outputs_v"].values())
    assert report["recording"]["schema_version"] == 3
    assert report["recording"]["sync"]["state"] == "synced"
    (ROOT/"final_check.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Live final check passed; outputs minimized, AC remains active; server ready with zero sessions.")


if __name__ == "__main__":
    main()
