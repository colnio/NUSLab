"""Independently cross-check experiment JSONL against server CSV and mirrors."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    results = []
    for path in sorted(ROOT.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not any(r["kind"] == "complete" for r in records):
            continue
        meta = next(r["result"] for r in records if r["kind"] == "recording")
        cleanup = next(r["result"] for r in records if r["kind"] == "cleanup")
        assert cleanup["status"] == "outputs_minimized"
        assert cleanup["sine_amplitude_v_rms"] == .004
        assert all(v == 0 for v in cleanup["aux_outputs_v"].values())
        assert cleanup["sine_output_active"] is True
        assert meta["status"] == "closed" and meta["sync"]["state"] == "synced"
        local = Path(meta["local_root"]) / meta["files"]["data"]
        mirror = Path(meta["dropbox_root"]) / meta["files"]["data"]
        data = local.read_bytes()
        assert mirror.read_bytes() == data
        with local.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == meta["measurement_count"]
        by_timestamp = {row["timestamp_utc"]: row for row in rows}
        checked = 0
        flags = 0
        magnitudes_by_point = defaultdict(list)
        for item in records:
            if item["kind"] not in {"measurement", "settling_check"}:
                continue
            reading = item["reading"]
            row = by_timestamp[reading["timestamp"]]
            for field in ("x", "y", "magnitude", "phase_deg", "frequency_hz"):
                assert float(row[field]) == reading[field], (path.name, field)
            assert row["unit"] == reading["unit"] and row["input_mode"] == reading["input_mode"]
            assert int(row["status_word"]) == reading["status_word"]
            if meta["schema_version"] >= 3:
                assert float(row["sensitivity"]) == reading["sensitivity"]
                assert (row["range_exceeded"] == "True") == reading["range_exceeded"]
            if item["kind"] == "measurement":
                magnitudes_by_point[item["point_index"]].append(reading["magnitude"])
                c = item["configuration"]
                assert not reading["overload"] and not reading["reference_unlocked"]
                assert reading["diagnostics"]["instrument_errors"] == []
                assert math.isclose(reading["frequency_hz"], c["frequency_hz"], rel_tol=1e-6)
                assert item["settling_s"] >= 15*c["time_constant_s"]
                exceeded = max(abs(reading[k]) for k in ("x", "y", "magnitude")) > c["sensitivity"]
                flags += int(exceeded)
                if meta["schema_version"] >= 3:
                    assert reading["range_exceeded"] == exceeded
            checked += 1
        assert checked == len(rows), "Every server row should be traceable to a client event"
        unstable = [index for index, values in magnitudes_by_point.items()
                    if (max(values)-min(values))/statistics.mean(values) > .005]
        results.append({"source": path.name, "run_id": meta["run_id"],
                        "schema_version": meta["schema_version"], "server_rows_verified": checked,
                        "replicate_readings_outside_range": flags,
                        "points_with_more_than_half_percent_replicate_spread": unstable,
                        "csv_sha256": hashlib.sha256(data).hexdigest(),
                        "mirror_byte_identical": True, "cleanup_verified": True})
    (ROOT / "verification.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
