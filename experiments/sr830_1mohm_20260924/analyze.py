"""Aggregate actual REST measurements and export standalone scientific plots.

Uses the repository's existing NumPy/Matplotlib; never generates synthetic data.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
COLORS = ["#176B9B", "#DD7C22", "#168071", "#9B4B9E", "#A64A3C", "#70757A", "#323C69"]


def load_points():
    groups = defaultdict(list)
    sessions = []
    all_rows = []
    exclusions = []
    for path in sorted(ROOT.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not any(r["kind"] == "complete" for r in records):
            continue  # In-progress/failed runs are retained, never silently plotted.
        header = next(r for r in records if r["kind"] == "session")
        sessions.append({"source": path.name, "session_id": header["session"]["session_id"],
                         "protocol": header["protocol"], "instrument": header["device"]["idn"]})
        if header["protocol"] == "voltage_check":
            continue
        for item in records:
            if item["kind"] != "measurement":
                continue
            c, r = item["configuration"], item["reading"]
            assert r["unit"] == "A" and r["input_mode"] == c["input_mode"]
            assert not r["overload"] and not r["reference_unlocked"]
            assert not r["diagnostics"]["instrument_errors"]
            row = {"source": path.name, "point_index": item["point_index"],
                   "replicate": item["replicate"], "series": item["series"],
                   "timestamp": r["timestamp"], "operation": item["operation"],
                   "frequency_hz": r["frequency_hz"], "sine_v_rms": c["sine_amplitude_v_rms"],
                   "input_mode": c["input_mode"], "coupling": c["coupling"],
                   "sensitivity_a": c["sensitivity"], "time_constant_s": c["time_constant_s"],
                   "settling_s": item["settling_s"], "spacing_s": item["spacing_s"],
                   "x_a": r["x"], "y_a": r["y"], "magnitude_a": r["magnitude"],
                   "phase_native_deg": r["phase_deg"], "status_word": r["status_word"],
                   "unit": r["unit"]}
            row["range_exceeded"] = max(abs(r[k]) for k in ("x", "y", "magnitude")) > c["sensitivity"]
            all_rows.append(row)
            if row["range_exceeded"]:
                exclusions.append({"source": path.name, "point_index": item["point_index"],
                                   "replicate": item["replicate"], "reason": "signal exceeds sensitivity"})
                continue
            groups[(path.name, item["point_index"])].append(row)
    points = []
    for rows in groups.values():
        assert len(rows) == 3 and {r["replicate"] for r in rows} == {0, 1, 2}
        p = {k: rows[0][k] for k in ("source", "series", "frequency_hz", "sine_v_rms",
                                   "input_mode", "coupling", "sensitivity_a", "time_constant_s", "settling_s")}
        voltage = p["sine_v_rms"]
        # Reverse the fixed current-preamplifier sign, keeping phase changes.
        currents = np.array([-r["x_a"] - 1j * r["y_a"] for r in rows])
        current = currents.mean()
        mags = np.array([r["magnitude_a"] for r in rows])
        p.update(n=len(rows), current_rms_a=float(mags.mean()), current_sd_a=float(mags.std(ddof=1)),
                 g_magnitude_s=float(mags.mean() / voltage),
                 g_sd_s=float(mags.std(ddof=1) / voltage),
                 g_real_s=float(current.real / voltage), g_imag_s=float(current.imag / voltage),
                 phase_deg=float(np.angle(current, deg=True)),
                 pair_difference_fraction=float((abs(current) - mags.mean()) / mags.mean()))
        points.append(p)
    # Later completed repeats supersede earlier points with the same settings.
    # Original reads remain in measurements.csv and JSONL for audit.
    by_setting = {}
    for point in points:
        key = tuple(point[k] for k in ("series", "frequency_hz", "sine_v_rms", "input_mode", "coupling"))
        by_setting[key] = point
    points = list(by_setting.values())
    for filename, rows in (("measurements.csv", all_rows), ("points.csv", points)):
        if rows:
            with (ROOT / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (ROOT / "provenance.json").write_text(json.dumps(sessions, indent=2) + "\n", encoding="utf-8")
    (ROOT / "exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n", encoding="utf-8")
    return points


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
        "legend.frameon": False, "savefig.dpi": 180, "figure.facecolor": "white"})


def finish(fig, filename, footer):
    fig.text(.06, .025, footer, fontsize=9, color="#424A53", va="bottom")
    fig.subplots_adjust(left=.09, right=.95, bottom=.22 if filename == "conductance_map.png" else .17,
                        top=.77 if filename == "signal_vs_amplitude.png" else .85, hspace=.42, wspace=.3)
    fig.savefig(PLOTS / filename)
    plt.close(fig)


def cutoff(points, plateau_hz=(5, 75)):
    points = sorted(points, key=lambda p: p["frequency_hz"])
    base = np.median([p["g_magnitude_s"] for p in points if plateau_hz[0] <= p["frequency_hz"] <= plateau_hz[1]])
    threshold = base / math.sqrt(2)
    for left, right in zip(points, points[1:]):
        if left["g_magnitude_s"] > threshold >= right["g_magnitude_s"]:
            fraction = ((math.log(threshold) - math.log(left["g_magnitude_s"])) /
                        (math.log(right["g_magnitude_s"]) - math.log(left["g_magnitude_s"])))
            return {"plateau_s": float(base), "cutoff_hz": float(math.exp(math.log(left["frequency_hz"]) +
                    fraction * math.log(right["frequency_hz"] / left["frequency_hz"]))),
                    "bracket_hz": [left["frequency_hz"], right["frequency_hz"]]}
    return {"plateau_s": float(base), "cutoff_hz": None}


def main():
    style()
    PLOTS.mkdir(exist_ok=True)
    points = load_points()
    summary = {"completed_points": len(points), "readings": 3 * len(points)}
    freq = sorted([p for p in points if p["series"] == "frequency"], key=lambda p: p["frequency_hz"])
    high = sorted([p for p in points if p["series"] == "high_gain"], key=lambda p: p["frequency_hz"])
    if freq:
        fig, axes = plt.subplots(2, 1, figsize=(11.5, 8), sharex=True)
        fig.suptitle("1 MΩ resistor • frequency response", x=.09, ha="left", fontsize=20, fontweight="bold", y=.965)
        fig.text(.09, .915, "SR830 #40423  |  DC coupling  |  zero applied DC bias", color="#515B65")
        for rows, label, color, marker in ((freq, "10⁶ V/A input · 100 mV RMS", COLORS[0], "o"),
                                          (high, "10⁸ V/A input · 4 mV RMS (diagnostic)", COLORS[1], "s")):
            if not rows:
                continue
            f = np.array([p["frequency_hz"] for p in rows])
            axes[0].errorbar(f, [p["g_magnitude_s"] * 1e6 for p in rows],
                            yerr=[p["g_sd_s"] * 1e6 for p in rows], color=color, marker=marker,
                            markersize=4, linewidth=1.5, capsize=2, label=label)
            axes[1].semilogx(f, [p["phase_deg"] for p in rows], color=color, marker=marker, markersize=4)
        low_corner = cutoff(freq)
        peak = max(freq, key=lambda p: p["g_magnitude_s"])
        info = f"Peak: {peak['frequency_hz']/1000:.1f} kHz (10⁶).  −3 dB vs low-frequency plateaus: {low_corner['cutoff_hz']/1000:.1f} kHz (10⁶)"
        if high:
            info += f"; {cutoff(high)['cutoff_hz']:.0f} Hz (10⁸)"
        fig.text(.09, .875, info, fontsize=10, color="#424A53")
        axes[0].set_xscale("log")
        axes[0].axhline(1, color="#424A53", ls="--", lw=1, label="Nominal resistor: 1 µS")
        axes[0].set_ylabel("|I| / Vsource (µS)")
        axes[0].set_ylim(bottom=0)
        axes[0].legend(loc="upper left", fontsize=10)
        axes[1].set_ylabel("Current phase (degrees)\n180° input inversion removed")
        axes[1].set_xlabel("Sine frequency (Hz)")
        axes[1].axhline(0, color="#737980", lw=.7)
        finish(fig, "signal_vs_frequency.png", "Three readings per point; error bars = repeat SD (often smaller than markers), not calibration uncertainty.\nSource voltage is the programmed RMS value. High-gain test uses 1 MΩ, below the manual's recommended >100 MΩ.")
        summary["low_gain"] = low_corner
        summary["low_gain"]["peak_hz"] = peak["frequency_hz"]
        summary["low_gain"]["peak_s"] = peak["g_magnitude_s"]
        if high:
            summary["high_gain_diagnostic"] = cutoff(high)
    map_points = [p for p in points if p["series"] == "map"]
    if map_points:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
        fig.suptitle("1 MΩ resistor • amplitude response", x=.09, ha="left", fontsize=20, fontweight="bold", y=.965)
        fig.text(.09, .91, "10⁶ V/A current input  |  DC coupling  |  zero applied DC bias", color="#515B65")
        summary["amplitude_fits"] = []
        for index, f in enumerate((13.033, 1033, 10330, 38000, 70000, 102000)):
            rows = sorted([p for p in map_points if p["frequency_hz"] == f], key=lambda p: p["sine_v_rms"])
            if not rows:
                continue
            a = np.array([p["sine_v_rms"] for p in rows])
            i = np.array([p["current_rms_a"] for p in rows])
            label = f"{f:g} Hz"
            marker = ("o", "s", "^", "D", "v", "P")[index]
            axes[0].plot(a * 1e3, i * 1e9, marker+"-", ms=4, color=COLORS[index], label=label)
            reference_g = next(p["g_magnitude_s"] for p in rows if math.isclose(p["sine_v_rms"], .05))
            axes[1].semilogx(a * 1e3, (i / a / reference_g - 1) * 100, marker+"-", ms=4, color=COLORS[index], label=label)
            slope, intercept = np.polyfit(a, i, 1)
            residuals = i - (slope * a + intercept)
            summary["amplitude_fits"].append({"frequency_hz": f, "slope_s": float(slope),
                "intercept_a": float(intercept), "max_residual_a": float(abs(residuals).max()),
                "r_squared": float(1 - sum(residuals**2) / sum((i - i.mean())**2)),
                "g_min_s": float(min(i/a)), "g_max_s": float(max(i/a))})
        axes[0].plot([0, 500], [0, 500], color="#424A53", ls="--", lw=1, label="Nominal 1 MΩ")
        axes[0].set(xlabel="Sine amplitude (mV RMS)", ylabel="Measured current (nA RMS)", xlim=(0, 520))
        axes[1].axhline(0, color="#424A53", ls="--", lw=1)
        axes[1].set(xlabel="Sine amplitude (mV RMS)", ylabel="Conductance change (%)\nrelative to 50 mV at each frequency")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .88), ncol=4, fontsize=9)
        finish(fig, "signal_vs_amplitude.png", "Each marker averages three settled reads. Right: each frequency normalized to its own 50 mV value.\nRatios use programmed source voltage; amplitude-dependent source error and resistor response are not separated.")
        fs = sorted({p["frequency_hz"] for p in map_points})
        amps = sorted({p["sine_v_rms"] for p in map_points})
        grid = np.full((len(amps), len(fs)), np.nan)
        for p in map_points:
            grid[amps.index(p["sine_v_rms"]), fs.index(p["frequency_hz"])] = p["g_magnitude_s"] * 1e6
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.suptitle("1 MΩ resistor • frequency × amplitude", x=.09, ha="left", fontsize=20, fontweight="bold", y=.965)
        fig.text(.09, .91, "Measured |I| / Vsource (µS)  |  10⁶ V/A current input  |  DC coupling", color="#515B65")
        cmap = matplotlib.colormaps["viridis"].copy()
        cmap.set_bad("#E6E9EC")
        im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(fs)), [f"{f:g}" for f in fs], rotation=35, ha="right")
        ax.set_yticks(range(len(amps)), [f"{a*1000:g}" for a in amps])
        ax.set(xlabel="Frequency (Hz; discrete tested settings)", ylabel="Sine amplitude (mV RMS)")
        ax.grid(False)
        for row in range(len(amps)):
            for col in range(len(fs)):
                if not np.isfinite(grid[row,col]):
                    ax.text(col, row, "skip", ha="center", va="center", fontsize=8, color="#424A53")
                    continue
                norm = (grid[row,col] - np.nanmin(grid)) / max(np.nanmax(grid)-np.nanmin(grid), 1e-10)
                ax.text(col, row, f"{grid[row,col]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if norm < .55 else "#162D34")
        fig.colorbar(im, ax=ax, pad=.02, label="AC conductance magnitude (µS)")
        finish(fig, "conductance_map.png", "Numbers = three settled reads; no interpolation. Discrete rows/columns; gray cells skipped to preserve range headroom.\nNominal 1 MΩ = 1 µS. Values include resistor, source, input amplifier, and wiring response.")
    coupling = [p for p in points if p["series"] == "coupling"]
    if coupling:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
        fig.suptitle("1 MΩ resistor • AC input coupling cutoff", x=.09, ha="left", fontsize=20, fontweight="bold", y=.965)
        fig.text(.09, .91, "100 mV RMS excitation  |  10⁶ V/A current input  |  zero applied DC bias", color="#515B65")
        for mode, color, marker in (("DC", COLORS[0], "o"), ("AC", COLORS[1], "s")):
            rows = sorted([p for p in coupling if p["coupling"] == mode], key=lambda p: p["frequency_hz"])
            fs = [p["frequency_hz"] for p in rows]
            axes[0].semilogx(fs, [p["g_magnitude_s"] * 1e6 for p in rows], marker+"-", color=color, label=mode+" coupling")
            axes[1].semilogx(fs, [p["phase_deg"] for p in rows], marker+"-", color=color, label=mode+" coupling")
        axes[0].set(xlabel="Frequency (Hz)", ylabel="|I| / Vsource (µS)", ylim=(0, 1.12))
        axes[1].set(xlabel="Frequency (Hz)", ylabel="Current phase (degrees)\n180° input inversion removed")
        for ax in axes:
            ax.axvline(.16, color="#424A53", ls="--", lw=1, label="Manual: 0.16 Hz")
            ax.legend(fontsize=10)
        finish(fig, "input_coupling_cutoff.png", "Three settled readings per point; source-cycle and filter settling explicitly applied. AC/DC refers to the input coupling.\nThe current preamplifier remains DC coupled in both modes. Dashed line is the manual specification, not a fitted result.")
        pairs = defaultdict(dict)
        for p in coupling:
            pairs[p["frequency_hz"]][p["coupling"]] = p["g_magnitude_s"]
        ratios = sorted((f, p["AC"] / p["DC"]) for f,p in pairs.items() if {"AC", "DC"} <= p.keys())
        summary["coupling_ratios"] = [{"frequency_hz": f, "ac_over_dc": ratio} for f,ratio in ratios]
        f_values = np.array([f for f,_ in ratios])
        measured_ratio = np.array([r for _,r in ratios])
        corner_grid = np.geomspace(.02, 1, 20000)
        losses = [sum((measured_ratio - f_values/np.sqrt(f_values**2 + fc**2))**2) for fc in corner_grid]
        best_corner = int(np.argmin(losses))
        summary["ac_coupling_rc_fit_hz"] = float(corner_grid[best_corner])
        summary["ac_coupling_ratio_fit_rms_residual"] = math.sqrt(losses[best_corner]/len(ratios))
        target = 1/math.sqrt(2)
        for (fl,gl), (fr,gr) in zip(ratios, ratios[1:]):
            if gl <= target < gr:
                summary["ac_coupling_cutoff_hz"] = math.exp(math.log(fl) +
                    (math.log(target/gl)/math.log(gr/gl))*math.log(fr/fl))
                summary["ac_coupling_cutoff_bracket_hz"] = [fl,fr]
                break
    voltage_files = sorted(ROOT.glob("voltage_check_*.jsonl"))
    if voltage_files:
        records = [json.loads(line) for line in voltage_files[-1].read_text(encoding="utf-8").splitlines()]
        if any(r["kind"] == "complete" for r in records):
            grouped = defaultdict(list)
            magnitudes = defaultdict(list)
            voltage_rows = []
            for item in records:
                if item["kind"] != "measurement":
                    continue
                r,c = item["reading"],item["configuration"]
                assert r["unit"] == "V" and not r["overload"] and not r["range_exceeded"]
                grouped[c["frequency_hz"]].append(complex(r["x"],r["y"])/c["sine_amplitude_v_rms"])
                magnitudes[c["frequency_hz"]].append(r["magnitude"])
                voltage_rows.append({"source": voltage_files[-1].name, "timestamp": r["timestamp"],
                    "frequency_hz": r["frequency_hz"], "source_v_rms": c["sine_amplitude_v_rms"],
                    "x_v": r["x"], "y_v": r["y"], "magnitude_v": r["magnitude"],
                    "phase_deg": r["phase_deg"], "unit": r["unit"], "status_word": r["status_word"]})
            with (ROOT / "voltage_measurements.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle,fieldnames=list(voltage_rows[0]))
                writer.writeheader()
                writer.writerows(voltage_rows)
            rejected = [f for f,values in magnitudes.items() if (max(values)-min(values))/np.mean(values) > .005]
            for frequency in rejected:
                del grouped[frequency]
            summary["voltage_points_rejected_for_settling_hz"] = rejected
            fs = np.array(sorted(grouped))
            h = np.array([np.mean(grouped[f]) for f in fs])
            selected = fs <= 10330
            best = None
            for fc in np.geomspace(10, 1e5, 12000):
                model = 1/(1+1j*fs[selected]/fc)
                scale = np.vdot(model,h[selected]).real/np.vdot(model,model).real
                error = float(sum(abs(h[selected]-scale*model)**2))
                if best is None or error < best[0]:
                    best = (error, float(fc), float(scale))
            error,fc,scale = best
            summary["voltage_diagnostic"] = {"fitted_corner_hz": fc, "fitted_dc_transfer": scale,
                "equivalent_parallel_capacitance_pf": (1+1e6/1e7)/(2*math.pi*1e6*fc)*1e12,
                "fit_complex_rms_residual": math.sqrt(error/sum(selected)),
                "fit_maximum_frequency_hz": float(max(fs[selected]))}
            fig,axes = plt.subplots(1,2,figsize=(12,6.5))
            fig.suptitle("Same wiring • voltage-input diagnostic", x=.09, ha="left", fontsize=20, fontweight="bold", y=.965)
            fig.text(.09,.91,"A voltage input  |  100 mV RMS source  |  DC coupling  |  no rewiring",color="#515B65")
            axes[0].loglog(fs,abs(h),"o",color=COLORS[0],label="Measured voltage ratio")
            axes[1].semilogx(fs,np.angle(h,deg=True),"o",color=COLORS[0])
            fm = np.geomspace(min(fs),max(fs),400)
            hm = scale/(1+1j*fm/fc)
            axes[0].loglog(fm,abs(hm),"--",color="#424A53",label=f"RC model: corner {fc:.0f} Hz")
            axes[1].semilogx(fm,np.angle(hm,deg=True),"--",color="#424A53")
            axes[0].set(xlabel="Frequency (Hz)",ylabel="|Vinput / Vsource|")
            axes[1].set(xlabel="Frequency (Hz)",ylabel="Voltage phase (degrees)")
            axes[0].legend(fontsize=10)
            finish(fig,"voltage_input_diagnostic.png","Model uses nominal 1 MΩ series and 10 MΩ input resistance; capacitance represents input plus wiring.\nFit uses complex data through 10.33 kHz. This diagnostic constrains loading; it does not identify a specific physical capacitor.")
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
