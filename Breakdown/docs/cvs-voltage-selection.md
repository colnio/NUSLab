# Choosing a constant-voltage stress level

This is the reasoning the program's CVS dialog is built around. The same text is
available in the app under **Help**, and in condensed form inside the stress
voltage prompt.

## What you are trading off

A CVS run holds one voltage until the dielectric fails, giving a time-to-breakdown
`t_BD`. The stress level decides whether that takes a useful amount of time:

| Stress level | What happens | Is it useful? |
|---|---|---|
| Near `V_BD` | Fails during the pre-ramp | No — you measured the ramp, not the hold |
| 0.80–0.92 × `V_BD` | Fails in ~10 s to ~10⁴ s | Yes |
| Below ~0.7 × `V_BD` | May outlive the session | Rarely |

Aim for `t_BD` between roughly **10 s and 10,000 s**.

## The rule

```
V_CVS = k × median(V_BD)
```

where `V_BD` comes from RVS runs on **sister devices of the same sample and the
same crosspoint size**, and `k` is between 0.80 and 0.92.

Two details matter:

- **Median, not mean.** One anomalous device — a poor probe landing, a particle —
  would otherwise drag the stress level for the whole campaign. The program
  offers the mean as an option but defaults to the median.
- **Same size.** Breakdown statistics scale with area, so pooling sizes is not
  meaningful. This is why the device index in this program is scoped per
  crosspoint size: the statistics stay separated automatically.

The program keeps a running median as RVS devices accumulate and shows its
reasoning in the dialog, e.g.

```
0.85 x median(V_BD) over 4 RVS devices at 5 um = 0.85 x 4.2 V = 3.57 V
```

## Why one stress level is not enough

A single level gives you a number. Three or more give you a *model* you can
extrapolate to operating field. Vary `k` across the batch — e.g. 0.90, 0.85,
0.80 — so the devices span a range of fields, then fit all three standard
lifetime models and quote the most conservative extrapolation:

| Model | Relationship | Physical picture |
|---|---|---|
| **E-model** (thermochemical) | `ln t_BD ∝ −γE` | Field-driven bond breakage; usually conservative at low field |
| **1/E-model** (anode hole injection) | `ln t_BD ∝ G/E` | Fowler–Nordheim injection; fits high-field data |
| **Power law** | `t_BD ∝ V^−n` | Common for high-k and thin films; `n` typically 40+ |

They agree where you measured and diverge badly where you extrapolate, which is
exactly the region you care about. Quoting the most conservative of the three is
the honest choice.

## How many devices

Weibull statistics need **5–10 devices per stress level**. The shape parameter
β is not just a goodness measure — it is what lets you area-scale between
crosspoint sizes:

```
t63(A₁) / t63(A₂) = (A₂ / A₁)^(1/β)
```

## Things that will bite you

**Ramp rate.** `V_BD` from an RVS depends on how fast you ramped — a faster ramp
gives a higher `V_BD`, because the device spends less time accumulating damage on
the way up. The program measures and records the *achieved* rate (not just the
requested one) in `achieved_rate_Vps`, and warns in the CVS dialog if the RVS
devices being pooled were ramped at rates differing by more than 2×.

**The approach to the stress level.** Charge injected while ramping up to
`V_CVS` is indistinguishable from charge injected during the hold. If the
approach is slow, you have pre-stressed the device and `t_BD` comes out short.
This is why `pre_ramp_rate_Vps` defaults to 100 V/s — much faster than any RVS
rate you would use — and why `t_BD` is clocked from the moment the stress level
is reached rather than from the start of the run.

**Breakdown threshold vs compliance.** The compliance limit is hardware
protection and should sit well above the level that means "this device has
failed". Detecting at `i_bd_A` *below* compliance cuts the stress before the
device is driven at the full protection current, and makes the failure criterion
an explicit recorded parameter rather than a side effect of how protection was
set. The program refuses to start if `i_bd_A >= compliance_A`, because breakdown
could then never trigger.

## What the program records

Every device contributes one row to `<sample>_summary.csv`, which is the table
to run the analysis from:

`V_BD_V`, `E_BD_MV_per_cm`, `t_BD_s`, `V_stress_V`, `ramp_rate_Vps`,
`achieved_rate_Vps`, `area_um2`, `thickness_nm`, `C_at_fref_F`,
`i_threshold_A`, `compliance_A`, `bd_detected`, `termination_reason`.

`E_BD` is only filled in when a dielectric thickness was entered.
