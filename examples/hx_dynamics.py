"""Transient response of the moving-bed particle-to-sCO2 heat exchanger, against Fang et al. (2022).

Usage (from the repository root)::

    .venv/bin/python examples/hx_dynamics.py                 # about 5 min on 4 processes
    .venv/bin/python examples/hx_dynamics.py --workers 1     # serial
    .venv/bin/python examples/hx_dynamics.py --skip-sensitivity

Runs :class:`pcsp.models.hx.dynamic.MovingBedHXDynamic` on the base case of
Fang, Chen & Shi, Energy 256 (2022) 124597 (:func:`~pcsp.models.hx.dynamic.fang2022_params`),
then the paper's three single-disturbance step families (six steps each: particle
inlet temperature, sCO2 inlet temperature, sCO2 inlet velocity), computes 95 %
response times with this script's own code and compares them with the published
ones. Then the open-loop combined disturbances of their Table 2, a sensitivity
study over the inputs the paper does not print, and a grid study.

Response time definitions (``Delta t`` from the step, :math:`T^* = (T(t) - T_0)/(T_\\infty - T_0)`,
:math:`T_\\infty` from a steady solve at the new inputs):

* ``t95_cross``: first time :math:`T^* \\ge 0.95` (linear interpolation between samples);
* ``t95_settle``: last time :math:`|T^* - 1| > 0.05`;
* ``t95_fit``: :math:`A + B \\ln 20` of the fit :math:`T^* = 1 - \\exp(-(t - A)/B)`
  (their eq. 8) pooled over the six steps of a family, which is how the published values
  follow from their fitted curves (e.g. 342.1 + 3 x 209.3 = 970 s).

Writes ``results.json``, ``results_table.md`` and ``figures/*.png`` under
``--output-dir`` (default ``outputs/hx_dynamics``), after each part
(``parts_done`` in ``results.json``), so an interrupted run keeps what finished.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from pcsp.models.hx.closures import BauxiteFitBed, StagnantBed, YagiKuniiBed  # noqa: E402
from pcsp.models.hx.conjugate import solve_conjugate  # noqa: E402
from pcsp.models.hx.dynamic import (  # noqa: E402
    FANG2022_BASE,
    FANG2022_FITS,
    FANG2022_RESPONSE_TIMES,
    FANG2022_STEPS,
    FANG2022_TABLE2,
    MovingBedHXDynamic,
    ScaledClosure,
    fang2022_co2_mass_flow,
    fang2022_params,
    simulate_response,
    steady_state,
)

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
FAMILIES = ("T_particle_in", "T_co2_in", "u_co2_in")
FAMILY_LABEL = {
    "T_particle_in": "particle inlet temperature step",
    "T_co2_in": "sCO2 inlet temperature step",
    "u_co2_in": "sCO2 inlet velocity step",
}
OUTPUTS = ("T_co2_out", "T_particle_out")
T_END = 3000.0
DT_SAMPLE = 5.0

# Read by eye from the published figures (Fang et al. 2022, Figs. 4, 6, 8 and 9);
# uncertainty about +-3 K. Keys are the disturbed input values (base value included).
FANG2022_FIG_STEADY = {
    "T_particle_in": {
        "x": (988.0, 1008.0, 1028.0, 1048.0, 1068.0, 1088.0, 1108.0),
        "T_co2_out": (928.0, 941.0, 954.0, 965.0, 979.0, 992.0, 1005.0),
        "T_particle_out": (843.0, 845.0, 847.0, 849.0, 851.0, 853.0, 855.0),
    },
    "T_co2_in": {
        "x": (763.0, 783.0, 803.0, 823.0, 843.0, 863.0, 883.0),
        "T_co2_out": (934.0, 945.0, 957.0, 966.0, 976.0, 986.0, 995.0),
        "T_particle_out": (795.0, 812.0, 831.0, 850.0, 868.0, 885.0, 903.0),
    },
    "u_co2_in": {
        "x": (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
        "T_co2_out": (1004.0, 990.0, 977.0, 966.0, 957.0, 949.0, 942.0),
        "T_particle_out": (866.0, 859.0, 853.0, 850.0, 847.0, 844.0, 842.0),
    },
}
FANG2022_FIG9_FINAL = {
    "case 1": (980.0, 843.0),
    "case 2": (1009.0, 887.0),
    "case 3": (1027.0, 857.0),
    "case 4": (1033.0, 894.0),
    "case 5": (929.0, 805.0),
    "case 6": (945.0, 805.0),
}
"""Final (T_co2_out, T_particle_out) [K] of the Table 2 cases, read by eye from Fig. 9 (+-3 K)."""

# --- variants --------------------------------------------------------------------------------

BASELINE = {"n_z": 100, "n_y": 10}


def variants() -> dict[str, dict[str, Any]]:
    """Sensitivity variants: changes to ``fang2022_params`` kwargs (``params``), model
    kwargs (``model``) and protocol (``co2_in_fixed_velocity``)."""
    bed = BauxiteFitBed()
    return {
        "baseline": {},
        "k_eff x0.7": {"params": {"closure": ScaledClosure(bed, k_factor=0.7)}},
        "k_eff x1.5": {"params": {"closure": ScaledClosure(bed, k_factor=1.5)}},
        "closure YagiKuniiBed (k about 0.27, no radiation)": {
            "params": {"closure": YagiKuniiBed()}
        },
        "closure StagnantBed (ZS + radiation)": {"params": {"closure": StagnantBed()}},
        "bed cp x0.9": {"cp_factor": 0.9},
        "bed cp x1.1": {"cp_factor": 1.1},
        "d_p 100 um": {"params": {"d_p": 100e-6}},
        "d_p 500 um": {"params": {"d_p": 500e-6}},
        "d_p 750 um": {"params": {"d_p": 750e-6}},
        "plate C_w x0.01 (no plate mass)": {"plate_factor": 0.01},
        "plate C_w x0.5": {"plate_factor": 0.5},
        "plate C_w x2": {"plate_factor": 2.0},
        "sCO2 laminar Nu = 8.235": {"model": {"co2_htc": "laminar"}},
        "sCO2 h = 1e5 (no sCO2 resistance)": {"model": {"co2_htc": 1e5}},
        "T_co2_in steps at fixed mass flow": {"co2_in_fixed_velocity": False},
        "grid n_z = 50, n_y = 8": {"model": {"n_z": 50, "n_y": 8}},
        "grid n_z = 200, n_y = 20": {"model": {"n_z": 200, "n_y": 20}},
    }


def build_model(spec: dict[str, Any]) -> MovingBedHXDynamic:
    params = fang2022_params(**spec.get("params", {}))
    if "cp_factor" in spec:
        part = params.particle
        params = dataclasses.replace(
            params, particle=part.with_(cp_a=part.cp_a * spec["cp_factor"])
        )
    kwargs = {**BASELINE, **spec.get("model", {})}
    model = MovingBedHXDynamic(params, **kwargs)
    if "plate_factor" in spec:
        kwargs["plate_heat_capacity"] = model.plate_heat_capacity * spec["plate_factor"]
        model = MovingBedHXDynamic(params, **kwargs)
    return model


# --- response analysis -----------------------------------------------------------------------


def t95_cross(t: np.ndarray, s: np.ndarray) -> float:
    above = np.nonzero(s >= 0.95)[0]
    if above.size == 0:
        return float("nan")
    i = int(above[0])
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (0.95 - s[i - 1]) * (t[i] - t[i - 1]) / (s[i] - s[i - 1]))


def t95_settle(t: np.ndarray, s: np.ndarray) -> float:
    outside = np.nonzero(np.abs(s - 1.0) > 0.05)[0]
    if outside.size == 0:
        return float(t[0])
    i = int(outside[-1])
    return float(t[min(i + 1, t.size - 1)])


def fit_exponential(t: np.ndarray, curves: list[np.ndarray]) -> tuple[float, float]:
    """Pooled least-squares fit of ``1 - exp(-max(t - A, 0) / B)``; returns ``(A, B)`` [s]."""
    tt = np.concatenate([t] * len(curves))
    ss = np.concatenate(curves)

    def resid(p: np.ndarray) -> np.ndarray:
        return 1.0 - np.exp(-np.maximum(tt - p[0], 0.0) / p[1]) - ss

    best = None
    for A0 in (0.0, 100.0, 300.0):
        r = least_squares(resid, [A0, 200.0], bounds=([0.0, 1.0], [2000.0, 5000.0]))
        if best is None or r.cost < best.cost:
            best = r
    return float(best.x[0]), float(best.x[1])


def disturbed_inputs(
    model: MovingBedHXDynamic, family: str, value: float, fixed_velocity: bool
) -> tuple[np.ndarray, np.ndarray]:
    u, d = model.nominal_inputs()
    d = d.copy()
    width = model.params.width
    if family == "T_particle_in":
        d[0] = value
    elif family == "T_co2_in":
        d[1] = value
        if fixed_velocity:
            d[2] = fang2022_co2_mass_flow(FANG2022_BASE["u_co2_in"], value, width)
    else:
        d[2] = fang2022_co2_mass_flow(value, FANG2022_BASE["T_co2_in"], width)
    return u, d


def base_summary(model: MovingBedHXDynamic, ss: Any) -> dict[str, float]:
    u, d = ss.u, ss.d
    P_p, P_c = (float(v) for v in model.casadi()["energy"](ss.x, u, d).full().ravel())
    T_c_out, T_p_out = (float(v) for v in ss.y)
    dT_hot, dT_cold = d[0] - T_c_out, T_p_out - d[1]
    lmtd = (dT_hot - dT_cold) / np.log(dT_hot / dT_cold)
    area = 2.0 * model.params.height * model.params.width
    return {
        "T_co2_out": T_c_out,
        "T_particle_out": T_p_out,
        "duty_W": P_c,
        "energy_residual": (P_p - P_c) / P_c,
        "U_W_m2K": P_c / (area * lmtd),
        "steady_residual_K_s": float(ss.residual),
    }


def run_family(
    model: MovingBedHXDynamic, base: Any, family: str, fixed_velocity: bool = True
) -> dict[str, Any]:
    t = np.arange(0.0, T_END + 0.5 * DT_SAMPLE, DT_SAMPLE)
    out: dict[str, Any] = {"steps": [], "t": t}
    curves: dict[str, list[np.ndarray]] = {k: [] for k in OUTPUTS}
    for value in FANG2022_STEPS[family]:
        u, d = disturbed_inputs(model, family, value, fixed_velocity)
        new = steady_state(model, u, d, x_guess=base.x)
        traj = simulate_response(model, base.x, u, d, t)
        model.check_range(traj.x)
        step: dict[str, Any] = {"value": value, "y": traj.y, "y_inf": new.y.tolist()}
        for k, name in enumerate(OUTPUTS):
            s = (traj.y[:, k] - base.y[k]) / (new.y[k] - base.y[k])
            curves[name].append(s)
            step[name] = {
                "t95_cross": t95_cross(t, s),
                "t95_settle": t95_settle(t, s),
                "end_vs_steady_K": float(traj.y[-1, k] - new.y[k]),
                "overshoot": float(np.max(s) - 1.0),
                "undershoot": float(-np.min(s)),
            }
        out["steps"].append(step)
    summary = {}
    for name in OUTPUTS:
        cross = [st[name]["t95_cross"] for st in out["steps"]]
        settle = [st[name]["t95_settle"] for st in out["steps"]]
        A, B = fit_exponential(t, curves[name])
        summary[name] = {
            "t95_cross_mean": float(np.mean(cross)),
            "t95_cross_min": float(np.min(cross)),
            "t95_cross_max": float(np.max(cross)),
            "t95_settle_mean": float(np.mean(settle)),
            "fit_A": A,
            "fit_B": B,
            "t95_fit": A + B * np.log(20.0),
            "published": FANG2022_RESPONSE_TIMES[family][name],
        }
    out["summary"] = summary
    out["curves"] = curves
    return out


def run_variant(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    model = build_model(spec)
    u, d = model.nominal_inputs()
    base = steady_state(model, u, d)
    result = {"name": name, "base": base_summary(model, base), "families": {}}
    for family in FAMILIES:
        fam = run_family(model, base, family, spec.get("co2_in_fixed_velocity", True))
        result["families"][family] = fam["summary"]
    result["nx"] = model.nx
    result["plate_heat_capacity"] = model.plate_heat_capacity
    result["runtime_s"] = time.perf_counter() - start
    return result


# --- output helpers ----------------------------------------------------------------------------


def to_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating | np.integer):
        return obj.item()
    return obj


class Store:
    def __init__(self, out_dir: Path) -> None:
        self.dir = out_dir
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        self.results: dict[str, Any] = {"parts_done": []}
        self.tables: dict[str, str] = {}

    def save(self, part: str, data: Any, table: str | None = None) -> None:
        self.results[part] = to_json(data)
        if part not in self.results["parts_done"]:
            self.results["parts_done"].append(part)
        if table is not None:
            self.tables[part] = table
        (self.dir / "results.json").write_text(json.dumps(self.results, indent=1))
        text = "# Moving-bed HX dynamics against Fang et al. (2022)\n\n" + "\n\n".join(
            self.tables.values()
        )
        (self.dir / "results_table.md").write_text(text + "\n")
        print(f"[saved] {part}", flush=True)


def response_table(families: dict[str, Any], title: str) -> str:
    lines = [
        f"## {title}",
        "",
        "| Step family | Output | Published t95 [s] | Fit A + 3B, this model [s] | "
        "First crossing, mean (min-max) [s] | Settling, mean [s] | Fit (A, B) [s] | "
        "Published (A, B) [s] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for family in (f for f in FAMILIES if f in families):
        for name in OUTPUTS:
            s = families[family][name]
            pa, pb = FANG2022_FITS[family][name]
            lines.append(
                f"| {family} | {name} | {s['published']:.0f} | {s['t95_fit']:.0f} | "
                f"{s['t95_cross_mean']:.0f} ({s['t95_cross_min']:.0f}-{s['t95_cross_max']:.0f}) "
                f"| {s['t95_settle_mean']:.0f} | ({s['fit_A']:.0f}, {s['fit_B']:.0f}) | "
                f"({pa:.1f}, {pb:.1f}) |"
            )
    return "\n".join(lines)


def plot_family(fam: dict[str, Any], family: str, base_y: np.ndarray, path: Path) -> None:
    t = fam["t"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for k, name in enumerate(OUTPUTS):
        ax_raw, ax_norm = axes[k, 0], axes[k, 1]
        for i, step in enumerate(fam["steps"]):
            label = f"{step['value']:g}"
            ax_raw.plot(t, step["y"][:, k], color=COLORS[i], lw=1.5, label=label)
            ax_norm.plot(t, fam["curves"][name][i], color=COLORS[i], lw=1.5)
        A, B = FANG2022_FITS[family][name]
        ax_norm.plot(
            t, 1.0 - np.exp(-np.maximum(t - A, 0.0) / B), "k--", lw=1.5, label="Fang et al. fit"
        )
        s = fam["summary"][name]
        ax_norm.plot(
            t,
            1.0 - np.exp(-np.maximum(t - s["fit_A"], 0.0) / s["fit_B"]),
            color="0.5",
            lw=1.0,
            ls=":",
            label="fit to this model",
        )
        ax_norm.axhline(0.95, color="0.8", lw=0.8)
        ax_raw.set_ylabel(f"{name} [K]")
        ax_norm.set_ylabel(f"normalised {name} [-]")
        ax_norm.set_ylim(-0.1, 1.15)
        ax_norm.legend(frameon=False, fontsize=8, loc="lower right")
        ax_raw.axhline(base_y[k], color="0.8", lw=0.8)
    axes[0, 0].legend(frameon=False, fontsize=8, title="new value", ncol=2)
    for ax in axes[1]:
        ax.set_xlabel("time after the step [s]")
    for ax in axes.ravel():
        ax.grid(color="0.92")
    fig.suptitle(f"{FAMILY_LABEL[family]} (this model, n_z = 100, n_y = 10)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- main ----------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hx_dynamics"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    workers = max(1, min(args.workers, 4))
    store = Store(args.output_dir)
    figs = args.output_dir / "figures"
    t_start = time.perf_counter()

    # 1. base case ---------------------------------------------------------------------------
    model = build_model({})
    u, d = model.nominal_inputs()
    base = steady_state(model, u, d)
    summary = base_summary(model, base)
    p = model.params
    conj = solve_conjugate(p, n_y=40)
    fang_base = (965.0, 849.5)
    part1 = {
        "inputs": {
            "m_dot_particle_kg_s": float(u[0]),
            "m_dot_co2_kg_s": float(d[2]),
            "width_m": p.width,
            "d_p_m": p.particle.d_p,
            "closure": p.closure.name,
            "plate_heat_capacity_J_m2K": model.plate_heat_capacity,
            "grid": BASELINE,
        },
        "fit_errors": model.fit_errors,
        "model": summary,
        "conjugate_n_y40": {"T_co2_out": conj.T_co2_out, "T_particle_out": conj.T_particle_out},
        "fang_read_from_figures": {"T_co2_out": fang_base[0], "T_particle_out": fang_base[1]},
    }
    table1 = "\n".join(
        [
            "## Base case steady state (Fang et al. Table 1; bed and plate properties from "
            "the papers they cite)",
            "",
            "| Quantity | This model (100 x 10) | 2-D conjugate (n_y = 40) | Fang et al. "
            "(read from Figs. 3-9, +-3 K) |",
            "|---|---|---|---|",
            f"| sCO2 outlet [K] | {summary['T_co2_out']:.1f} | {conj.T_co2_out:.1f} | "
            f"{fang_base[0]:.0f} |",
            f"| particle outlet [K] | {summary['T_particle_out']:.1f} | "
            f"{conj.T_particle_out:.1f} | {fang_base[1]:.1f} |",
            f"| U = Q/(A LMTD) [W/m2/K] | {summary['U_W_m2K']:.1f} | {conj.U_mean:.1f} | "
            "about 137 (from the read temperatures) |",
        ]
    )
    store.save("base_case", part1, table1)
    prof = model.profiles(base.x, u, d)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(prof["z_cell"], prof["T_bed_mean"], color=COLORS[1], lw=2, label="bed, mean")
    ax.plot(prof["z_cell"], prof["T_bed"][:, 0], color=COLORS[1], lw=1, ls="--", label="bed, wall")
    ax.plot(prof["z_cell"], prof["T_plate"], color="0.4", lw=1.5, label="plate")
    ax.plot(prof["z_face"], prof["T_co2"], color=COLORS[0], lw=2, label="sCO2")
    ax.plot(conj.x, conj.T_particle, "k:", lw=1, label="conjugate solution")
    ax.plot(conj.x, conj.T_co2, "k:", lw=1)
    ax.set_xlabel("depth below particle inlet z [m]")
    ax.set_ylabel("temperature [K]")
    ax.legend(frameon=False)
    ax.grid(color="0.92")
    fig.tight_layout()
    fig.savefig(figs / "base_case_profiles.png", dpi=130)
    plt.close(fig)

    # 2. step families -------------------------------------------------------------------------
    families: dict[str, Any] = {}
    steady_vs_input: dict[str, Any] = {}
    for family in FAMILIES:
        fam = run_family(model, base, family)
        families[family] = fam["summary"]
        steady_vs_input[family] = {
            "x": list(FANG2022_STEPS[family]),
            "T_co2_out": [st["y_inf"][0] for st in fam["steps"]],
            "T_particle_out": [st["y_inf"][1] for st in fam["steps"]],
            "per_step": [
                {"value": st["value"], **{n: st[n] for n in OUTPUTS}} for st in fam["steps"]
            ],
        }
        plot_family(fam, family, base.y, figs / f"steps_{family}.png")
        store.save(
            "response_times_baseline",
            {"families": families, "steady_vs_input": steady_vs_input},
            response_table(families, "95 % response times, baseline inputs"),
        )

    # 3. steady outlet against disturbed input (their Figs. 4, 6, 8) -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    rows = [
        "## New steady outlet temperatures against the disturbed input (their Figs. 4, 6, 8)",
        "",
        "| Family | Input value | sCO2 out, model [K] | sCO2 out, Fang (read) [K] | "
        "particle out, model [K] | particle out, Fang (read) [K] |",
        "|---|---|---|---|---|---|",
    ]
    for ax, family in zip(axes, FAMILIES, strict=True):
        ref = FANG2022_FIG_STEADY[family]
        sv = steady_vs_input[family]
        base_x = {"T_particle_in": 1048.0, "T_co2_in": 823.0, "u_co2_in": 0.5}[family]
        xs = sorted([*sv["x"], base_x])
        order = np.argsort([*sv["x"], base_x])
        tc = np.array([*sv["T_co2_out"], base.y[0]])[order]
        tp = np.array([*sv["T_particle_out"], base.y[1]])[order]
        ax.plot(xs, tc, color=COLORS[0], lw=2, marker="o", ms=5, label="sCO2, this model")
        ax.plot(xs, tp, color=COLORS[1], lw=2, marker="o", ms=5, label="particle, this model")
        ax.plot(
            ref["x"],
            ref["T_co2_out"],
            "D",
            mfc="none",
            color=COLORS[0],
            ms=8,
            label="sCO2, Fang et al.",
        )
        ax.plot(
            ref["x"],
            ref["T_particle_out"],
            "s",
            mfc="none",
            color=COLORS[1],
            ms=8,
            label="particle, Fang et al.",
        )
        ax.set_xlabel(family + (" [m/s]" if family == "u_co2_in" else " [K]"))
        ax.set_ylabel("new steady outlet temperature [K]")
        ax.grid(color="0.92")
        for xv, a, b, c, e in zip(xs, tc, ref["T_co2_out"], tp, ref["T_particle_out"], strict=True):
            rows.append(f"| {family} | {xv:g} | {a:.1f} | {b:.0f} | {c:.1f} | {e:.0f} |")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "steady_vs_input.png", dpi=130)
    plt.close(fig)
    store.save("steady_vs_input", steady_vs_input, "\n".join(rows))

    # 4. Table 2 combined disturbances (open loop) -------------------------------------------------
    t = np.arange(0.0, T_END + 0.5 * DT_SAMPLE, DT_SAMPLE)
    table2 = {}
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    rows = [
        "## Table 2 combined disturbances, open loop",
        "",
        "| Case | sCO2 out final, model [K] | Fang (read, Fig. 9) [K] | particle out final, "
        "model [K] | Fang (read) [K] | particle out min during transient, model [K] |",
        "|---|---|---|---|---|---|",
    ]
    for i, (case, (Tp, Tc, v)) in enumerate(FANG2022_TABLE2.items()):
        uu, dd = model.nominal_inputs()
        dd = np.array([Tp, Tc, fang2022_co2_mass_flow(v, Tc, p.width)])
        traj = simulate_response(model, base.x, uu, dd, t)
        new = steady_state(model, uu, dd, x_guess=base.x)
        table2[case] = {
            "final": new.y.tolist(),
            "y": traj.y[::10],
            "particle_min": float(traj.y[:, 1].min()),
        }
        for k in range(2):
            axes[k].plot(t, traj.y[:, k], color=COLORS[i], lw=1.5, label=case)
        ref = FANG2022_FIG9_FINAL[case]
        rows.append(
            f"| {case} | {new.y[0]:.1f} | {ref[0]:.0f} | {new.y[1]:.1f} | {ref[1]:.0f} | "
            f"{traj.y[:, 1].min():.1f} |"
        )
    axes[0].set_ylabel("T_co2_out [K]")
    axes[1].set_ylabel("T_particle_out [K]")
    axes[1].set_xlabel("time after the disturbance [s]")
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    for ax in axes:
        ax.grid(color="0.92")
    fig.tight_layout()
    fig.savefig(figs / "table2_cases.png", dpi=130)
    plt.close(fig)
    store.save("table2_cases", table2, "\n".join(rows))

    # 5. sensitivity and grid ---------------------------------------------------------------
    if not args.skip_sensitivity:
        specs = variants()
        results: dict[str, Any] = {}
        rows = [
            "## Sensitivity of base outlets and 95 % response times (fit A + 3B / first "
            "crossing, mean of six steps) [s]",
            "",
            "| Variant | sCO2 out [K] | particle out [K] | U [W/m2/K] | "
            + " | ".join(f"{f}: {o}" for f in FAMILIES for o in ("sCO2", "particle"))
            + " |",
            "|---|---|---|---|" + "---|" * 6,
            "| **published** | 965 | 849.5 | ~137 | "
            + " | ".join(
                f"**{FANG2022_RESPONSE_TIMES[f][o]:.0f}**" for f in FAMILIES for o in OUTPUTS
            )
            + " |",
        ]

        def add_row(res: dict[str, Any]) -> None:
            b = res["base"]
            cells = [
                f"{res['families'][f][o]['t95_fit']:.0f} / "
                f"{res['families'][f][o]['t95_cross_mean']:.0f}"
                for f in FAMILIES
                for o in OUTPUTS
            ]
            rows.append(
                f"| {res['name']} | {b['T_co2_out']:.1f} | {b['T_particle_out']:.1f} | "
                f"{b['U_W_m2K']:.1f} | " + " | ".join(cells) + " |"
            )

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {name: pool.submit(run_variant, name, spec) for name, spec in specs.items()}
            for name in specs:
                res = futures[name].result()
                results[name] = res
                add_row(res)
                print(f"  {name}: {res['runtime_s']:.0f} s", flush=True)
                store.save("sensitivity", results, "\n".join(rows))

        names = list(results)
        fig, axes = plt.subplots(1, 6, figsize=(16, 6.5), sharey=True)
        for ax, (family, out) in zip(
            axes, [(f, o) for f in FAMILIES for o in OUTPUTS], strict=True
        ):
            vals = [results[n]["families"][family][out]["t95_fit"] for n in names]
            ax.barh(range(len(names)), vals, color=COLORS[0], height=0.6)
            ax.axvline(FANG2022_RESPONSE_TIMES[family][out], color="k", ls="--", lw=1.5)
            ax.set_title(f"{family}\n{out}", fontsize=9)
            ax.set_xlabel("t95 (fit) [s]")
            ax.grid(axis="x", color="0.92")
        axes[0].set_yticks(range(len(names)), names, fontsize=8)
        axes[0].invert_yaxis()
        fig.suptitle("95 % response time by variant; dashed line: Fang et al. (2022)")
        fig.tight_layout()
        fig.savefig(figs / "sensitivity_t95.png", dpi=130)
        plt.close(fig)

    print(f"done in {time.perf_counter() - t_start:.0f} s; see {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
