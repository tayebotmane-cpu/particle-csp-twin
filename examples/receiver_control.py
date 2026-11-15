"""Full receiver outlet-temperature control study: digital twin versus PID.

Usage (from the repository root)::

    .venv/bin/python examples/receiver_control.py                       # 20 seeds, ~2 h
    .venv/bin/python examples/receiver_control.py --seeds 5 --sensitivity-seeds 3 --workers 4

Steps: tune the baselines and the twin on the controller model (PID gains,
power-signal filter, NMPC move weight; ``--skip-tuning`` uses the stored
defaults), check observability of the incident power, measure the estimators'
open-loop detection delay, run the Monte Carlo over noise seeds for all
configurations, then the mismatch / step-size sensitivity. Writes under
``--output-dir`` (default ``outputs/receiver_control``): ``results_table.txt``,
``results.json`` and ``figures/*.png``. They are written after each step, so a
run that is interrupted keeps the steps it finished (``stages_done`` in
``results.json``). See :mod:`pcsp.pipelines.receiver_control`.
"""

from __future__ import annotations

import os

# One BLAS thread per process: the Monte Carlo parallelises over processes.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pcsp.constants import T_ZERO_C  # noqa: E402
from pcsp.pipelines.receiver_control import (  # noqa: E402
    CONFIGURATIONS,
    LABELS,
    SANDIA_LIMIT,
    SETTLING_BAND,
    ConfigSummary,
    ExperimentSetup,
    RunResult,
    SensitivityPoint,
    default_workers,
    estimator_detection,
    format_results_table,
    observability_report,
    run_monte_carlo,
    run_sensitivity,
    summarise,
    tune_move_weight,
    tune_pid,
    tune_power_filter,
)

# Reference categorical palette (light surface), slots 1-6 in configuration order.
COLOR = dict(
    zip(
        CONFIGURATIONS,
        ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"),
        strict=True,
    )
)
# Idealised configurations (they receive plant truth) are drawn dashed.
STYLE = {c: "--" if c in ("B_true", "E") else "-" for c in CONFIGURATIONS}
METRIC_NAME = {"iae": "integral absolute error", "two_sigma_max": "worst 5-min 2 sigma"}
SURFACE, INK, INK_2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#eeede9"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "font.size": 9,
    }
)


def tune(setup: ExperimentSetup, workers: int) -> tuple[ExperimentSetup, dict]:
    """PID gains, then the power-signal filter, then the NMPC move weight (controller model)."""
    t0 = time.perf_counter()
    pid = tune_pid(setup)
    setup = dataclasses.replace(setup, pid=pid.gains)
    filt = tune_power_filter(setup)
    best_filter = min(filt, key=filt.get)
    setup = dataclasses.replace(
        setup, instruments=dataclasses.replace(setup.instruments, power_filter_time=best_filter)
    )
    weights = tune_move_weight(setup, workers=workers)
    best_weight = min(weights, key=weights.get)
    setup = dataclasses.replace(
        setup, nmpc=dataclasses.replace(setup.nmpc, move_weight=best_weight)
    )
    info = {
        "pid": dataclasses.asdict(pid),
        "power_filter_iae": filt,
        "move_weight_iae": weights,
        "wall_time_s": time.perf_counter() - t0,
    }
    print(
        f"PID tuned: kp={pid.gains.kp:.4g} ki={pid.gains.ki:.4g} kd={pid.gains.kd:.3g} "
        f"(IAE {pid.initial_objective:.0f} -> {pid.objective:.0f} K s, {pid.n_evaluations} evals)"
    )
    print(f"power filter scan (IAE): {filt} -> {best_filter} s")
    print(f"move weight scan (IAE): {weights} -> {best_weight}")
    return setup, info


# ------------------------------------------------------------------- figures
def _mark_steps(ax: plt.Axes, setup: ExperimentSetup) -> None:
    sc = setup.scenario
    for t, text in ((sc.t_step, "power step down"), (sc.t_restore, "power restored")):
        ax.axvline(t, color=INK_2, linewidth=0.8, linestyle=":")
        ax.annotate(text, (t, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -10),
                    textcoords="offset points", fontsize=8, color=INK_2)  # fmt: skip


def plot_timeseries(results: list[RunResult], setup: ExperimentSetup, path: Path) -> None:
    sc = setup.scenario
    by = {r.config: r.trajectory for r in results if r.trajectory is not None}
    seed = next(r.seed for r in results if r.trajectory is not None)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    T_set = sc.T_set - T_ZERO_C
    ax1.axhspan(T_set - SETTLING_BAND, T_set + SETTLING_BAND, color=BAND, zorder=0,
                label=f"setpoint +/- {SETTLING_BAND:.0f} K")  # fmt: skip
    ax1.axhline(T_set, color=INK, linewidth=0.8)
    for c in CONFIGURATIONS:
        tr = by[c]
        ax1.plot(tr.t, tr.T_true - T_ZERO_C, STYLE[c], color=COLOR[c], label=LABELS[c],
                 linewidth=1.2)  # fmt: skip
        ax2.plot(tr.t, tr.m_dot, STYLE[c], color=COLOR[c], linewidth=1.2)
    ax1.set_ylabel("outlet thermocouple, noise-free [degC]")
    ax1.set_ylim(T_set - 22.0, T_set + 22.0)
    ax2.set_ylabel("particle mass flow [kg/s]")
    ax2.set_xlabel("time [s]")
    for ax in (ax1, ax2):
        _mark_steps(ax, setup)
    ax1.legend(loc="lower left", ncol=2, fontsize=8)
    ax1.set_title(f"Outlet temperature control, power 739 -> 610 -> 739 kW (noise seed {seed})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_power_estimate(results: list[RunResult], setup: ExperimentSetup, path: Path) -> None:
    by = {r.config: r.trajectory for r in results if r.trajectory is not None}
    fig, ax = plt.subplots(figsize=(9, 4))
    tr = by["C"]
    ax.plot(tr.t, by["D"].Q_signal / 1e3, color=INK_2, linewidth=0.6, alpha=0.35,
            label="power signal (biased +8 %, lagged, noisy)")  # fmt: skip
    ax.plot(tr.t, tr.Q_true / 1e3, color=INK, linewidth=1.4, label="true incident power")
    ax.plot(tr.t, tr.Q_estimate / 1e3, color=COLOR["C"], label="C: EnKF estimate (T, m_dot)")
    ax.plot(tr.t, by["D"].Q_estimate / 1e3, color=COLOR["D"],
            label="D: EnKF estimate (T, m_dot, power signal)")  # fmt: skip
    _mark_steps(ax, setup)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("incident power [kW]")
    ax.set_title("Incident-power estimate in closed loop; the offset absorbs the model mismatch")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary(results: list[RunResult], summaries: dict[str, ConfigSummary], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    ypos = np.arange(len(CONFIGURATIONS))[::-1]
    for ax, field, unit in ((axes[0], "two_sigma_max", "K"), (axes[1], "iae", "K s")):
        for y, c in zip(ypos, CONFIGURATIONS, strict=True):
            vals = [getattr(r.metrics, field) for r in results if r.config == c]
            s = summaries[c]
            ax.barh(y, s.mean[field], height=0.6, color=COLOR[c], alpha=0.85)
            ax.errorbar(s.mean[field], y, xerr=s.std[field], color=INK, capsize=3, linewidth=1)
            jitter = np.linspace(-0.18, 0.18, len(vals))
            ax.plot(vals, y + jitter, "o", color=INK, markersize=2.5, alpha=0.5)
        ax.set_xlabel(f"{METRIC_NAME[field]} [{unit}]")
    axes[0].axvline(SANDIA_LIMIT, color=INK, linestyle="--", linewidth=1)
    axes[0].annotate("Sandia limit 10 K", (SANDIA_LIMIT, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(3, -10), textcoords="offset points", fontsize=8)  # fmt: skip
    axes[0].set_title("worst 5-min 2 sigma of measured outlet T")
    axes[1].set_title("integral absolute error after the step")
    axes[0].set_yticks(ypos, [LABELS[c] for c in CONFIGURATIONS])
    n = summaries[CONFIGURATIONS[0]].n_runs
    fig.suptitle(f"Monte Carlo over {n} noise seeds: bars mean, whiskers +/- 1 std, dots runs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensitivity(points: list[SensitivityPoint], path: Path) -> None:
    mismatch = [p for p in points if "step" not in p.label]
    big = [p for p in points if "step" in p.label]
    configs = [c for c in CONFIGURATIONS if c in mismatch[0].summaries]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.array([p.h_adv_ratio for p in mismatch])
    for ax, field, unit in ((axes[0], "iae", "K s"), (axes[1], "two_sigma_max", "K")):
        for c in configs:
            m = np.array([p.summaries[c].mean[field] for p in mismatch])
            sd = np.array([p.summaries[c].std[field] for p in mismatch])
            ax.errorbar(x, m, yerr=sd, fmt="o" + STYLE[c], color=COLOR[c], label=LABELS[c],
                        capsize=3, markersize=5)  # fmt: skip
        ax.set_xlabel("plant h_adv / model h_adv [-]")
        ax.set_ylabel(f"{METRIC_NAME[field]} [{unit}]")
    axes[1].axhline(SANDIA_LIMIT, color=INK, linestyle="--", linewidth=1)
    axes[0].legend(fontsize=8)
    axes[0].set_title("IAE vs advective-loss error")
    axes[1].set_title("worst 2 sigma vs advective-loss error")
    ax = axes[2]
    # The large step is compared with the published step at the same plant mismatch.
    same = [p for p in mismatch if big and np.isclose(p.h_adv_ratio, big[0].h_adv_ratio)]
    if same:
        ref = same[0]
        width = 0.38
        xs = np.arange(len(configs))
        for k, (p, alpha) in enumerate(((ref, 0.45), (big[0], 0.95))):
            m = [p.summaries[c].mean["iae"] for c in configs]
            sd = [p.summaries[c].std["iae"] for c in configs]
            ax.bar(xs + (k - 0.5) * width, m, width=width - 0.04, yerr=sd, capsize=3,
                   color=[COLOR[c] for c in configs], alpha=alpha,
                   label=f"{100 * p.step_fraction:.1f} % step")  # fmt: skip
        ax.set_xticks(xs, [c for c in configs])
        ax.set_ylabel("IAE [K s]")
        light, dark = 100 * ref.step_fraction, 100 * big[0].step_fraction
        ax.set_title(f"step size at h_adv x{ref.h_adv_ratio:g} (light: {light:.1f} %, "
                     f"dark: {dark:.0f} %)", fontsize=8)  # fmt: skip
    n = mismatch[0].summaries[configs[0]].n_runs
    fig.suptitle(f"Sensitivity (mean +/- 1 std over {n} seeds; controllers not re-tuned)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def _jsonable(obj: object) -> object:
    if dataclasses.is_dataclass(obj):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float | np.floating) and not np.isfinite(obj):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=20, help="Monte Carlo noise seeds")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--sensitivity-seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/receiver_control"))
    parser.add_argument("--skip-tuning", action="store_true", help="use the stored defaults")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    wall = {}
    t_start = time.perf_counter()

    setup = ExperimentSetup()
    tuning = None
    if not args.skip_tuning:
        setup, tuning = tune(setup, args.workers)
        wall["tuning"] = tuning["wall_time_s"]

    stages_done: list[str] = []
    payload: dict[str, object] = {
        "setup": setup,
        "tuning": tuning,
        "workers": args.workers,
        "stages_done": stages_done,
        "wall_time_s": wall,
    }

    def save(stage: str) -> None:
        """Write everything computed so far, so an interrupted run keeps its finished stages."""
        stages_done.append(stage)
        wall["total"] = time.perf_counter() - t_start
        text = json.dumps(_jsonable(payload), indent=1)
        (args.output_dir / "results.json").write_text(text)

    save("setup" if tuning is None else "tuning")
    obs = observability_report(setup)
    payload["observability"] = obs
    print("\nObservability of incident power (augmented controller model):")
    for row in obs:
        print(f"  {row.sensors:30s} Q={row.power / 1e3:4.0f} kW  rank {row.rank}/{row.n_states}  "
              f"cond {row.condition_number:7.1f}")  # fmt: skip

    t0 = time.perf_counter()
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    detection = [estimator_detection(c, setup, s) for c in ("C", "D") for s in seeds[:5]]
    wall["detection"] = time.perf_counter() - t0
    print(f"\nOpen-loop detection of the power step (gate held; mean over {len(seeds[:5])} "
          "seeds) [s]:")  # fmt: skip
    for c in ("C", "D"):
        rows = [d for d in detection if d.config == c]
        mean = {k: np.mean([getattr(d, k) for d in rows]) for k in
                ("t_detect", "t50", "t90", "tc_t50", "tc_t90", "final_error")}  # fmt: skip
        print(f"  {c}: 3-sigma departure {mean['t_detect']:.1f}, 50 % {mean['t50']:.1f}, "
              f"90 % {mean['t90']:.1f}; thermocouple 50 % {mean['tc_t50']:.1f}, "
              f"90 % {mean['tc_t90']:.1f}; settled estimate - true "
              f"{mean['final_error'] / 1e3:.1f} kW")  # fmt: skip
    payload["detection"] = detection
    save("detection")

    t0 = time.perf_counter()
    results = run_monte_carlo(setup, seeds=seeds, workers=args.workers, trajectory_seed=seeds[0])
    wall["monte_carlo"] = time.perf_counter() - t0
    summaries = summarise(results)
    table = format_results_table(summaries)
    print(f"\nMonte Carlo, {len(seeds)} seeds (mean +/- std):\n{table}")
    for c in CONFIGURATIONS:
        print(f"  {c}: Sandia pass rate {100 * summaries[c].pass_rate:.0f} %, worst 2 sigma "
              f"{summaries[c].worst['two_sigma_max']:.2f} K")  # fmt: skip
    (args.output_dir / "results_table.txt").write_text(table + "\n")
    plot_timeseries(results, setup, fig_dir / "timeseries.png")
    plot_power_estimate(results, setup, fig_dir / "power_estimate.png")
    plot_summary(results, summaries, fig_dir / "summary.png")
    payload["summaries"] = summaries
    payload["runs"] = [dataclasses.replace(r, trajectory=None) for r in results]
    save("monte_carlo")

    sensitivity = []
    if not args.skip_sensitivity:
        t0 = time.perf_counter()
        sensitivity = run_sensitivity(
            setup, seeds=tuple(range(args.sensitivity_seeds)), workers=args.workers
        )
        wall["sensitivity"] = time.perf_counter() - t0
        print("\nSensitivity (IAE [K s] / worst-window 2 sigma [K], mean over seeds):")
        for p in sensitivity:
            cells = "  ".join(
                f"{c}: {s.mean['iae']:6.0f} / {s.mean['two_sigma_max']:5.2f} "
                f"({100 * s.pass_rate:3.0f} % pass)"
                for c, s in p.summaries.items()
            )
            print(f"  {p.label:14s} {cells}")
        plot_sensitivity(sensitivity, fig_dir / "sensitivity.png")
        payload["sensitivity"] = sensitivity
        save("sensitivity")

    print(f"\nwall time [s]: {', '.join(f'{k} {v:.0f}' for k, v in wall.items())}")
    print(f"results in {args.output_dir}")


if __name__ == "__main__":
    main()
