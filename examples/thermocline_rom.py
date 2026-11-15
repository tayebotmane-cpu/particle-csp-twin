"""Full thermocline-surrogate study: campaign, surrogates A and B, table and figures.

Usage (from the repository root)::

    .venv/bin/python examples/thermocline_rom.py                 # 120 runs, seed 0
    .venv/bin/python examples/thermocline_rom.py --n-runs 150 --seed 1 --workers 4

Writes under ``--output-dir`` (default ``outputs/thermocline_rom``):
``campaign/`` (manifest + one Zarr store per run; re-running resumes it),
``results_table.txt``, ``results.json``, ``per_run_errors.csv`` and ``figures/*.png``.
See :mod:`pcsp.pipelines.thermocline_rom` for the method.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pcsp.pipelines.thermocline_rom import (  # noqa: E402
    VARIANTS,
    Evaluation,
    StudyConfig,
    StudyResult,
    format_results_table,
    run_study,
    summarise,
)
from pcsp.rom import rank_for_energy  # noqa: E402

# Reference categorical palette (light surface): slot 1 blue, slot 2 orange.
COLOR = {"A": "#2a78d6", "B": "#eb6834"}
LABEL = {"A": "A: physical time t", "B": "B: front-aligned time tau"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SPLITS = ("interpolation", "extrapolation")

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
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
        "legend.frameon": False,
        "font.size": 10,
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spectra(study: StudyResult, path: Path) -> None:
    """Singular values and training projection error of A and B (interpolation training set)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for v in VARIANTS:
        s = study.spectra[v][:-1]  # centred data: the last singular value is zero by construction
        r = np.arange(1, s.size + 1)
        ax1.semilogy(r, s / s[0], "o-", color=COLOR[v], label=LABEL[v], markersize=4)
        tail = np.sqrt(np.maximum(1.0 - np.cumsum(s**2) / np.sum(s**2), 0.0))  # round-off
        ax2.semilogy(r[:-1], tail[:-1], "o-", color=COLOR[v], label=LABEL[v], markersize=4)
    ax1.set(xlabel="mode index i", ylabel="s_i / s_1  [-]", title="Singular values")
    ax2.set(
        xlabel="retained rank r",
        ylabel="sqrt(1 - E_r)  [-]",
        title="Training projection error of the fluctuations",
    )
    ax1.legend()
    _save(fig, path)


def plot_rank_sweeps(study: StudyResult, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, split in zip(axes, SPLITS, strict=True):
        for v in VARIANTS:
            sw = study.rank_sweeps[(split, v)]
            ax.loglog(sw.x, sw.field_error, "o-", color=COLOR[v], label=f"{v}: ROM error")
            ax.loglog(sw.x, sw.projection_error, "s--", color=COLOR[v], label=f"{v}: projection")
        ax.set(
            xlabel="retained rank r",
            ylabel="median relative L2 error of T_s - T_cold  [-]",
            title=f"{split} test set",
        )
    axes[0].legend()
    _save(fig, path)


def plot_size_sweep(study: StudyResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for v in VARIANTS:
        sw = study.size_sweeps[v]
        ax.loglog(sw.x, sw.field_error, "o-", color=COLOR[v], label=f"{v}: ROM error (median)")
        ax.loglog(sw.x, sw.projection_error, "s--", color=COLOR[v], label=f"{v}: projection")
        ax.annotate(f"r={sw.rank[-1]}", (sw.x[-1], sw.field_error[-1]), color=INK_2,
                    textcoords="offset points", xytext=(4, 4), fontsize=8)  # fmt: skip
    ax.set(
        xlabel="number of training runs",
        ylabel="relative L2 error of T_s - T_cold  [-]",
        title="Interpolation test set (rank by energy criterion)",
    )
    ax.legend()
    _save(fig, path)


def plot_profiles(study: StudyResult, split: str, path: Path) -> None:
    """True (black) and predicted (coloured, dashed) T_s(z) for the median and worst run."""
    cfg, data = study.config, study.data
    t = data.grids.t
    picks = [int(round(f * (t.size - 1))) for f in (0.25, 0.5, 0.75, 1.0)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for col, v in enumerate(VARIANTS):
        e: Evaluation = study.results[(split, v)].evaluation
        order = np.argsort(e.field_error)
        cases = {"median case": order[len(order) // 2], "worst case": order[-1]}
        for row, (name, k) in enumerate(cases.items()):
            ax = axes[row, col]
            for i in picks:
                ax.plot(data.z, cfg.T_cold + e.truth[k, i], color=INK, linewidth=1.2)
                ax.plot(data.z, cfg.T_cold + e.predicted[k, i], "--", color=COLOR[v])
                # Label each time at its front: the cell nearest the mid-temperature rise.
                j = int(np.argmin(np.abs(e.truth[k, i] - 0.5 * e.truth[k, i, 0])))
                ax.annotate(f"{t[i] / 3600:.1f} h", (data.z[j], cfg.T_cold + e.truth[k, i, j]),
                            fontsize=8, color=INK_2, textcoords="offset points",
                            xytext=(5, 2))  # fmt: skip
            m, th, dp = e.params[k]
            ax.set_title(
                f"{v}, {name}: err {e.field_error[k]:.1e}\n"
                f"m={m:.2f} kg/s, T_hot={th:.0f} K, d_p={dp * 1e3:.1f} mm",
                fontsize=9,
            )
            if row == 1:
                ax.set_xlabel("z from hot end  [m]")
            if col == 0:
                ax.set_ylabel("solid temperature T_s  [K]")
    axes[0, 0].plot([], [], color=INK, linewidth=1.2, label="model")
    axes[0, 0].plot([], [], "--", color=COLOR["A"], label="surrogate A")
    axes[0, 1].plot([], [], "--", color=COLOR["B"], label="surrogate B")
    axes[0, 0].legend(loc="lower left")  # cold region near the hot end: always empty
    axes[0, 1].legend(loc="lower left")
    fig.suptitle(f"Profiles at 1/4, 1/2, 3/4 and the end of charge ({split} test set)")
    _save(fig, path)


def plot_extrapolation(study: StudyResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for v in VARIANTS:
        e = study.results[("extrapolation", v)].evaluation
        ax.semilogy(e.distance_outside, e.field_error, "o", color=COLOR[v], label=LABEL[v],
                    markeredgecolor=SURFACE, markeredgewidth=1.0)  # fmt: skip
        floor = np.median(study.results[("interpolation", v)].evaluation.field_error)
        ax.axhline(floor, color=COLOR[v], linestyle=":", linewidth=1.2)
        ax.annotate(f"{v}: interpolation median", (0.0, floor), color=INK_2, fontsize=8,
                    textcoords="offset points", xytext=(2, 3))  # fmt: skip
    ax.set(
        xlabel="distance outside training range (normalised mass flow)  [-]",
        ylabel="relative L2 error of T_s - T_cold  [-]",
        title="Extrapolation to higher mass flow",
    )
    ax.legend()
    _save(fig, path)


def results_json(study: StudyResult) -> dict:
    cfg = dataclasses.asdict(study.config)
    out = {
        "config": json.loads(json.dumps(cfg, default=str)),
        "n_ok": study.n_ok,
        "n_failed": study.n_failed,
        "campaign_seconds_all_runs": study.campaign_seconds,
        "rank_for_training_projection_error": {
            v: {str(eps): rank_for_energy(study.spectra[v], 1.0 - eps**2) for eps in (1e-2, 1e-3)}
            for v in VARIANTS
        },
        "results": [],
    }
    for (split, v), r in study.results.items():
        e = r.evaluation
        out["results"].append(
            {
                "split": split,
                "variant": v,
                "n_train": r.n_train,
                "n_test": int(e.field_error.size),
                "rank": r.rank,
                "projection_error": dataclasses.asdict(summarise(e.projection_error)),
                "field_error": dataclasses.asdict(summarise(e.field_error)),
                "max_abs_error_K": dataclasses.asdict(summarise(e.max_abs_error)),
                "soc_error_pp": dataclasses.asdict(summarise(e.soc_error)),
                "outlet_error_K": dataclasses.asdict(summarise(e.outlet_error)),
                "rom_seconds": r.speed.rom_seconds,
                "model_seconds_manifest": r.speed.model_seconds_manifest,
                "model_seconds_window": r.speed.model_seconds_window,
                "speedup_manifest": r.speed.speedup_manifest,
                "speedup_window": r.speed.speedup_window,
                "campaign_seconds_train": r.campaign_seconds,
                "fit_seconds": r.fit_seconds,
                "training_cost_seconds": r.training_cost,
            }
        )
    return out


def write_per_run_csv(study: StudyResult, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["split", "variant", "param_hash", "mass_flow_kg_s", "T_hot_K", "d_p_m",
             "field_error", "projection_error", "max_abs_error_K", "soc_error_pp",
             "outlet_error_K", "distance_outside"]
        )  # fmt: skip
        for (split, v), r in study.results.items():
            e = r.evaluation
            for k in range(e.field_error.size):
                w.writerow(
                    [split, v, e.param_hash[k], *e.params[k], e.field_error[k],
                     e.projection_error[k], e.max_abs_error[k], e.soc_error[k],
                     e.outlet_error[k], e.distance_outside[k]]
                )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-runs", type=int, default=StudyConfig.n_runs)
    parser.add_argument("--seed", type=int, default=StudyConfig.seed)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/thermocline_rom"))
    parser.add_argument("--workers", type=int, default=4, help="campaign worker processes")
    args = parser.parse_args()

    config = StudyConfig(n_runs=args.n_runs, seed=args.seed)
    out = args.output_dir
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    study = run_study(config, out / "campaign", max_workers=args.workers)
    elapsed = time.perf_counter() - start

    table = format_results_table(study)
    ranks = results_json(study)["rank_for_training_projection_error"]
    notes = "\n".join(
        [
            f"runs ok/failed: {study.n_ok}/{study.n_failed} (executed now: {study.n_executed}); "
            f"campaign model time, all runs: {study.campaign_seconds:.1f} s; "
            f"study wall time: {elapsed:.1f} s",
            "errors are of T_s - T_cold on t in [0, t_end]; 'proj med' = median projection "
            "error of the basis in physical time (for B incl. tau resampling)",
            "speed-up = manifest model wall time / one prediction; '(window)' = model timed on "
            "[0, t_end] only; 'train s' = training-run model time + fit time",
            "rank for training fluctuation projection error 1e-2 / 1e-3: "
            + ", ".join(f"{v}: {ranks[v]['0.01']} / {ranks[v]['0.001']}" for v in VARIANTS),
        ]
    )
    print(table)
    print(notes)
    (out / "results_table.txt").write_text(table + "\n\n" + notes + "\n")
    (out / "results.json").write_text(json.dumps(results_json(study), indent=2) + "\n")
    write_per_run_csv(study, out / "per_run_errors.csv")

    plot_spectra(study, figures / "singular_values.png")
    plot_rank_sweeps(study, figures / "error_vs_rank.png")
    plot_size_sweep(study, figures / "error_vs_training_runs.png")
    for split in SPLITS:
        plot_profiles(study, split, figures / f"profiles_{split}.png")
    plot_extrapolation(study, figures / "extrapolation_error.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
