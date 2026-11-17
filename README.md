# particle-csp-twin

Research code for a PhD project on hybrid physics-AI modelling and digital twins
of particle-based concentrating solar power (CSP) and thermal energy storage.

The project climbs a three-tier fidelity ladder: high-fidelity simulation
(CFD-DEM) produces data, reduced-order models and surrogates compress it, and a
digital twin uses the fast models for state estimation and control. This
repository currently holds a first slice through all three tiers that runs on a
laptop. Fast first-principles models stand in for the CFD-DEM data, which plugs
into the same data format later.

## What is in the repository

| Tier | Implemented | Not yet |
|---|---|---|
| Data | One dataset schema for every fidelity, provenance stamping, Zarr/netCDF storage, design of experiments, a resumable campaign runner, train/validation/test splits by operating condition | Parameter units in the schema |
| Physics models | Packed-bed thermal storage (two-phase Schumann model), moving packed-bed particle-to-sCO2 heat exchanger (steady, with a two-dimensional reference solution, and time-dependent), falling-particle receiver (steady curtain model and a control-oriented dynamic model) | CFD-DEM and CFD cases, containers, cluster job scripts |
| Reduced-order models | POD, POD with RBF interpolation over parameters, DMD and DMD with control, operator inference | Neural operators, PINNs, graph networks (`torch` is an optional extra and is not installed by default) |
| Digital twin | Ensemble Kalman filter (stochastic and square-root), Kalman and extended Kalman filters, disturbance augmentation for offset-free control, PID, nonlinear MPC, closed-loop simulator, regulation metrics | Plant-wide system model (Modelica or TESPy), coupling of the three components |

## Quickstart

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                        # create .venv with runtime and dev dependencies
uv run pytest                                  # test suite
uv run ruff check . && uv run ruff format --check .
uv run python examples/thermocline_rom.py      # storage campaign -> surrogate study, about 1 min
uv run python examples/hx_dynamics.py          # heat exchanger step responses, about 2 min
```

Examples write figures and tables to `outputs/`, which is git-ignored.

## Layout

```
src/pcsp/
  core/            NumPy/CasADi backend, model/estimator/controller protocols, RK4
  materials/       particles (CARBO HSP 40/70, silica sand), air, sCO2
  data/            schema, provenance, I/O, design of experiments, campaigns, splits
  models/
    storage/       packed bed and its analytic (Anzelius-Schumann) solution
    hx/            moving packed-bed heat exchanger
    receiver/      falling-particle receiver, with the on-sun test data used for validation
    correlations.py
  rom/             POD, interpolation, DMD, operator inference, error metrics
  twin/
    estimation/    EnKF, Kalman, EKF, augmented models, consistency diagnostics
    control/       PID, NMPC, closed-loop simulator, metrics, input profiles
  pipelines/       end-to-end studies assembled from the packages above
examples/          scripts that run a study and write its figures
tests/             mirrors src/pcsp
```

## Conventions

- SI units throughout, temperatures in kelvin. Every dataset variable carries
  `units` and `long_name`.
- Every model, at every fidelity, returns the same kind of `xarray.Dataset`,
  built with `pcsp.data.schema.new_dataset` and checked with `validate`.
  Parameters are stored as canonical JSON and hashed; the hash names the run on
  disk, so a campaign can be resumed and a result traced to its inputs.
- Data is split by run (operating condition), never by snapshot, so test data
  never shares a trajectory with training data. Use `pcsp.data.splits`.
- Snapshot matrices have shape `(n_dof, n_snapshots)`.
- A control-oriented model implements the `ControlModel` protocol and is
  written once for both NumPy and CasADi (`pcsp.core.backend.ops_for`). The same
  `rhs` is then simulated, differentiated and optimised.
- Estimators and controllers are written against the protocols in
  `pcsp.core.interfaces`, never against a concrete model, so a physics model
  can be replaced by a surrogate without touching the twin.

## Results so far

### Thermocline surrogate (`examples/thermocline_rom.py`)

120 packed-bed charging runs over mass flow, inlet temperature and particle
diameter train a POD-interpolation surrogate of the bed temperature field.
Variant A works in physical time; variant B first aligns the runs on the
thermal front (dimensionless time `t * u_front / L`).

| Test set | Variant | Modes | Median field error | Worst state-of-charge error | Speed-up over the model |
|---|---|---|---|---|---|
| Interpolation (24 random runs held out) | A | 25 | 2.3 % | 1.10 points | about 1000x |
| Interpolation (24 random runs held out) | B | 7 | 0.20 % | 0.03 points | about 1000x |
| Extrapolation (top quarter of mass flows held out) | A | 24 | 9.0 % | 16.4 points | about 1000x |
| Extrapolation (top quarter of mass flows held out) | B | 7 | 0.57 % | 0.18 points | about 1000x |

Aligning on the front cuts the number of modes by more than three and the error
by about ten. A moving front is the standard hard case for linear reduced
bases, and it is the reason the later tiers look at nonlinear surrogates.

### Receiver outlet-temperature control (`examples/receiver_control.py`)

A simulated receiver is held at a 700 degC outlet temperature while the
incident power steps from 739 to 610 kW and back (the published Sandia test).
The pass criterion is Sandia's: twice the standard deviation of the measured
outlet temperature over any 5 minutes must stay within 10 K. The simulated
plant differs from the controller's model in its heat losses (20 % higher),
its time constants and its resolution. Values are means over 20 noise seeds.

| Configuration | Worst 5-min 2 sigma [K] | Seeds passing | Integral absolute error [K s] |
|---|---|---|---|
| A: PID | 9.7 | 95 % | 1536 |
| B: PID with feedforward from the true power (idealised) | 5.2 | 100 % | 864 |
| B: PID with feedforward from a biased, lagged power signal | 8.7 | 100 % | 1635 |
| C: twin (EnKF estimating the power, NMPC) | 7.0 | 100 % | 1212 |
| D: twin that also fuses the power signal | 6.0 | 100 % | 927 |
| E: NMPC with perfect model and information (upper bound) | 4.1 | 100 % | 220 |

- The twin beats tuned PID by 21 % in integral error using only the outlet
  thermocouple and the mass-flow meter, and it keeps passing when the plant's
  heat losses are up to twice the model's. PID fails from about 1.5 times.
- Most of the benefit comes from knowing the disturbance, not from the
  optimisation: PID with a perfect power measurement beats the twin.
- A 30 % power step defeats both PID and the twin.
- The gap between the twin and the upper bound is estimation lag and noise.

This is a simulation against a virtual plant. The time constants, sensor noise
levels and actuator limits are assumptions, not measurements. The full study
takes about two hours on four cores; `--seeds 5 --sensitivity-seeds 3` gives a
shorter version.

### Heat exchanger dynamics (`examples/hx_dynamics.py`)

The time-dependent heat exchanger model is compared with the response times
published by Fang et al. (2022, Energy 256, 124597): the time for an outlet
temperature to complete 95 % of its change after a step at one inlet. The
bed conductivity and heat capacity are the fits printed in Fang et al. (2021,
Energy 217, 119389); the plate properties are those of Fernandez-Torrijos et
al. (2018, Applied Energy 226, 595), against which that paper validated its
transients. The particle diameter (250 um) remains an assumption. Times are
from an exponential fit, as in the paper; steps keep the inlet velocity fixed,
as the paper specifies.

| Step in | Outlet | Published [s] | This model [s] | Difference |
|---|---|---|---|---|
| Particle inlet temperature | sCO2 | 338 | 368 | +9 % |
| Particle inlet temperature | Particles | 969 | 975 | +1 % |
| sCO2 inlet temperature | sCO2 | 560 | 657 | +17 % |
| sCO2 inlet temperature | Particles | 461 | 424 | -8 % |
| sCO2 flow | sCO2 | 367 | 403 | +10 % |
| sCO2 flow | Particles | 979 | 878 | -10 % |

Five of the six are within 10 %. No input was adjusted to obtain this. The
steady outlet temperatures are 4 to 6 K below the paper's (961 and 844 K
against about 965 and 850 K). A likely contributor to what remains is the
paper's treatment of the sCO2 flow: it solves the velocity field with
constant properties and the energy equation with local ones, which this
model, conserving mass, does not imitate. With the inputs assumed before the
two source papers were available, three times agreed within 4 % and three
were 16 to 32 % off.

## Verification and validation status

Each model is verified against analytic results and conservation laws.
Validation against experiments is incomplete, and the gaps are listed here so
that no result is read as stronger than it is.

| Model | Verified against | Validation gap |
|---|---|---|
| Packed-bed storage | Anzelius-Schumann analytic solution (second-order convergence, normalised temperature error below 5e-5 at 200 cells), energy conservation, charge/discharge symmetry | No experimental comparison yet |
| Moving-bed heat exchanger | Graetz series and its limits, energy balance between the two streams, sCO2 properties against CoolProp, and an independent two-dimensional solution of the same equations (reduced model within 3 %) | Against the design study of Albrecht and Ho (2019) with its own inputs: 165 W/m2/K where the paper reports 144; the two-dimensional solution gives 160. The paper's tables are reproduced to within 2 to 5 % with a uniform-temperature wall condition. Two closure constants are not printed in the paper and were inferred from its numbers. No comparison with measurements yet: Sandia's 100 kW prototype measured 50 to 80 W/m2/K, 30 to 50 % below that group's own detailed model (Albrecht et al., 2020). The time-dependent model converges to the two-dimensional steady solution and conserves energy; its response times are within 10 % of a published simulation for five of six cases (see above). No measured response times have been used |
| Falling-particle receiver | Energy budget closure, agreement of the NumPy and CasADi evaluations | Compared with 39 of 47 G3P3 on-sun tests (Ho et al., 2020), which used a different receiver configuration and do not tabulate incident power: outlet temperature RMSE 9 K with the nominal loss coefficient, thermal efficiency RMSE 0.20. Fitting the loss coefficient on half the cases does not improve the other half. The 26-case NSTTF validation set has not been used. The dynamic time constants are assumed, not identified |

Particle properties also need checking against primary sources: the CARBO HSP
heat-capacity correlation in `materials/particles.py` differs from published
measurements by roughly 8 to 14 % at 700 degC, and the particle diameter,
density and emissivity are single-source values.

## Development

- `ruff` for linting and formatting (line length 100), `pytest` for tests; a
  pre-commit configuration is provided.
- A GitHub Actions workflow is included but has never run, because the
  repository has not been pushed.

## Licence

No licence has been chosen yet. Until one is, the code is not available for
reuse or redistribution.
