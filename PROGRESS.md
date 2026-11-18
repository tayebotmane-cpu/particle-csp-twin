# Progress report

18 November 2026

## Purpose

This repository is a first, laptop-scale slice through the three tiers planned
for the PhD on hybrid physics-AI modelling and digital twins of particle-based
concentrating solar power (CSP):

1. physics models that produce data,
2. reduced-order models (ROMs) that compress it,
3. state estimation and control that use the fast models.

High-fidelity simulation (coupled computational fluid dynamics and discrete
element method, CFD-DEM) is not part of it yet. Reduced first-principles
models stand in for it, and results are stored in the data format that the
CFD-DEM results will use. The aim was to have the complete chain running and
tested before investing in the expensive tier.

`README.md` holds the technical description. This note summarises what
exists, what it shows, what it does not yet show, and what is proposed next.

## What is implemented

| Area | Content |
|---|---|
| Component models | Falling-particle receiver: steady curtain model after González-Portillo, Albrecht and Ho (2021), and a control-oriented dynamic model. Packed-bed thermocline storage: two-phase Schumann model. Moving packed-bed particle-to-sCO2 heat exchanger: reduced steady model, two-dimensional steady reference solution, and transient model. |
| Data | One dataset schema for every fidelity, with provenance; design of experiments; resumable simulation campaigns; splits by operating condition. |
| Reduced-order models | Proper orthogonal decomposition (POD), POD with interpolation over parameters, dynamic mode decomposition, operator inference. |
| Estimation and control | Ensemble Kalman filter (EnKF), Kalman and extended Kalman filters, disturbance estimation, PID, nonlinear model predictive control (NMPC), closed-loop simulator. |
| Software quality | 411 automated tests, linting, a continuous-integration workflow. |

## Results

**1. Surrogate of the thermocline storage.** A POD-interpolation surrogate
trained on 120 simulated charging runs reproduces the bed temperature field
with a median error of 0.2 % at about 1000 times the speed of the model. The
error is 0.6 % when extrapolating to mass flows above the training range.
Both figures require aligning the runs on the thermal front first; without
alignment the surrogate needs 25 modes instead of 7 and the errors are 2.3 %
and 9.0 %.

**2. Receiver outlet-temperature control.** The published Sandia test (power
step from 739 to 610 kW at a 700 degC setpoint, criterion: two standard
deviations of the outlet temperature within 10 K over 5 minutes) was run
against a simulated plant that differs from the controller's model. Means
over 20 noise seeds:

| Controller | Worst 5-min 2 sigma [K] | Seeds passing | Integral absolute error [K s] |
|---|---|---|---|
| PID | 9.7 | 95 % | 1536 |
| PID with ideal feedforward of the true power | 5.2 | 100 % | 864 |
| EnKF + NMPC, incident power estimated | 7.0 | 100 % | 1212 |
| EnKF + NMPC, also using a biased power signal | 6.0 | 100 % | 927 |
| NMPC with perfect model and information | 4.1 | 100 % | 220 |

The estimator-plus-NMPC scheme beats tuned PID by 21 % and stays within the
criterion when the plant's advective loss is up to twice the model's, where
PID fails from 1.5 times. Most of the benefit comes from knowing the
disturbance, not from the optimisation. A 30 % power step defeats both.

**3. Heat exchanger against the literature.**

- Design study of Albrecht and Ho (2019), with its printed inputs: the
  reduced model gives an overall coefficient of 165 W/m2/K and a converged
  two-dimensional solution of the paper's equations gives 160, where the
  paper reports 144. The paper's tables are reproduced to within 2 to 5 %
  only with a uniform-temperature Nusselt number, whereas the
  two-dimensional solution gives a Nusselt number near the uniform-flux
  value. Two closure constants that the paper does not print were inferred
  from its other numbers.
- Transient study of Fang et al. (2022), with the bed properties of Fang et
  al. (2021) and the plate properties of Fernández-Torrijos et al. (2018):
  five of the six published 95 % response times are reproduced within 10 %,
  the sixth within 17 %. No input was adjusted to obtain this.

## What is not yet established

- **Validation against measurements.** The receiver model has only been
  compared with the 2020 G3P3 on-sun test table (outlet temperature RMSE 9 K,
  efficiency RMSE 0.20), which is a weak test. The 26-case NSTTF validation
  set has not been used yet. The storage and heat exchanger models have not
  been compared with any measurement. For the heat exchanger, Sandia's 100 kW
  prototype measured 50 to 80 W/m2/K, 30 to 50 % below that group's own
  model, so a model of this kind should be expected to over-predict.
- **Control results rest on a simulated plant.** The receiver's time
  constants, the sensor noise and the actuator limits are assumptions.
- **Material data.** The particle heat capacity correlation in use differs
  from published measurements by 8 to 14 % at 700 degC, and several particle
  properties are single-source values.

## Points for discussion

1. **Licence and intellectual property.** The repository is public without a
   licence, so all rights are reserved until the university's policy and the
   preferred licence are settled.
2. **Validation data.** Access to the NSTTF 26-case set, and whether
   time-series data from a test facility or laboratory rig could be obtained.
3. **Computing.** Which cluster to use for the CFD-DEM work.
4. **Benchmark for physics-informed neural networks.** The 2026 sand-bed
   study pencilled in for this may describe a different configuration from
   a gas-heated packed bed; this needs checking before it is relied on.
5. **Priorities** among the steps below.

## Proposed next steps

1. Validate the receiver model against the NSTTF 26-case set.
2. Compare estimator-plus-NMPC control of the heat exchanger with the
   published schemes on the same disturbance cases: feedback control within
   16 degC and 3 minutes settling (Fernández-Torrijos et al., 2018) and a
   neural-network scheme with deviations up to 32 K (Fang et al., 2022).
3. Couple the receiver, storage and heat exchanger into one plant model.
4. Start the CFD-DEM verification cases once computing is available.

## Note on tools

The code was written with the help of an AI coding assistant; the commit
history records this. Every result above can be re-run with the example
scripts described in `README.md`, and the models behind them are covered by
the test suite.
