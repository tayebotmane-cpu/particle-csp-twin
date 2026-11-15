"""Textbook closures shared by more than one component model.

Only correlations used by several components live here. Component-specific
closures (near-wall resistance of a moving bed, curtain opacity, ...) belong in
that component's own package.

All functions are backend-agnostic unless stated otherwise. Reynolds numbers
are based on the *superficial* velocity and the particle diameter:

.. math::  Re_p = \\rho_f \\, u_s \\, d_p / \\mu_f
"""

from __future__ import annotations

from typing import Any

from pcsp.core.backend import ops_for


def reynolds_particle(u_s: Any, d_p: Any, rho_f: Any, mu_f: Any) -> Any:
    """Particle Reynolds number from the superficial velocity [-]."""
    return rho_f * u_s * d_p / mu_f


def ergun_pressure_gradient(u_s: Any, d_p: Any, eps: Any, rho_f: Any, mu_f: Any) -> Any:
    """Pressure drop per unit bed length [Pa m^-1] (Ergun, 1952).

    .. math::
        \\frac{\\Delta p}{L} = 150 \\frac{\\mu_f (1-\\varepsilon)^2}{\\varepsilon^3 d_p^2} u_s
                             + 1.75 \\frac{\\rho_f (1-\\varepsilon)}{\\varepsilon^3 d_p} u_s^2

    Parameters
    ----------
    u_s : superficial velocity [m s^-1] (must be non-negative)
    d_p : particle diameter [m]
    eps : bed void fraction [-]
    rho_f, mu_f : fluid density [kg m^-3] and dynamic viscosity [Pa s]
    """
    viscous = 150.0 * mu_f * (1.0 - eps) ** 2 * u_s / (eps**3 * d_p**2)
    inertial = 1.75 * rho_f * (1.0 - eps) * u_s**2 / (eps**3 * d_p)
    return viscous + inertial


def nu_ranz_marshall(Re: Any, Pr: Any) -> Any:
    """Nusselt number of an isolated sphere (Ranz & Marshall, 1952).

    ``Nu = 2 + 0.6 Re^(1/2) Pr^(1/3)``. Use for dilute particles, e.g. a
    falling curtain; not for packed beds.
    """
    return 2.0 + 0.6 * Re**0.5 * Pr ** (1.0 / 3.0)


def nu_wakao_kaguei(Re: Any, Pr: Any) -> Any:
    """Fluid-particle Nusselt number in a packed bed (Wakao & Kaguei, 1982).

    ``Nu = 2 + 1.1 Re^0.6 Pr^(1/3)``, correlated for roughly 15 < Re < 8500.
    """
    return 2.0 + 1.1 * Re**0.6 * Pr ** (1.0 / 3.0)


def nu_gunn(Re: Any, Pr: Any, eps: Any) -> Any:
    """Fluid-particle Nusselt number for fixed and fluidized beds (Gunn, 1978).

    .. math::
        Nu = (7 - 10\\varepsilon + 5\\varepsilon^2)(1 + 0.7 Re^{0.2} Pr^{1/3})
             + (1.33 - 2.4\\varepsilon + 1.2\\varepsilon^2) Re^{0.7} Pr^{1/3}

    Valid for void fractions 0.35-1 and Re up to 1e5. Recovers ``Nu = 2`` for
    a single sphere in stagnant fluid (``eps = 1``, ``Re = 0``).
    """
    pr13 = Pr ** (1.0 / 3.0)
    return (7.0 - 10.0 * eps + 5.0 * eps**2) * (1.0 + 0.7 * Re**0.2 * pr13) + (
        1.33 - 2.4 * eps + 1.2 * eps**2
    ) * Re**0.7 * pr13


def k_eff_zehner_schlunder(k_f: Any, k_s: Any, eps: Any) -> Any:
    """Stagnant effective conductivity of a packed bed of spheres [W m^-1 K^-1].

    Zehner & Schlunder (1970) unit-cell model *without* the radiation and
    contact-area extensions of Bauer & Schlunder. With
    :math:`\\kappa = k_s / k_f` and the sphere deformation parameter
    :math:`B = 1.25 \\, ((1-\\varepsilon)/\\varepsilon)^{10/9}`:

    .. math::
        \\frac{k_{eff}}{k_f} = 1 - \\sqrt{1-\\varepsilon}
          + \\frac{2\\sqrt{1-\\varepsilon}}{1 - B/\\kappa}
            \\left[ \\frac{(1 - 1/\\kappa) B}{(1 - B/\\kappa)^2} \\ln\\frac{\\kappa}{B}
                   - \\frac{B+1}{2} - \\frac{B-1}{1 - B/\\kappa} \\right]

    The expression is singular where :math:`\\kappa = B`; that cannot occur for
    gas-solid beds (:math:`\\kappa \\gg B`).
    """
    xp = ops_for(k_f, k_s, eps)
    kappa = k_s / k_f
    B = 1.25 * ((1.0 - eps) / eps) ** (10.0 / 9.0)
    n = 1.0 - B / kappa
    root = xp.sqrt(1.0 - eps)
    bracket = (1.0 - 1.0 / kappa) * B / n**2 * xp.log(kappa / B) - (B + 1.0) / 2.0 - (B - 1.0) / n
    return k_f * (1.0 - root + 2.0 * root / n * bracket)
