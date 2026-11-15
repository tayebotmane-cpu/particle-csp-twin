"""Shared fixtures for the packed-bed storage tests."""

import pytest

from pcsp.materials.particles import ParticleMaterial
from pcsp.models.storage import PackedBedParams

# Illustrative rock-like solid for numerical verification only. The numbers are
# round, order-of-magnitude values for crushed rock, not a characterised material.
ROCK = ParticleMaterial(
    name="illustrative rock (verification only)",
    rho_p=2650.0,
    d_p=0.02,
    k_s=2.5,
    emissivity=0.9,
    solar_absorptivity=0.9,
    cp_a=900.0,
    cp_b=0.0,
    packing_fraction=0.6,
    source="round illustrative values; not a real material",
)


@pytest.fixture
def rock() -> ParticleMaterial:
    return ROCK


@pytest.fixture
def bed(rock) -> PackedBedParams:
    """1 m x 1 m bed, 0.4 kg/s of air, 298 -> 873 K; thermal front speed ~3.7e-4 m/s."""
    return PackedBedParams(
        material=rock,
        length=1.0,
        diameter=1.0,
        T_hot=873.15,
        T_cold=298.15,
        mass_flow=0.4,
        duration=2400.0,
        n_cells=100,
        n_output=25,
    )
