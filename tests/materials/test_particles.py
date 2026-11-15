import casadi as ca
import numpy as np
import pytest
from scipy.integrate import quad

from pcsp.materials import CARBO_HSP_40_70, SILICA_SAND, ParticleMaterial


def test_carbo_cp_follows_sandia_power_law():
    # cp = 365 * T_C^0.18 -> at 700 degC: 365 * 700**0.18
    assert CARBO_HSP_40_70.cp(973.15) == pytest.approx(365.0 * 700.0**0.18)
    # Sanity: roughly 1.1-1.3 kJ/kg/K across receiver operating temperatures.
    cp = CARBO_HSP_40_70.cp(np.array([573.15, 1073.15]))
    assert np.all((cp > 1000.0) & (cp < 1300.0))


@pytest.mark.parametrize("material", [CARBO_HSP_40_70, SILICA_SAND])
def test_enthalpy_is_the_integral_of_cp(material):
    T1, T2 = 400.0, 1200.0
    integral, _ = quad(material.cp, T1, T2)
    assert material.enthalpy(T2) - material.enthalpy(T1) == pytest.approx(integral, rel=1e-10)
    assert material.enthalpy(298.15) == pytest.approx(0.0, abs=1e-9)


def test_constant_cp_keeps_array_shape():
    T = np.array([300.0, 600.0, 900.0])
    np.testing.assert_allclose(SILICA_SAND.cp(T), [1000.0, 1000.0, 1000.0])


def test_bulk_density_and_void_fraction():
    assert CARBO_HSP_40_70.rho_bulk == pytest.approx(3300.0 * 0.606)
    assert CARBO_HSP_40_70.void_fraction == pytest.approx(0.394)


def test_properties_accept_casadi_symbols():
    T = ca.SX.sym("T")
    f = ca.Function("f", [T], [CARBO_HSP_40_70.cp(T), CARBO_HSP_40_70.enthalpy(T)])
    cp, h = (float(v) for v in f(1000.0))
    assert cp == pytest.approx(CARBO_HSP_40_70.cp(1000.0))
    assert h == pytest.approx(CARBO_HSP_40_70.enthalpy(1000.0))


def test_with_replaces_fields_and_validates():
    finer = CARBO_HSP_40_70.with_(d_p=200e-6)
    assert finer.d_p == 200e-6 and finer.rho_p == CARBO_HSP_40_70.rho_p
    with pytest.raises(ValueError, match="emissivity"):
        CARBO_HSP_40_70.with_(emissivity=1.2)
    with pytest.raises(ValueError, match="d_p"):
        ParticleMaterial("bad", 3000.0, -1.0, 2.0, 0.9, 0.9, 1000.0)
