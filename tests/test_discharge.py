"""The reduced-order discharge model: conservation, limits and convergence."""
import numpy as np
import pytest

from battery_prediction import ageing
from battery_prediction import discharge as dc
from battery_prediction import params as p


def test_charge_passed_equals_lithium_moved():
    r = dc.simulate(288.15, 0.9, power=14.0, model="SPMe", thermal="lumped", dt=2.0, n_r=20)
    st = ageing.cell_state(0.9)
    moved_n = (st.x0 - r["x_mean_end"][0]) * ageing.Q_N_SITES * p.F / 3600
    moved_p = (r["y_mean_end"][0] - st.y0) * ageing.Q_P_SITES * p.F / 3600
    # the last, partial step is counted in charge_Ah but the particles advance a full step
    assert abs(moved_n - r["charge_Ah"][0]) < 2.0 * 14.0 / 2.5 / 3600 + 1e-9
    assert abs(moved_n - moved_p) < 1e-9
    assert abs(r["salt_end"][0] / r["salt_start"] - 1) < 1e-10      # round-off over ~3000 steps


def test_state_of_health_is_reproduced():
    for soh in (0.95, 0.8, 0.7):
        assert abs(ageing.cell_state(soh).capacity_Ah / ageing.CAPACITY_NEW - soh) < 1e-9


def test_low_rate_discharge_delivers_the_thermodynamic_capacity():
    r = dc.simulate(298.15, 1.0, current=0.1, model="SPMe", thermal="isothermal", dt=60.0, n_r=20)
    assert abs(r["charge_Ah"][0] / ageing.CAPACITY_NEW - 1.0) < 0.01


def test_constant_power_is_held():
    r = dc.simulate([273.15, 318.15], 1.0, power=[27.0, 13.5], dt=1.0, return_traces=True)
    P = r["I"] * r["V"]
    live = np.isfinite(P) & (r["t"][:, None] > 0) & (r["t"][:, None] < r["runtime_s"][None, :] - 1)
    assert np.max(np.abs(P[live[:, 0], 0] / 27.0 - 1)) < 1e-6
    assert np.max(np.abs(P[live[:, 1], 1] / 13.5 - 1)) < 1e-6


def test_spme_reduces_to_spm_without_electrolyte_and_solid_resistance():
    big = 1e6
    q = p.variant(kappa_e=lambda c, T: big + 0.0 * np.asarray(c, float), D_e=lambda c, T: 1e-3 + 0.0 * np.asarray(c,
                                                                                                                  float),
                  SIGMA_N=big, sigma_p=lambda T: big + 0.0 * np.asarray(T, float))
    a = dc.simulate(273.15, 0.9, power=18.0, model="SPMe", thermal="lumped", prm=q)["runtime_s"][0]
    b = dc.simulate(273.15, 0.9, power=18.0, model="SPM", thermal="lumped")["runtime_s"][0]
    assert abs(a / b - 1) < 2e-4, (a, b)


def test_strong_cooling_approaches_the_isothermal_model():
    a = dc.simulate(273.15, 1.0, power=18.0, thermal="lumped", h=1e5)
    b = dc.simulate(273.15, 1.0, power=18.0, thermal="isothermal")
    assert abs(a["runtime_s"][0] / b["runtime_s"][0] - 1) < 2e-3
    assert a["T_max_K"][0] - 273.15 < 0.1


def test_weaker_cooling_runs_hotter_and_longer_in_the_cold():
    r = dc.simulate(273.15, 1.0, power=18.0, thermal="lumped", h=np.array([2.0, 10.0, 50.0]))
    assert np.all(np.diff(r["T_max_K"]) < 0) and np.all(np.diff(r["runtime_s"]) < 0)


def test_grid_and_time_step_convergence_in_a_cold_heavy_case():
    kw = dict(T_amb=273.15, soh=0.8, power=27.0, model="SPMe", thermal="lumped")
    ref = dc.simulate(**kw, dt=0.25, n_r=60, n_x=(40, 20, 40))["runtime_s"][0]
    for opts in (dict(dt=1.0, n_r=30, n_x=(20, 10, 20)), dict(dt=0.25, n_r=30, n_x=(20, 10, 20)),
                 dict(dt=1.0, n_r=60, n_x=(40, 20, 40))):
        assert abs(dc.simulate(**kw, **opts)["runtime_s"][0] / ref - 1) < 3e-3, opts


def test_input_checks():
    with pytest.raises(ValueError):
        dc.simulate(298.15, power=1.0, current=1.0)
    with pytest.raises(ValueError):
        dc.simulate(298.15, power=1.0, model="DFN")
    with pytest.raises(ValueError):          # the cell state of ageing.py would not see a changed geometry
        dc.simulate(298.15, power=1.0, prm=p.variant(L_N=1.1 * p.L_N))


def test_parameter_variant_reaches_the_particles():
    # a replaced particle radius reaches the particle grids (only the grids: derived constants such as the
    # interfacial area A_N are not recomputed by params.variant); an empty variant changes nothing
    kw = dict(T_amb=273.15, soh=1.0, current=10.0, model="SPM", thermal="isothermal", dt=2.0, n_r=20)
    base = dc.simulate(**kw)["runtime_s"][0]
    assert dc.simulate(**kw, prm=p.variant())["runtime_s"][0] == base
    assert abs(dc.simulate(**kw, prm=p.variant(R_N=2.0 * p.R_N))["runtime_s"][0] / base - 1) > 0.02
