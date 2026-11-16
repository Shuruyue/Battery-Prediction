"""Every parameter function and constant against PyBaMM's ORegan2022 set (skipped without PyBaMM)."""
import pytest

from ups_battery_prediction import params as p


@pytest.fixture(scope="module")
def pv():
    pybamm = pytest.importorskip("pybamm")
    return pybamm.ParameterValues("ORegan2022")


def _ev(pv, sym):
    return float(pv.process_symbol(sym).evaluate())


def test_functions_match_pybamm(pv):
    import pybamm
    Sc = pybamm.Scalar
    for T in (263.15, 273.15, 298.15, 318.15):
        for s in (0.02, 0.3, 0.6, 0.95):
            pairs = [
                (p.D_n(s, T), pv["Negative particle diffusivity [m2.s-1]"](Sc(s), Sc(T))),
                (p.D_p(s, T), pv["Positive particle diffusivity [m2.s-1]"](Sc(s), Sc(T))),
                (p.ocp_n_ref(s), pv["Negative electrode OCP [V]"](Sc(s))),
                (p.ocp_p_ref(s), pv["Positive electrode OCP [V]"](Sc(s))),
                (p.dUdT_n(s), pv["Negative electrode OCP entropic change [V.K-1]"](Sc(s))),
                (p.dUdT_p(s), pv["Positive electrode OCP entropic change [V.K-1]"](Sc(s))),
                (p.j0_n(s, T, 800.0), pv["Negative electrode exchange-current density [A.m-2]"](
                    Sc(800.0), Sc(s * p.C_N_MAX), Sc(p.C_N_MAX), Sc(T))),
                (p.j0_p(s, T, 800.0), pv["Positive electrode exchange-current density [A.m-2]"](
                    Sc(800.0), Sc(s * p.C_P_MAX), Sc(p.C_P_MAX), Sc(T))),
                (p.sigma_p(T), pv["Positive electrode conductivity [S.m-1]"](Sc(s), Sc(T))),
            ]
            for a, b in pairs:
                assert a == pytest.approx(_ev(pv, b), rel=1e-9, abs=1e-15)
        for c in (300.0, 1000.0, 2500.0):
            pairs = [(p.kappa_e(c, T), pv["Electrolyte conductivity [S.m-1]"](Sc(c), Sc(T))),
                     (p.D_e(c, T), pv["Electrolyte diffusivity [m2.s-1]"](Sc(c), Sc(T))),
                     (p.t_plus(c, T), pv["Cation transference number"](Sc(c), Sc(T))),
                     (p.thermodynamic_factor(c, T), pv["Thermodynamic factor"](Sc(c), Sc(T)))]
            for a, b in pairs:
                assert a == pytest.approx(_ev(pv, b), rel=1e-9)
        for name, f in (("Negative electrode", p.cp_n), ("Positive electrode", p.cp_p), ("Separator", p.cp_s),
                        ("Negative current collector", p.cp_cu), ("Positive current collector", p.cp_al)):
            assert f(T) == pytest.approx(_ev(pv, pv[f"{name} specific heat capacity [J.kg-1.K-1]"](Sc(T))), rel=1e-12)


def test_ocp_with_temperature_matches_pybamm(pv):
    import pybamm
    param = pybamm.LithiumIonParameters()
    for T in (263.15, 298.15, 318.15):
        for s in (0.0005, 0.05, 0.5, 0.97):
            assert p.ocp_n(s, T) == pytest.approx(_ev(pv, param.n.prim.U(pybamm.Scalar(s), pybamm.Scalar(T))),
                                                  abs=1e-12)
            assert p.ocp_p(s, T) == pytest.approx(_ev(pv, param.p.prim.U(pybamm.Scalar(s), pybamm.Scalar(T))),
                                                  abs=1e-12)


def test_constants_match_pybamm(pv):
    constants = {
        "Negative electrode thickness [m]": p.L_N, "Separator thickness [m]": p.L_S,
        "Positive electrode thickness [m]": p.L_P, "Negative current collector thickness [m]": p.L_CC_N,
        "Positive current collector thickness [m]": p.L_CC_P, "Negative particle radius [m]": p.R_N,
        "Positive particle radius [m]": p.R_P, "Maximum concentration in negative electrode [mol.m-3]": p.C_N_MAX,
        "Maximum concentration in positive electrode [mol.m-3]": p.C_P_MAX,
        "Initial concentration in negative electrode [mol.m-3]": p.C_N_FULL,
        "Initial concentration in positive electrode [mol.m-3]": p.C_P_FULL,
        "Initial concentration in electrolyte [mol.m-3]": p.C_E0,
        "Negative electrode active material volume fraction": p.EPS_ACT_N,
        "Positive electrode active material volume fraction": p.EPS_ACT_P,
        "Negative electrode porosity": p.EPS_E_N, "Separator porosity": p.EPS_E_S,
        "Positive electrode porosity": p.EPS_E_P, "Negative electrode conductivity [S.m-1]": p.SIGMA_N,
        "Negative electrode Bruggeman coefficient (electrolyte)": p.BRUG,
        "Separator Bruggeman coefficient (electrolyte)": p.BRUG,
        "Positive electrode Bruggeman coefficient (electrolyte)": p.BRUG,
        "Negative electrode Bruggeman coefficient (electrode)": 0.0,
        "Positive electrode Bruggeman coefficient (electrode)": 0.0,
        "Cell volume [m3]": p.CELL_VOLUME, "Cell cooling surface area [m2]": p.COOLING_AREA,
        "Total heat transfer coefficient [W.m-2.K-1]": p.H_COOL,
        "Negative current collector density [kg.m-3]": p.RHO_CC_N, "Negative electrode density [kg.m-3]": p.RHO_N,
        "Separator density [kg.m-3]": p.RHO_S, "Positive electrode density [kg.m-3]": p.RHO_P,
        "Positive current collector density [kg.m-3]": p.RHO_CC_P, "Reference temperature [K]": p.T_REF,
        "Lower voltage cut-off [V]": p.V_MIN, "Nominal cell capacity [A.h]": p.Q_NOMINAL,
        "Negative electrode charge transfer coefficient": 0.5, "Positive electrode charge transfer coefficient": 0.5,
    }
    for name, value in constants.items():
        assert pv[name] == value, name
    assert pv["Electrode height [m]"] * pv["Electrode width [m]"] == pytest.approx(p.AREA, rel=1e-15)


def test_heat_capacity_matches_pybamm_effective_value(pv):
    import pybamm
    therm = pybamm.LithiumIonParameters().therm
    for T in (263.15, 298.15, 330.0):
        assert p.rho_cp_eff(T) == pytest.approx(_ev(pv, therm.rho_c_p_eff(pybamm.Scalar(T))), rel=1e-12)


def test_variant_replaces_only_what_is_asked():
    q = p.variant(chi=lambda c, T: 1.0)
    assert q.chi(500.0, 300.0) == 1.0 and q.kappa_e is p.kappa_e and q.L_N == p.L_N
    with pytest.raises(KeyError):
        p.variant(not_a_parameter=1.0)
