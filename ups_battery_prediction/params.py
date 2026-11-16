"""Parameters of the LG M50 cell (21700, NMC811 / graphite-SiOx, 5 A h).

Every value and fit is transcribed from the published parameter set and checked function by function
against PyBaMM's ``ORegan2022`` set (tests/test_params.py):

* K. O'Regan, F. Brosa Planella, W. D. Widanage, E. Kendrick, "Thermal-electrochemical parameters of a high
  energy lithium-ion cylindrical battery", Electrochim. Acta 425 (2022) 140700,
  doi:10.1016/j.electacta.2022.140700 (CC BY 4.0): solid diffusivities, exchange currents, positive-electrode
  conductivity, entropic coefficients, heat capacities.
* C.-H. Chen et al., J. Electrochem. Soc. 167 (2020) 080534, doi:10.1149/1945-7111/ab9050 (CC BY 4.0):
  open-circuit potentials and geometry.
* J. Landesfeind, H. A. Gasteiger, J. Electrochem. Soc. 166 (2019) A3079, doi:10.1149/2.0571912jes:
  LiPF6 in EC:EMC 3:7 (conductivity, diffusivity, transference number, thermodynamic factor).
* Copper and aluminium heat capacities: CRC Handbook fits, as in PyBaMM.
* SEI film (molar volume, resistivity, Li per SEI): values of PyBaMM's Chen2020 set (after Safari et al.,
  J. Electrochem. Soc. 156 (2009) A145); used only for aged cells.

Stoichiometries are x (negative, Li_x C6) and y (positive, Li_y NMC). Temperatures in K, SI units throughout.
"""
import sys
import types

import numpy as np

F = 96485.33212      # C/mol
R = 8.314462618      # J/(mol K)
T_REF = 298.15       # reference temperature of all fits, K

# geometry ------------------------------------------------------------------------------------------------------
L_N, L_S, L_P = 8.52e-5, 1.2e-5, 7.56e-5          # electrode / separator thickness, m
L_CC_N, L_CC_P = 1.2e-5, 1.6e-5                   # current-collector thickness, m
L_X = L_N + L_S + L_P                             # electrode stack, m
AREA = 0.065 * 1.58                               # electrode area (height x width), m2
R_N, R_P = 5.86e-6, 5.22e-6                       # particle radius, m
EPS_ACT_N, EPS_ACT_P = 0.75, 0.665                # active-material volume fraction
EPS_E_N, EPS_E_S, EPS_E_P = 0.25, 0.47, 0.335     # porosity
BRUG = 1.5                                        # Bruggeman exponent of the electrolyte (solid: 0)
A_N = 3.0 * EPS_ACT_N / R_N                       # interfacial area per volume, 1/m
A_P = 3.0 * EPS_ACT_P / R_P

C_N_MAX, C_P_MAX = 29583.0, 51765.0               # mol/m3
C_N_FULL, C_P_FULL = 28866.0, 13975.0             # fully charged new cell, mol/m3
C_E0 = 1000.0                                     # electrolyte concentration at rest, mol/m3
SIGMA_N = 215.0                                   # S/m
V_MIN = 2.5                                       # lower cut-off voltage, V
Q_NOMINAL = 5.0                                   # A h

# thermal (lumped cell) -----------------------------------------------------------------------------------------
CELL_VOLUME = 2.42e-5                             # m3
COOLING_AREA = 0.00531                            # m2
H_COOL = 10.0                                     # total heat-transfer coefficient, W/(m2 K)
RHO_CC_N, RHO_N, RHO_S, RHO_P, RHO_CC_P = 8933.0, 2060.0, 1548.0, 3699.0, 2702.0   # kg/m3

# SEI film (aged cells only)
SEI_MOLAR_VOLUME = 9.585e-5   # m3/mol
SEI_RESISTIVITY = 2.0e5       # Ohm m
SEI_LI_PER_MOL = 2.0
SEI_THICKNESS_0 = 5.0e-9      # m

# activation temperatures E/R of the solid diffusivities, K
D_N_THETA, D_P_THETA = 2092.0, 1449.0
_C_E_FLOOR = 10.0             # PyBaMM evaluates D_e and kappa_e at max(c_e, 10 mol/m3)
_STO_TOL = 1e-10              # PyBaMM clips the stoichiometry of the OCP to [tol, 1 - tol]


def arrhenius(E, T):
    """exp(E/R (1/T_ref - 1/T)) for an activation energy E in J/mol."""
    return np.exp(E / R * (1.0 / T_REF - 1.0 / np.asarray(T, float)))


# open-circuit potentials ---------------------------------------------------------------------------------------
def _asymptote(sto):
    """PyBaMM's OCP barrier: 1 mV at sto = 0.001, 1000 mV at sto = 0 (zero elsewhere to round-off)."""
    a, b, c = 205.0568621937484, 6910.192179565431, -7.7e-4
    z = np.minimum(-b * (sto - c), 700.0)
    return a * np.log1p(np.exp(z))


def ocp_n_ref(x):
    x = np.asarray(x, float)
    return (1.9793 * np.exp(-39.3631 * x) + 0.2482
            - 0.0909 * np.tanh(29.8538 * (x - 0.1234))
            - 0.04478 * np.tanh(14.9159 * (x - 0.2769))
            - 0.0205 * np.tanh(30.4444 * (x - 0.6103)))


def ocp_p_ref(y):
    y = np.asarray(y, float)
    return (-0.8090 * y + 4.4875
            - 0.0428 * np.tanh(18.5138 * (y - 0.5542))
            - 17.7326 * np.tanh(15.7890 * (y - 0.3117))
            + 17.5842 * np.tanh(15.9308 * (y - 0.3120)))


def dUdT_n(x):
    """Entropic coefficient of graphite-SiOx, V/K (O'Regan 2022; the tanh term is switched off there)."""
    x = np.asarray(x, float)
    return (-0.1112 * x + 0.02914 + 0.3561 * np.exp(-(x - 0.08309) ** 2 / 0.004616)) / 1000.0


def dUdT_p(y):
    """Entropic coefficient of NMC811, V/K (O'Regan 2022)."""
    y = np.asarray(y, float)
    return (0.04006 * np.exp(-(y - 0.2828) ** 2 / 0.0009855)
            - 0.06656 * np.exp(-(y - 0.8032) ** 2 / 0.02179)) / 1000.0


def ocp_n(x, T=T_REF):
    """U_n(x, T) = U_ref(x) + (T - T_ref) dU/dT(x), plus PyBaMM's barrier near x = 0 and 1."""
    x = np.clip(np.asarray(x, float), _STO_TOL, 1.0 - _STO_TOL)
    return ocp_n_ref(x) + (np.asarray(T, float) - T_REF) * dUdT_n(x) + _asymptote(x) - _asymptote(1.0 - x)


def ocp_p(y, T=T_REF):
    y = np.clip(np.asarray(y, float), _STO_TOL, 1.0 - _STO_TOL)
    return ocp_p_ref(y) + (np.asarray(T, float) - T_REF) * dUdT_p(y) + _asymptote(y) - _asymptote(1.0 - y)


# solid phase ---------------------------------------------------------------------------------------------------
def D_n_ref(x):
    """Graphite-SiOx solid diffusivity at T_ref, m2/s (O'Regan 2022, with its correction factor 3.0321)."""
    x = np.asarray(x, float)
    return 10.0 ** (11.17 * x - 15.11
                    - 1.553 * np.exp(-((x - 0.2031) ** 2) / 0.0006091)
                    - 6.136 * np.exp(-((x - 0.5375) ** 2) / 0.06438)
                    - 9.725 * np.exp(-((x - 0.9144) ** 2) / 0.0578)
                    + 1.85 * np.exp(-((x - 0.5953) ** 2) / 0.001356)) * 3.0321


def D_p_ref(y):
    """NMC811 solid diffusivity at T_ref, m2/s (O'Regan 2022, with its correction factor 2.7)."""
    y = np.asarray(y, float)
    return 10.0 ** (-13.96
                    - 0.9231 * np.exp(-((y - 0.3216) ** 2) / 0.002534)
                    - 0.4066 * np.exp(-((y - 0.4532) ** 2) / 0.003926)
                    - 0.993 * np.exp(-((y - 0.8098) ** 2) / 0.09924)) * 2.7


def D_n(x, T):
    return D_n_ref(x) * np.exp(D_N_THETA * (1.0 / T_REF - 1.0 / np.asarray(T, float)))


def D_p(y, T):
    return D_p_ref(y) * np.exp(D_P_THETA * (1.0 / T_REF - 1.0 / np.asarray(T, float)))


def j0_n(x_surf, T, c_e=C_E0):
    """Exchange-current density of the negative electrode, A/m2."""
    al = 0.792
    return (2.668 * arrhenius(4.0e4, T) * (np.asarray(c_e, float) / C_E0) ** (1 - al)
            * x_surf ** al * (1.0 - x_surf) ** (1 - al))


def j0_p(y_surf, T, c_e=C_E0):
    """Exchange-current density of the positive electrode, A/m2."""
    al = 0.43
    return (5.028 * arrhenius(2.401e4, T) * (np.asarray(c_e, float) / C_E0) ** (1 - al)
            * y_surf ** al * (1.0 - y_surf) ** (1 - al))


def sigma_p(T):
    """Electronic conductivity of the positive electrode, S/m (the electrode Bruggeman exponent is 0)."""
    return 0.8473 * arrhenius(3.5e3, T)


# electrolyte: LiPF6 in EC:EMC 3:7 (Landesfeind and Gasteiger 2019) --------------------------------------------
def kappa_e(c_e, T):
    """Ionic conductivity, S/m."""
    c = np.maximum(np.asarray(c_e, float), _C_E_FLOOR) / 1000.0
    T = np.asarray(T, float)
    p1, p2, p3, p4, p5, p6 = 5.21e-1, 2.28e2, -1.06, 3.53e-1, -3.59e-3, 1.48e-3
    A = p1 * (1.0 + (T - p2))
    B = 1.0 + p3 * np.sqrt(c) + p4 * (1.0 + p5 * np.exp(1000.0 / T)) * c
    C = 1.0 + c ** 4 * (p6 * np.exp(1000.0 / T))
    return A * c * B / C / 10.0


def D_e(c_e, T):
    """Salt diffusivity, m2/s."""
    c = np.maximum(np.asarray(c_e, float), _C_E_FLOOR) / 1000.0
    T = np.asarray(T, float)
    return 1010.0 * np.exp(1.01 * c) * np.exp(-1560.0 / T) * np.exp(-487.0 * c / T) * 1e-10


def _poly9(c_e, T, p):
    c = np.asarray(c_e, float) / 1000.0
    T = np.asarray(T, float)
    return (p[0] + p[1] * c + p[2] * T + p[3] * c ** 2 + p[4] * c * T + p[5] * T ** 2
            + p[6] * c ** 3 + p[7] * c ** 2 * T + p[8] * c * T ** 2)


def t_plus(c_e, T):
    """Cation transference number."""
    return _poly9(c_e, T, (-12.8, -6.12, 0.0821, 0.904, 0.0318, -0.000127, 0.0175, -0.00312, -3.96e-05))


def thermodynamic_factor(c_e, T):
    """1 + d ln f / d ln c."""
    return _poly9(c_e, T, (25.7, -45.1, -0.177, 1.94, 0.295, 0.000308, 0.259, -0.00946, -0.000454))


def chi(c_e, T):
    """Diffusion-potential factor of the Stefan-Maxwell form, 2 (1 - t+) (1 + d ln f / d ln c)."""
    return 2.0 * (1.0 - t_plus(c_e, T)) * thermodynamic_factor(c_e, T)


# heat capacities -----------------------------------------------------------------------------------------------
def _wet(cp_dry, rho_dry, eps):
    """Specific heat of an electrode or separator soaked with electrolyte (rho 1280 kg/m3, cp 229 J/(kg K))."""
    rho_e, cp_e = 1280.0, 229.0
    return (rho_dry * cp_dry + rho_e * cp_e * eps) / (rho_dry + rho_e * eps)


def cp_n(T):
    T = np.asarray(T, float)
    return _wet(0.0004932 * T ** 3 - 0.491 * T ** 2 + 169.4 * T - 18970.0, 1740.0, EPS_E_N)


def cp_p(T):
    T = np.asarray(T, float)
    return _wet(-0.0008414 * T ** 3 + 0.7892 * T ** 2 - 241.3 * T + 25080.0, 3270.0, EPS_E_P)


def cp_s(T):
    T = np.asarray(T, float)
    return _wet(0.001494 * T ** 3 - 1.444 * T ** 2 + 475.5 * T - 51300.0, 946.0, EPS_E_S)


def cp_cu(T):
    T = np.asarray(T, float)
    return 1.445e-06 * T ** 3 - 0.001946 * T ** 2 + 0.9633 * T + 236.0


def cp_al(T):
    T = np.asarray(T, float)
    return 4.503e-06 * T ** 3 - 0.006256 * T ** 2 + 3.281 * T + 355.7


def rho_cp_eff(T):
    """Thickness-weighted volumetric heat capacity of the stack (current collectors included), J/(m3 K)."""
    return (RHO_CC_N * cp_cu(T) * L_CC_N + RHO_N * cp_n(T) * L_N + RHO_S * cp_s(T) * L_S
            + RHO_P * cp_p(T) * L_P + RHO_CC_P * cp_al(T) * L_CC_P) / (L_CC_N + L_X + L_CC_P)


def heat_capacity(T):
    """Heat capacity of the lumped cell, J/K: rho_cp_eff times the cell volume (PyBaMM's lumped model)."""
    return rho_cp_eff(T) * CELL_VOLUME


def variant(**overrides):
    """A copy of this parameter set with some functions or constants replaced, e.g.
    variant(chi=lambda c, T: chi(1000.0, T)). Used by the code checks against PyBaMM."""
    module = sys.modules[__name__]
    ns = types.SimpleNamespace(**{k: getattr(module, k) for k in dir(module) if not k.startswith("__")})
    for key, value in overrides.items():
        if not hasattr(ns, key):
            raise KeyError(key)
        setattr(ns, key, value)
    return ns
