"""State of a fully charged cell at a given state of health.

Ageing is represented as loss of lithium inventory (LLI) to the SEI on the negative electrode, the dominant mode
for cells kept on float charge at moderate temperature. The lost lithium thickens the SEI (2 Li per SEI
molecule), which adds a film resistance in series. Other modes (loss of active material, particle cracking,
lithium plating, electrolyte loss) are not represented. Full charge means the open-circuit voltage of the new
cell, at the reference temperature.
"""
from dataclasses import dataclass

from scipy.optimize import brentq

from . import params as p


@dataclass(frozen=True)
class CellState:
    x0: float            # negative stoichiometry when fully charged
    y0: float            # positive stoichiometry when fully charged
    r_film: float        # SEI film resistance per interfacial area, Ohm m2
    lli: float           # fraction of cyclable lithium lost
    capacity_Ah: float   # low-rate capacity between full charge and V_MIN

    @property
    def r_contact(self):
        """The film resistance as a series resistance of the cell, Ohm (same voltage drop i_n r_film)."""
        return self.r_film / (p.A_N * p.L_N * p.AREA)


Q_N_SITES = p.C_N_MAX * p.EPS_ACT_N * p.L_N * p.AREA      # mol of sites, negative
Q_P_SITES = p.C_P_MAX * p.EPS_ACT_P * p.L_P * p.AREA      # mol of sites, positive
X_NEW, Y_NEW = p.C_N_FULL / p.C_N_MAX, p.C_P_FULL / p.C_P_MAX
N_LI_NEW = X_NEW * Q_N_SITES + Y_NEW * Q_P_SITES         # mol of cyclable lithium, new cell
V_FULL = float(p.ocp_p(Y_NEW) - p.ocp_n(X_NEW))          # open-circuit voltage when fully charged


def balance(n_li, v):
    """Stoichiometries (x, y) with x Q_n + y Q_p = n_li and open-circuit voltage v at T_ref."""
    def f(x):
        return float(p.ocp_p((n_li - x * Q_N_SITES) / Q_P_SITES) - p.ocp_n(x)) - v
    lo = max(1e-6, (n_li - (1 - 1e-6) * Q_P_SITES) / Q_N_SITES)
    hi = min(1 - 1e-6, (n_li - 1e-6 * Q_P_SITES) / Q_N_SITES)
    x = brentq(f, lo, hi, xtol=1e-14)
    return x, (n_li - x * Q_N_SITES) / Q_P_SITES


def low_rate_capacity(lli):
    n_li = N_LI_NEW * (1.0 - lli)
    x_full, _ = balance(n_li, V_FULL)
    x_empty, _ = balance(n_li, p.V_MIN)
    return (x_full - x_empty) * Q_N_SITES * p.F / 3600.0


CAPACITY_NEW = low_rate_capacity(0.0)
_STATES = {}


def cell_state(soh=1.0):
    """Fully charged cell whose low-rate capacity is `soh` times that of the new cell."""
    soh = float(soh)
    if soh in _STATES:
        return _STATES[soh]
    if soh >= 1.0:
        lli = 0.0
    else:
        lli = brentq(lambda z: low_rate_capacity(z) / CAPACITY_NEW - soh, 0.0, 0.6, xtol=1e-12)
    n_li = N_LI_NEW * (1.0 - lli)
    x0, y0 = balance(n_li, V_FULL)
    surface = p.A_N * p.L_N * p.AREA
    thickness = p.SEI_THICKNESS_0 + (N_LI_NEW * lli / p.SEI_LI_PER_MOL) * p.SEI_MOLAR_VOLUME / surface
    _STATES[soh] = CellState(x0, y0, thickness * p.SEI_RESISTIVITY, lli, low_rate_capacity(lli))
    return _STATES[soh]
