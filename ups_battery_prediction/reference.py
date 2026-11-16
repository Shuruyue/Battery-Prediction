"""The reference ("reality"): PyBaMM's Doyle-Fuller-Newman model with a lumped thermal model.

Same parameter set as the hand-written model (O'Regan 2022): the cell starts at the ambient temperature and
discharges at constant power, cooled by h = 10 W/(m2 K) over its surface. It plays the part of the real battery:
the training 'measurements' of the machine-learning tools come from it, and every predictor is scored against it.
The operating points are those of design.py.
"""
import numpy as np

from . import params as p
from .ageing import cell_state
from .design import P_RATED

MESH = dict(x_n=20, x_s=10, x_p=20, r_n=160, r_p=160)   # chosen by scripts/audit_reference_mesh.py
THERMAL = "lumped"


def build_simulation(mesh=None, thermal=THERMAL, h=p.H_COOL, rtol=1e-7):
    """A PyBaMM DFN simulation whose operating point is given by inputs (see run_point)."""
    import pybamm
    pybamm.set_logging_level("ERROR")
    options = {"operating mode": "power", "contact resistance": "true", "thermal": thermal}
    pv = pybamm.ParameterValues("ORegan2022")
    pv.update({"Ambient temperature [K]": "[input]", "Initial temperature [K]": "[input]",
               "Power function [W]": "[input]", "Contact resistance [Ohm]": "[input]",
               "Initial concentration in negative electrode [mol.m-3]": "[input]",
               "Initial concentration in positive electrode [mol.m-3]": "[input]",
               "Total heat transfer coefficient [W.m-2.K-1]": h}, check_already_exists=False)
    return pybamm.Simulation(pybamm.lithium_ion.DFN(options=options), parameter_values=pv,
                             var_pts=dict(mesh or MESH), solver=pybamm.IDAKLUSolver(rtol=rtol, atol=rtol))


def run_point(sim, T_C, load, soh):
    """Discharge of one design point; returns runtime (min), end voltage, peak temperature (C), termination.
    The SEI film of an aged cell enters as PyBaMM's contact resistance, whose Joule heat the lumped model counts."""
    st = cell_state(soh)
    T = T_C + 273.15
    inputs = {"Ambient temperature [K]": T, "Initial temperature [K]": T, "Power function [W]": P_RATED * load,
              "Contact resistance [Ohm]": st.r_contact,
              "Initial concentration in negative electrode [mol.m-3]": st.x0 * p.C_N_MAX,
              "Initial concentration in positive electrode [mol.m-3]": st.y0 * p.C_P_MAX}
    t_max = 3.0 * 3600 * 18.2 / (P_RATED * load)
    sol = sim.solve(np.linspace(0, t_max, 400), inputs=inputs)
    t = sol["Time [s]"].entries
    v = sol["Voltage [V]"].entries
    T_cell = sol["X-averaged cell temperature [K]"].entries
    return dict(runtime_min=float(t[-1] / 60), v_end=float(v[-1]), T_peak_C=float(T_cell.max() - 273.15),
                termination=str(sol.termination), ok=bool(abs(v[-1] - p.V_MIN) < 0.02))


def pybamm_version():
    import pybamm
    return pybamm.__version__
