"""Finite-volume building blocks of the reduced-order cell model.

Both grids carry a leading batch axis, so many operating points advance together. Each time step is backward
Euler with the coefficients (diffusivities, transference number) taken from the previous step, which leaves a
linear tridiagonal system; its solution is returned as an affine function of the cell current, u + I w, so the
current of a constant-power step can be found afterwards by Newton iteration on the terminal voltage.
"""
import numpy as np


def thomas(a, b, c, d):
    """Solve tridiagonal systems. a, b, c have shape (B, N), d shape (B, N) or (B, N, K) for K right-hand sides
    sharing one matrix; a[:, 0] and c[:, -1] are unused."""
    many = d.ndim == 3
    col = (lambda v: v[:, None]) if many else (lambda v: v)
    n = b.shape[1]
    cp = np.empty_like(b)
    dp = np.empty_like(d)
    cp[:, 0] = c[:, 0] / b[:, 0]
    dp[:, 0] = d[:, 0] / col(b[:, 0])
    for i in range(1, n):
        m = b[:, i] - a[:, i] * cp[:, i - 1]
        cp[:, i] = c[:, i] / m
        dp[:, i] = (d[:, i] - col(a[:, i]) * dp[:, i - 1]) / col(m)
    x = np.empty_like(d)
    x[:, -1] = dp[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - col(cp[:, i]) * x[:, i + 1]
    return x


class Particle:
    """Spherical particle on n equal radial cells, in stoichiometry theta = c / c_max.

        d theta/dt = (1/r^2) d/dr ( r^2 D(theta, T) d theta/dr ),   -D d theta/dr = q  at r = R,

    with D(theta, T) = D_ref(theta) A(T). The diffusivity of both electrodes changes by orders of magnitude over
    narrow stoichiometry windows, so the profile contains steep fronts. Face diffusivities therefore come from
    the Kirchhoff transform Phi(theta) = integral of D_ref d theta,
        D_face = (Phi_i+1 - Phi_i) / (theta_i+1 - theta_i),
    which gives the exact flux of a steady front between two cells."""

    def __init__(self, radius, n, D_ref, m=40001):
        self.R, self.n = radius, n
        self.dr = radius / n
        rf = np.linspace(0.0, radius, n + 1)
        self.vol = (rf[1:] ** 3 - rf[:-1] ** 3) / 3.0
        self.af = rf ** 2
        self.w = self.vol / self.vol.sum()          # weights of the volume average
        self.D_ref = D_ref
        self.tg = np.linspace(0.0, 1.0, m)
        Dg = D_ref(self.tg)
        self.phi = np.concatenate([[0.0], np.cumsum(0.5 * (Dg[1:] + Dg[:-1]) * np.diff(self.tg))])

    def Phi(self, theta):
        return np.interp(theta, self.tg, self.phi)

    def Phi_inv(self, phi):
        return np.interp(phi, self.phi, self.tg)

    def system(self, theta, arr, dt):
        """Matrix (a, b, c) and right-hand sides (B, n, 2) of one step; see step."""
        B, n = theta.shape
        dth = theta[:, 1:] - theta[:, :-1]
        small = np.abs(dth) < 1e-10
        Ph = self.Phi(theta)
        Dmid = self.D_ref(0.5 * (theta[:, 1:] + theta[:, :-1]))
        Df = np.where(small, Dmid, (Ph[:, 1:] - Ph[:, :-1]) / np.where(small, 1.0, dth)) * arr
        g = self.af[1:-1] * Df / self.dr
        a = np.zeros((B, n))
        c = np.zeros((B, n))
        a[:, 1:] = -dt * g / self.vol[1:]
        c[:, :-1] = -dt * g / self.vol[:-1]
        b = 1.0 - a - c
        rhs = np.zeros((B, n, 2))
        rhs[:, :, 0] = theta
        rhs[:, -1, 1] = -dt * self.af[-1] / self.vol[-1]
        return a, b, c, rhs

    def step(self, theta, arr, dt):
        """One step; arr is the factor A(T), shape (B, 1). Returns (u, w) with theta_new = u + q w, q being the
        outward surface flux divided by c_max (m/s)."""
        x = thomas(*self.system(theta, arr, dt))
        return x[:, :, 0], x[:, :, 1]

    def surface(self, theta_last, q, arr):
        """Surface stoichiometry from the outermost cell: Phi_s = Phi_N - q dr / (2 A)."""
        return self.Phi_inv(self.Phi(theta_last) - q * self.dr / (2.0 * arr))

    def mean(self, theta):
        return theta @ self.w


def step_pair(pn, th_n, arr_n, pp, th_p, arr_p, dt):
    """Steps of the two particles of a cell in one tridiagonal solve (both grids need the same size)."""
    if th_n.shape[1] != th_p.shape[1]:
        return pn.step(th_n, arr_n, dt) + pp.step(th_p, arr_p, dt)
    sn, sp = pn.system(th_n, arr_n, dt), pp.system(th_p, arr_p, dt)
    x = thomas(*(np.concatenate([u, v]) for u, v in zip(sn, sp, strict=True)))
    B = th_n.shape[0]
    return x[:B, :, 0], x[:B, :, 1], x[B:, :, 0], x[B:, :, 1]


class Electrolyte:
    """Salt concentration across negative electrode, separator and positive electrode (SPMe electrolyte).

        eps dc/dt = -dN/dx + s,   N = -eps^b D_e(c, T) dc/dx + t+(c, T) i_e / F,

    with the reaction spread evenly over each electrode (single-particle assumption): the electrolyte current
    density i_e rises linearly from 0 to i through the negative electrode, is i in the separator and falls back
    to 0 through the positive one, and the source s = (d i_e/dx)/F. No flux at either current collector.
    Equal cells within each region; the face conductance between two cells is that of the two half cells in
    series, which keeps the flux continuous across the change of porosity. The scheme conserves the salt
    (sum of eps c h) to round-off."""

    def __init__(self, p, n_n=20, n_s=10, n_p=20):
        h = np.concatenate([np.full(n_n, p.L_N / n_n), np.full(n_s, p.L_S / n_s), np.full(n_p, p.L_P / n_p)])
        self.h = h
        edges = np.concatenate([[0.0], np.cumsum(h)])
        self.x = 0.5 * (edges[1:] + edges[:-1])
        self.xf = edges[1:-1]                          # interior faces
        self.eps = np.concatenate([np.full(n_n, p.EPS_E_N), np.full(n_s, p.EPS_E_S), np.full(n_p, p.EPS_E_P)])
        self.tor = self.eps ** p.BRUG
        self.region = np.concatenate([np.zeros(n_n, int), np.ones(n_s, int), np.full(n_p, 2)])
        self.N = h.size
        # electrolyte current per unit cell current density, at faces and at cell centres
        self.ie_f = self._ie(self.xf, p)
        # at the centres of the two half cells between neighbouring cell centres (exact for the linear i_e)
        self.ie_right_half = self._ie(self.x[:-1] + 0.25 * h[:-1], p)
        self.ie_left_half = self._ie(self.x[1:] - 0.25 * h[1:], p)
        # source per unit current density (A/m2 -> mol/(m3 s) after division by F)
        self.src = np.where(self.region == 0, 1.0 / p.L_N, np.where(self.region == 2, -1.0 / p.L_P, 0.0)) / p.F
        self.p = p
        self.in_n, self.in_p = self.region == 0, self.region == 2

    @staticmethod
    def _ie(x, p):
        x = np.asarray(x, float)
        return np.where(x <= p.L_N, x / p.L_N, np.where(x <= p.L_N + p.L_S, 1.0, (p.L_X - x) / p.L_P))

    def step(self, c, T, dt):
        """One step; c has shape (B, N), T shape (B,). Returns (u, w) with c_new = u + i w, i in A/m2."""
        p = self.p
        B, N = c.shape
        k = self.tor * p.D_e(c, T[:, None])                         # cell conductivities, m2/s
        hl, hr = self.h[:-1], self.h[1:]
        G = 1.0 / (0.5 * hl / k[:, :-1] + 0.5 * hr / k[:, 1:])      # face conductance, m/s
        cf = 0.5 * (c[:, :-1] + c[:, 1:])
        mig = p.t_plus(cf, T[:, None]) * self.ie_f / p.F            # migration flux per unit i, mol/(m2 s) / (A/m2)
        cap = self.eps * self.h / dt
        a = np.zeros((B, N))
        cc = np.zeros((B, N))
        a[:, 1:] = -G
        cc[:, :-1] = -G
        b = cap - a - cc
        rhs = np.empty((B, N, 2))
        rhs[:, :, 0] = cap * c
        # current-proportional part: source minus divergence of the migration flux
        r = rhs[:, :, 1]
        r[:] = self.h * self.src
        r[:, :-1] -= mig
        r[:, 1:] += mig
        x = thomas(a, b, cc, rhs)
        return x[:, :, 0], x[:, :, 1]

    def potential_difference(self, c, T, i):
        """Mean electrolyte potential of the positive electrode minus that of the negative one, V, for the
        cell current density i (A/m2, shape (B,)):

            d phi_e/dx = chi(c, T) (RT/F) d ln c/dx - i_e / (eps^b kappa_e(c, T)),

        integrated from cell centre to cell centre with chi and kappa_e evaluated locally (the 'integrated'
        electrolyte potential of the SPMe). With chi and kappa_e independent of c this is exactly the
        composite SPMe: chi (RT/F) (mean_p ln c - mean_n ln c) - i (L_n/(3 kappa_n) + L_s/kappa_s + L_p/(3 kappa_p))."""
        p = self.p
        Tc = T[:, None]
        lnc = np.log(np.maximum(c, 1e-15))
        cf = 0.5 * (c[:, :-1] + c[:, 1:])
        diffusion = (p.R * Tc / p.F) * p.chi(cf, Tc) * (lnc[:, 1:] - lnc[:, :-1])
        k = p.kappa_e(c, Tc) * self.tor
        resist = 0.5 * self.h[:-1] * self.ie_right_half / k[:, :-1] + 0.5 * self.h[1:] * self.ie_left_half / k[:, 1:]
        steps = diffusion - i[:, None] * resist
        phi = np.concatenate([np.zeros((c.shape[0], 1)), np.cumsum(steps, axis=1)], axis=1)
        return (phi[:, self.in_p] @ self.h[self.in_p]) / p.L_P - (phi[:, self.in_n] @ self.h[self.in_n]) / p.L_N

    def salt(self, c):
        return c @ (self.eps * self.h)
