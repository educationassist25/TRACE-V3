"""
stats.py - vectorised two-group differential statistics with label permutations.

Moderated t-statistics follow limma's empirical-Bayes variance shrinkage (Smyth 2004): prior degrees of
freedom d0 and prior variance s0^2 are estimated once from the observed data (fitFDist) and the same prior
is applied to every permutation. Every statistic is turned into a signed z-score so that layers measured on
different scales are integrated on a common scale.
"""
from __future__ import annotations

import numpy as np
from scipy import special, stats


def _trigamma_inverse(x: float) -> float:
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y += dif
        if -dif / y < 1e-8:
            break
    return y


def fit_f_dist(s2: np.ndarray, d: float) -> tuple[float, float]:
    """limma::fitFDist (moment estimates). Returns (d0, s0^2); d0 may be np.inf."""
    s2 = s2[np.isfinite(s2) & (s2 > 1e-12)]
    if s2.size < 10 or d <= 0:
        return 0.0, float(np.median(s2)) if s2.size else 1.0
    z = np.log(s2)
    e = z - special.digamma(d / 2) + np.log(d / 2)
    emean = e.mean()
    evar = e.var(ddof=1) - special.polygamma(1, d / 2)
    if evar > 0:
        d0 = 2 * _trigamma_inverse(evar)
        s0 = np.exp(emean + special.digamma(d0 / 2) - np.log(d0 / 2))
        return float(d0), float(s0)
    return np.inf, float(np.exp(emean))


def p_to_z(p: np.ndarray, sign: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-300, 1.0)
    return np.sign(sign) * stats.norm.isf(p / 2)


class TwoGroup:
    """Two-group moderated statistics for one layer. X: features x samples (ndarray)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, moderated: bool = True):
        self.X = np.asarray(X, float)
        self.X2 = self.X ** 2
        y = np.asarray(y, int)
        n1, n0 = y.sum(), (1 - y).sum()
        d = n1 + n0 - 2
        diff, s2 = self._raw(y[None, :])
        self.d = float(d)
        if moderated:
            self.d0, self.s0 = fit_f_dist(s2[:, 0], self.d)
        else:
            self.d0, self.s0 = 0.0, 1.0

    def _raw(self, L: np.ndarray):
        L = L.astype(float)
        n1 = L.sum(1)
        n0 = (1 - L).sum(1)
        S1 = self.X @ L.T
        S0 = self.X @ (1 - L).T
        Q1 = self.X2 @ L.T
        Q0 = self.X2 @ (1 - L).T
        m1, m0 = S1 / n1, S0 / n0
        ss = (Q1 - n1 * m1 ** 2) + (Q0 - n0 * m0 ** 2)
        s2 = np.maximum(ss / (n1 + n0 - 2), 1e-12)
        return m1 - m0, s2

    def run(self, L: np.ndarray) -> dict:
        """L: B x n matrix of case(1)/reference(0) labels. Returns dicts of f x B arrays."""
        L = np.atleast_2d(L)
        diff, s2 = self._raw(L)
        n1 = L.sum(1).astype(float)
        n0 = (1 - L).sum(1).astype(float)
        d = n1 + n0 - 2
        if np.isinf(self.d0):
            s2p, dft = np.full_like(s2, self.s0), None
        elif self.d0 > 0:
            s2p = (self.d0 * self.s0 + d * s2) / (self.d0 + d)
            dft = d + self.d0
        else:
            s2p, dft = s2, d
        se = np.sqrt(s2p * (1 / n1 + 1 / n0))
        t = diff / se
        if dft is None:
            p = 2 * stats.norm.sf(np.abs(t))
        else:
            p = 2 * stats.t.sf(np.abs(t), np.broadcast_to(dft, t.shape))
        return {"logfc": diff, "se": se, "t": t, "p": p, "z": p_to_z(p, t)}


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN-safe)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty_like(pv)
    out[o] = np.minimum(r, 1)
    q[ok] = out
    return q


def empirical_p(obs: np.ndarray, null: np.ndarray, two_sided: bool = True) -> np.ndarray:
    """obs: k ; null: k x B. p = (b + 1) / (B + 1). NaN where obs is NaN."""
    obs = np.asarray(obs, float)
    null = np.asarray(null, float)
    if two_sided:
        b = (np.abs(null) >= np.abs(obs)[:, None] - 1e-12).sum(1)
    else:
        b = (null >= obs[:, None] - 1e-12).sum(1)
    B = np.isfinite(null).sum(1)
    p = (b + 1) / (B + 1)
    p[~np.isfinite(obs)] = np.nan
    return p


def permutation_labels(samples: list, groups: dict, case: str, B: int, seed: int) -> tuple[np.ndarray, list]:
    """Label permutations on the union of samples (row 0 = observed labels). The same permutation is
    used by every layer, so cross-omics dependence between matched samples is preserved (Tier 1);
    unmatched samples are automatically permuted independently (Tiers 2-3)."""
    rng = np.random.default_rng(seed)
    y = np.array([1 if groups[s] == case else 0 for s in samples])
    L = [y]
    seen = {y.tobytes()}
    tries = 0
    while len(L) < B + 1 and tries < 50 * (B + 1):
        tries += 1
        yp = rng.permutation(y)
        k = yp.tobytes()
        if k in seen and len(y) > 12:
            continue
        seen.add(k)
        L.append(yp)
    return np.array(L), samples


# --------------------------------------------------------------------------- permutation-calibrated parametric nulls
def product_normal_p(c: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Two-sided p-value of an observed product c = x*y, where (x, y) are standard bivariate normal with
    correlation rho under the null. Density (Craig 1936 / Nadarajah & Pogany 2016):
        f(z) = exp(rho z / (1 - rho^2)) K0(|z| / (1 - rho^2)) / (pi sqrt(1 - rho^2))
    Means, scales and rho are estimated from the label permutations, so the null is calibrated by
    permutation while p-values below 1/B remain resolvable."""
    from scipy import integrate

    c = np.asarray(c, float)
    rho = np.clip(np.nan_to_num(np.asarray(rho, float)), -0.95, 0.95)
    out = np.full(c.shape, np.nan)
    for i, (ci, ri) in enumerate(zip(c, rho)):
        if not np.isfinite(ci):
            continue
        a = abs(ci)
        s = 1 - ri ** 2

        def f(z, sign):
            u = abs(z) / s
            # exp(rho*z/s) * K0(u) = k0e(u) * exp(rho*z/s - u)
            return special.k0e(u) * np.exp(sign * ri * z / s - u) / (np.pi * np.sqrt(s))

        if a < 1e-9:
            out[i] = 1.0
            continue
        up = integrate.quad(lambda z: f(z, 1), a, np.inf, limit=200)[0]
        lo = integrate.quad(lambda z: f(z, -1), a, np.inf, limit=200)[0]
        out[i] = min(1.0, up + lo)
    return out


def calibrated_product_p(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X, Y: k x (1 + B) arrays (column 0 observed, others permutations) of the two factors of a product
    score. Returns (p, rho_null): parametric product-normal p with permutation-estimated location, scale
    and correlation."""
    xn, yn = X[:, 1:], Y[:, 1:]
    mx, sx = np.nanmean(xn, 1), np.nanstd(xn, 1)
    my, sy = np.nanmean(yn, 1), np.nanstd(yn, 1)
    sx = np.where(sx > 1e-9, sx, np.nan)
    sy = np.where(sy > 1e-9, sy, np.nan)
    xs = (xn - mx[:, None]) / sx[:, None]
    ys = (yn - my[:, None]) / sy[:, None]
    rho = np.nanmean(xs * ys, 1)
    c = ((X[:, 0] - mx) / sx) * ((Y[:, 0] - my) / sy)
    return product_normal_p(c, rho), rho


def normal_p_from_null(obs: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise by the permutation null (mean, sd) and return (z, two-sided normal p)."""
    mu = np.nanmean(null, 1)
    sd = np.nanstd(null, 1)
    z = (obs - mu) / np.where(sd > 1e-12, sd, np.nan)
    return z, 2 * stats.norm.sf(np.abs(z))
