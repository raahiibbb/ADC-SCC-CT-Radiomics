"""Phase 10 - ComBat and prevalence-balanced ComBat.

Why a balanced variant is needed
--------------------------------
Standard ComBat estimates a site's location as the mean of that site's
patients.  When the class mix differs between sites (LUNG1 25 % ADC,
Radiogenomics 79 % ADC) that mean is partly an "ADC-vs-SCC" mean, so removing
it removes disease signal and leaves a residual site-label correlation.
Adding the label as a ComBat covariate fixes the estimate but needs the label
of every patient being harmonised - i.e. it leaks test labels.

Prevalence-balanced ComBat estimates each site's location as the UNWEIGHTED
average of its two class-conditional means and its scale from the pooled
WITHIN-class spread.  Those estimates are what the site would look like at a
50/50 class mix, so they are not confounded by prevalence.  Applying the
correction needs only the site of the patient, never its label.

For a site with no labels at all (leave-one-site-out transport) the class-
conditional means are estimated by EM over the source model's posteriors,
jointly with the destination prior (Saerens et al. 2002 prior-shift EM).

Everything is fitted on a training partition only.
"""
from __future__ import annotations

import numpy as np


def _eb_prior(gamma_hat, delta_hat):
    g_bar, t2 = gamma_hat.mean(), gamma_hat.var(ddof=1)
    m, v = delta_hat.mean(), delta_hat.var(ddof=1)
    v = max(v, 1e-12)
    a = (2.0 * v + m * m) / v
    b = (m * v + m ** 3) / v
    return g_bar, max(t2, 1e-12), a, b


def _eb_shrink(gamma_hat, delta_hat, n, iters=100, tol=1e-4):
    """Parametric empirical-Bayes shrinkage (Johnson et al. 2007), per site,
    across features.  Sum of squares uses n*(delta_hat + (gamma_hat-g)^2),
    which is exact for standard ComBat and the within-class analogue for the
    balanced variant."""
    g_bar, t2, a, b = _eb_prior(gamma_hat, delta_hat)
    g, d = gamma_hat.copy(), delta_hat.copy()
    for _ in range(iters):
        g_new = (n * t2 * gamma_hat + d * g_bar) / (n * t2 + d)
        ss = n * (delta_hat + (gamma_hat - g_new) ** 2)
        d_new = (0.5 * ss + b) / (n / 2.0 + a - 1.0)
        conv = max(np.max(np.abs(g_new - g) / (np.abs(g) + 1e-8)),
                   np.max(np.abs(d_new - d) / (np.abs(d) + 1e-8)))
        g, d = g_new, d_new
        if conv < tol:
            break
    return g, d


def _site_stats(X, y, balanced, w1=None):
    """Location and within-site variance of one site.

    balanced=False: plain mean / variance.
    balanced=True : mean of the two class means / pooled within-class variance.
                    Class membership is either hard (y) or soft (w1 = P(ADC)).
    """
    if not balanced:
        mu = X.mean(axis=0)
        return mu, ((X - mu) ** 2).mean(axis=0)
    if w1 is None:
        w1 = (np.asarray(y) == 1).astype(float)
    w0 = 1.0 - w1
    m1 = (w1[:, None] * X).sum(0) / max(w1.sum(), 1e-8)
    m0 = (w0[:, None] * X).sum(0) / max(w0.sum(), 1e-8)
    var = (w1[:, None] * (X - m1) ** 2 + w0[:, None] * (X - m0) ** 2).sum(0) / len(X)
    return 0.5 * (m1 + m0), var


class Harmoniser:
    """method in {"none", "combat", "balanced_combat"}."""

    def __init__(self, method="none", empirical_bayes=True):
        self.method = method
        self.eb = empirical_bayes

    # ------------------------------------------------------------------ fit
    def fit(self, X, site, y):
        X = np.asarray(X, float)
        site = np.asarray(site)
        self.sites_ = list(np.unique(site))
        if self.method == "none":
            return self
        bal = self.method == "balanced_combat"
        loc, var, n = {}, {}, {}
        for s in self.sites_:
            m = site == s
            loc[s], var[s] = _site_stats(X[m], None if y is None else np.asarray(y)[m], bal)
            n[s] = int(m.sum())
        if bal:  # every site counts equally in the reference
            self.alpha_ = np.mean([loc[s] for s in self.sites_], axis=0)
            pooled = np.mean([var[s] for s in self.sites_], axis=0)
        else:
            tot = sum(n.values())
            self.alpha_ = sum(n[s] * loc[s] for s in self.sites_) / tot
            pooled = sum(n[s] * var[s] for s in self.sites_) / tot
        self.sigma_ = np.sqrt(np.maximum(pooled, 1e-12))
        self.gamma_, self.delta_ = {}, {}
        for s in self.sites_:
            gh = (loc[s] - self.alpha_) / self.sigma_
            dh = np.maximum(var[s] / self.sigma_ ** 2, 1e-8)
            if self.eb and len(gh) > 2:
                gh, dh = _eb_shrink(gh, dh, n[s])
            self.gamma_[s], self.delta_[s] = gh, dh
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X, site):
        X = np.asarray(X, float)
        if self.method == "none":
            return X.copy()
        site = np.asarray(site)
        out = np.empty_like(X)
        for s in np.unique(site):
            if s not in self.gamma_:
                raise KeyError("site %r unseen at fit time; use fit_new_site" % s)
            m = site == s
            z = (X[m] - self.alpha_) / self.sigma_
            out[m] = self.sigma_ * (z - self.gamma_[s]) / np.sqrt(self.delta_[s]) + self.alpha_
        return out

    # ---------------------------------------------- unseen, unlabelled site
    def fit_new_site(self, X_new, name, balanced, w1=None):
        """Add location/scale for an unlabelled site (no EB: one batch).
        balanced=True requires soft labels w1 = P(ADC | x)."""
        X_new = np.asarray(X_new, float)
        loc, var = _site_stats(X_new, None, balanced, w1)
        self.gamma_[name] = (loc - self.alpha_) / self.sigma_
        self.delta_[name] = np.maximum(var / self.sigma_ ** 2, 1e-8)
        return self


def reference_to_site(X_src, y_src, method):
    """For leave-one-site-out: the reference space is the source site itself.
    Fitting on a single site gives gamma=0, delta=1 for that site (identity),
    so the source data are unchanged and only the destination is mapped."""
    h = Harmoniser(method if method != "none" else "none", empirical_bayes=False)
    h.fit(X_src, np.array(["SRC"] * len(X_src)), y_src)
    return h


def saerens_em_prior(p, train_prior=0.5, iters=100, tol=1e-6):
    """Prior-shift EM (Saerens, Latinne & Decaestecker 2002).  Returns the
    adjusted posteriors and the estimated destination prior of class 1."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    pi = train_prior
    q = p
    for _ in range(iters):
        num = (pi / train_prior) * p
        den = num + ((1 - pi) / (1 - train_prior)) * (1 - p)
        q = num / den
        pi_new = q.mean()
        if abs(pi_new - pi) < tol:
            pi = pi_new
            break
        pi = pi_new
    return q, pi
