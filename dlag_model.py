"""
dlag_model.py

A from-scratch implementation of DLAG (Delayed Latents Across Groups; Gokcen
et al., "Disentangling the flow of signals between populations of neurons",
Nat Comput Sci 2024) for exactly two simultaneously-recorded, trial-paired
populations (here: cortex npix vs. cerebellum zF).

GENERATIVE MODEL
-----------------
Group 1 (cortex, dim q1) and group 2 (cerebellum, dim q2) share a common
per-trial time grid t = 0..T-1. Each trial has:

  - p1 "within-group-1" latents: private to group 1, GP(0, k_i) each.
  - p2 "within-group-2" latents: private to group 2, GP(0, k_j) each.
  - p12 "across-group" latents: ONE underlying GP(0, k_k) per latent, but
    group 1 observes it at delay 0 (reference) and group 2 observes it at a
    learned delay D_k (in time-bin units). So
        u1_k(t)  = x_k(t)        (group 1's view)
        u2_k(t') = x_k(t' - D_k) (group 2's view)
    giving cross-covariance Cov[u1_k(t), u2_k(t')] = k_k((t - t') + D_k).
    D_k > 0 means group 2 (cerebellum) lags group 1 (cortex) -- i.e. cortex
    leads. D_k < 0 means cerebellum leads.

  Each kernel is squared-exponential with a small fixed nugget for
  conditioning k(Δt; τ) = (1-ε)·exp(-Δt²/2τ²) + ε·[Δt=0], ε fixed (not
  learned), consistent with GPFA/DLAG's own convention.

  v1(t) = [within-1 latents at t ; across latents' group-1 view at t],  dim p1+p12
  v2(t) = [within-2 latents at t ; across latents' group-2 view at t],  dim p2+p12

  y1(t) = d1 + C1 @ v1(t) + eps1(t),  eps1(t) ~ N(0, R1),  R1 = diag(r1)
  y2(t) = d2 + C2 @ v2(t) + eps2(t),  eps2(t) ~ N(0, R2),  R2 = diag(r2)

  v1(t) and v2(t) at a FIXED t are otherwise independent (only the
  across-group latents correlate v1 and v2, and only via the GP prior across
  time/groups, never directly through the loadings).

FITTING
-------
Rather than hand-deriving DLAG's EM (delay-aware Kalman-smoothing E-step +
closed-form/numerical M-step), we directly maximize the exact marginal
log-likelihood of y = (y1, y2) via torch autograd + Adam. This is the same
objective the paper's EM converges to (same generative model), just a
different optimizer -- standard modern practice (e.g. this is exactly how
GPyTorch fits exact GPs), and far less error-prone to implement correctly.

The whole per-trial latent vector z = [v1(0),...,v1(T-1), v2(0),...,v2(T-1)]
(flattened, "time-major, latent-minor" within each group) has a Gaussian
prior z ~ N(0, K_z) that is dense but built from small per-latent blocks (see
build_K_z). Because C1, C2, R1, R2, and K_z do NOT depend on the trial (only
y does), the linear-Gaussian conjugacy gives, once per gradient step (shared
across all trials):

    M   = K_z^{-1} + Phi^T R^{-1} Phi        (posterior precision, N_z x N_z)
    b   = Phi^T R^{-1} (y - d)                (per trial, N_z,)
    posterior mean   z_post = M^{-1} b = alpha
    posterior cov     Cov[z|y] = M^{-1}       (same for every trial)

and via the matrix determinant lemma,

    NLL = 0.5 * [ (y-d)^T R^{-1} (y-d) - b^T M^{-1} b
                  + logdet(R) + logdet(K_z) + logdet(M) + N_obs*log(2*pi) ]

Only ONE N_z x N_z Cholesky/inverse (N_z ~ 1000-2000 for typical settings) is
needed per gradient step, never an N_obs x N_obs (~76,000-dim here) one --
this is what makes fitting tractable. Phi^T R^{-1} Phi itself is block
diagonal (block per group, and within each group block-diagonal over time
with a repeated small (p_group x p_group) block), so it's built via
torch.kron rather than ever materializing Phi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def se_kernel(delta_t: torch.Tensor, tau: torch.Tensor, eps: float) -> torch.Tensor:
    """Squared-exponential kernel with a fixed nugget: (1-eps)*exp(-dt^2/2tau^2) + eps*[dt==0]."""
    k = (1.0 - eps) * torch.exp(-0.5 * (delta_t / tau) ** 2)
    nugget = eps * (delta_t == 0).to(k.dtype)
    return k + nugget


def _inv_softplus(y: float) -> float:
    """Inverse of softplus, for initializing raw parameters to a target positive value."""
    y = max(y, 1e-6)
    return math.log(math.expm1(y))


@dataclass
class DLAGDims:
    q1: int
    q2: int
    T: int
    p1: int
    p2: int
    p12: int


class DLAGModel(nn.Module):
    def __init__(
        self,
        q1: int,
        q2: int,
        T: int,
        p1: int,
        p2: int,
        p12: int,
        dt: float = 1.0,
        gp_eps: float = 1e-3,
        max_delay_bins: float | None = None,
        init_tau_bins: float = 5.0,
        noise_floor: float = 1e-4,
        dtype: torch.dtype = torch.float64,
    ):
        """
        q1, q2 : observed dims of group 1 (cortex) / group 2 (cerebellum)
        T      : number of timepoints per trial (shared time grid)
        p1, p2 : number of within-group-1 / within-group-2 private latents
        p12    : number of across-group (shared) latents
        dt     : seconds per time bin (for reporting delays/timescales in
                 real units via .timescales_ms() / .delays_ms()); has no
                 effect on fitting itself, which works in bin units throughout
        gp_eps : fixed GP nugget (not learned), for conditioning (observation noise)
        max_delay_bins : cap on |delay| in bins (delays beyond ~1 timescale
                 are unidentifiable anyway); defaults to T // 4
        """
        super().__init__()
        self.dims = DLAGDims(q1, q2, T, p1, p2, p12)
        self.dt = dt
        self.gp_eps = gp_eps
        self.noise_floor = noise_floor
        self.max_delay_bins = max_delay_bins if max_delay_bins is not None else T / 4.0
        self._dtype = dtype

        P1, P2 = p1 + p12, p2 + p12
        self.P1, self.P2 = P1, P2
        self.N1 = T * P1
        self.N2 = T * P2
        self.N_z = self.N1 + self.N2

        # --- GP hyperparameters (raw, unconstrained; transformed in properties below) ---
        init_raw_tau = _inv_softplus(init_tau_bins)
        self.raw_tau_w1 = nn.Parameter(torch.full((p1,), init_raw_tau, dtype=dtype))
        self.raw_tau_w2 = nn.Parameter(torch.full((p2,), init_raw_tau, dtype=dtype))
        self.raw_tau_a = nn.Parameter(torch.full((p12,), init_raw_tau, dtype=dtype))
        self.raw_delay = nn.Parameter(torch.zeros(p12, dtype=dtype))  # tanh(0)=0 -> start at zero delay

        # --- observation model parameters ---
        self.C1 = nn.Parameter(torch.randn(q1, P1, dtype=dtype) / math.sqrt(max(P1, 1)))
        self.C2 = nn.Parameter(torch.randn(q2, P2, dtype=dtype) / math.sqrt(max(P2, 1)))
        self.d1 = nn.Parameter(torch.zeros(q1, dtype=dtype))
        self.d2 = nn.Parameter(torch.zeros(q2, dtype=dtype))
        init_raw_r = _inv_softplus(1.0)
        self.raw_r1 = nn.Parameter(torch.full((q1,), init_raw_r, dtype=dtype))
        self.raw_r2 = nn.Parameter(torch.full((q2,), init_raw_r, dtype=dtype))

        # time-difference matrix Dt[t,t'] = t - t', shared/static
        t_grid = torch.arange(T, dtype=dtype)
        self.register_buffer("Dt", t_grid[:, None] - t_grid[None, :])

    # ------------------------------------------------------------------
    # Transformed (constrained) hyperparameters
    # ------------------------------------------------------------------
    @property
    def tau_w1(self):
        return nn.functional.softplus(self.raw_tau_w1) + 0.5

    @property
    def tau_w2(self):
        return nn.functional.softplus(self.raw_tau_w2) + 0.5

    @property
    def tau_a(self):
        return nn.functional.softplus(self.raw_tau_a) + 0.5

    @property
    def delay_a(self):
        return torch.tanh(self.raw_delay) * self.max_delay_bins

    @property
    def r1(self):
        return nn.functional.softplus(self.raw_r1) + self.noise_floor

    @property
    def r2(self):
        return nn.functional.softplus(self.raw_r2) + self.noise_floor

    # ------------------------------------------------------------------
    # Prior covariance K_z
    # ------------------------------------------------------------------
    def build_K_z(self):
        p1, p2, p12 = self.dims.p1, self.dims.p2, self.dims.p12
        T, P1, P2 = self.dims.T, self.P1, self.P2
        Dt = self.Dt
        eps = self.gp_eps
        dtype = self._dtype

        Kv1 = torch.zeros(T, P1, T, P1, dtype=dtype)
        for i in range(p1):
            Kv1[:, i, :, i] = se_kernel(Dt, self.tau_w1[i], eps)
        for k in range(p12):
            Kv1[:, p1 + k, :, p1 + k] = se_kernel(Dt, self.tau_a[k], eps)
        Kv1_mat = Kv1.reshape(T * P1, T * P1)

        Kv2 = torch.zeros(T, P2, T, P2, dtype=dtype)
        for j in range(p2):
            Kv2[:, j, :, j] = se_kernel(Dt, self.tau_w2[j], eps)
        for k in range(p12):
            Kv2[:, p2 + k, :, p2 + k] = se_kernel(Dt, self.tau_a[k], eps)
        Kv2_mat = Kv2.reshape(T * P2, T * P2)

        Kcross = torch.zeros(T, P1, T, P2, dtype=dtype)
        for k in range(p12):
            shifted = Dt + self.delay_a[k]
            Kcross[:, p1 + k, :, p2 + k] = se_kernel(shifted, self.tau_a[k], eps)
        Kcross_mat = Kcross.reshape(T * P1, T * P2)

        top = torch.cat([Kv1_mat, Kcross_mat], dim=1)
        bottom = torch.cat([Kcross_mat.transpose(0, 1), Kv2_mat], dim=1)
        K_z = torch.cat([top, bottom], dim=0)
        return K_z

    # ------------------------------------------------------------------
    # Core: build the shared (trial-independent) pieces once per call
    # ------------------------------------------------------------------
    def _shared_terms(self):
        K_z = self.build_K_z()
        N_z = self.N_z
        jitter = 1e-8
        L_Kz = torch.linalg.cholesky(K_z + jitter * torch.eye(N_z, dtype=self._dtype))
        Kz_inv = torch.cholesky_inverse(L_Kz)
        logdet_Kz = 2.0 * torch.sum(torch.log(torch.diagonal(L_Kz)))

        M1 = self.C1.T @ (self.C1 / self.r1[:, None])  # (P1,P1) = C1^T diag(1/r1) C1
        M2 = self.C2.T @ (self.C2 / self.r2[:, None])
        eyeT = torch.eye(self.dims.T, dtype=self._dtype)
        PhiTRinvPhi = torch.zeros(N_z, N_z, dtype=self._dtype)
        PhiTRinvPhi[: self.N1, : self.N1] = torch.kron(eyeT, M1)
        PhiTRinvPhi[self.N1 :, self.N1 :] = torch.kron(eyeT, M2)

        M = Kz_inv + PhiTRinvPhi
        L_M = torch.linalg.cholesky(M)
        logdet_M = 2.0 * torch.sum(torch.log(torch.diagonal(L_M)))

        logdet_R = self.dims.T * (torch.sum(torch.log(self.r1)) + torch.sum(torch.log(self.r2)))

        return {"L_M": L_M, "logdet_M": logdet_M, "logdet_Kz": logdet_Kz, "logdet_R": logdet_R}

    def _per_trial_b_and_quad(self, Y1: torch.Tensor, Y2: torch.Tensor):
        """
        Y1: (n_trials, T, q1), Y2: (n_trials, T, q2)
        Returns b (n_trials, N_z) and quad1 = (y-d)^T R^{-1} (y-d) per trial.
        """
        resid1 = Y1 - self.d1  # (n,T,q1)
        resid2 = Y2 - self.d2
        w1 = resid1 / self.r1  # (n,T,q1)
        w2 = resid2 / self.r2

        b1 = torch.einsum("ntq,qp->ntp", w1, self.C1).reshape(Y1.shape[0], self.N1)
        b2 = torch.einsum("ntq,qp->ntp", w2, self.C2).reshape(Y2.shape[0], self.N2)
        b = torch.cat([b1, b2], dim=1)  # (n, N_z)

        quad1 = (resid1 * w1).sum(dim=(1, 2)) + (resid2 * w2).sum(dim=(1, 2))  # (n,)
        return b, quad1

    def neg_log_likelihood(self, Y1: torch.Tensor, Y2: torch.Tensor, reduction: str = "mean"):
        """Y1: (n_trials, T, q1), Y2: (n_trials, T, q2) -> scalar NLL (mean or sum over trials)."""
        shared = self._shared_terms()
        b, quad1 = self._per_trial_b_and_quad(Y1, Y2)

        alpha = torch.cholesky_solve(b.T, shared["L_M"])  # (N_z, n_trials), M^{-1} b
        quad2 = (b.T * alpha).sum(dim=0)  # (n_trials,)

        N_obs = self.dims.T * (self.dims.q1 + self.dims.q2)
        nll_per_trial = 0.5 * (
            quad1 - quad2
            + shared["logdet_R"]
            + shared["logdet_Kz"]
            + shared["logdet_M"]
            + N_obs * math.log(2 * math.pi)
        )
        if reduction == "mean":
            return nll_per_trial.mean()
        elif reduction == "sum":
            return nll_per_trial.sum()
        elif reduction == "none":
            return nll_per_trial
        raise ValueError(f"Unknown reduction {reduction!r}")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        Y1_train: np.ndarray,
        Y2_train: np.ndarray,
        n_epochs: int = 300,
        lr: float = 0.05,
        init_from_data: bool = True,
        verbose: bool = True,
        print_every: int = 25,
    ):
        Y1_t = torch.as_tensor(np.asarray(Y1_train), dtype=self._dtype)
        Y2_t = torch.as_tensor(np.asarray(Y2_train), dtype=self._dtype)

        if init_from_data:
            with torch.no_grad():
                self.d1.copy_(Y1_t.mean(dim=(0, 1)))
                self.d2.copy_(Y2_t.mean(dim=(0, 1)))
                var1 = Y1_t.var(dim=(0, 1)).clamp_min(1e-3)
                var2 = Y2_t.var(dim=(0, 1)).clamp_min(1e-3)
                self.raw_r1.copy_(torch.log(torch.expm1(var1)))
                self.raw_r2.copy_(torch.log(torch.expm1(var2)))

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_history = []
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = self.neg_log_likelihood(Y1_t, Y2_t, reduction="mean")
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            if verbose and (epoch % print_every == 0 or epoch == n_epochs - 1):
                print(f"[DLAG] epoch {epoch:4d}/{n_epochs}  NLL/trial = {loss.item():.3f}")
        return np.array(loss_history)

    # ------------------------------------------------------------------
    # Posterior inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def posterior_latents(self, Y1: np.ndarray, Y2: np.ndarray):
        """
        Returns dict of posterior MEANS, each (n_trials, T, p):
            "within1" (p1), "within2" (p2), "across1" (p12, group-1-aligned),
            "across2" (p12, group-2-aligned, i.e. delayed relative to across1)
        plus "posterior_var" (N_z,) marginal posterior variance (diag of M^{-1}),
        shared across all trials, split the same way under "*_var" keys.
        """
        Y1_t = torch.as_tensor(np.asarray(Y1), dtype=self._dtype)
        Y2_t = torch.as_tensor(np.asarray(Y2), dtype=self._dtype)
        n_trials = Y1_t.shape[0]

        shared = self._shared_terms()
        b, _ = self._per_trial_b_and_quad(Y1_t, Y2_t)
        alpha = torch.cholesky_solve(b.T, shared["L_M"]).T  # (n_trials, N_z)

        M_inv_diag = torch.cholesky_inverse(shared["L_M"]).diagonal()

        p1, p2, p12, T = self.dims.p1, self.dims.p2, self.dims.p12, self.dims.T
        v1 = alpha[:, : self.N1].reshape(n_trials, T, self.P1)
        v2 = alpha[:, self.N1 :].reshape(n_trials, T, self.P2)
        v1_var = M_inv_diag[: self.N1].reshape(T, self.P1)
        v2_var = M_inv_diag[self.N1 :].reshape(T, self.P2)

        out = {
            "within1": v1[:, :, :p1].numpy(),
            "across1": v1[:, :, p1:].numpy(),
            "within2": v2[:, :, :p2].numpy(),
            "across2": v2[:, :, p2:].numpy(),
            "within1_var": v1_var[:, :p1].numpy(),
            "across1_var": v1_var[:, p1:].numpy(),
            "within2_var": v2_var[:, :p2].numpy(),
            "across2_var": v2_var[:, p2:].numpy(),
        }
        return out

    # ------------------------------------------------------------------
    # Interpretable summaries
    # ------------------------------------------------------------------
    @torch.no_grad()
    def timescales_ms(self):
        return {
            "within1": (self.tau_w1 * self.dt * 1000).numpy(),
            "within2": (self.tau_w2 * self.dt * 1000).numpy(),
            "across": (self.tau_a * self.dt * 1000).numpy(),
        }

    @torch.no_grad()
    def delays_ms(self):
        """Positive = group2 (cerebellum) lags group1 (cortex), i.e. cortex leads."""
        return (self.delay_a * self.dt * 1000).numpy()

    @torch.no_grad()
    def shared_variance_explained(self):
        """
        Per group, fraction of total modeled (signal + noise) variance
        attributable to across-group vs. within-group latents, using each
        latent's unit prior variance (k(0)=1) so total signal covariance is
        C @ C^T (independent unit-variance latents).
        """
        p1, p2 = self.dims.p1, self.dims.p2
        C1_w, C1_a = self.C1[:, :p1], self.C1[:, p1:]
        C2_w, C2_a = self.C2[:, :p2], self.C2[:, p2:]

        var1_within = (C1_w**2).sum(dim=1)
        var1_across = (C1_a**2).sum(dim=1)
        var1_total = var1_within + var1_across + self.r1

        var2_within = (C2_w**2).sum(dim=1)
        var2_across = (C2_a**2).sum(dim=1)
        var2_total = var2_within + var2_across + self.r2

        return {
            "group1_frac_across_of_signal": (var1_across.sum() / (var1_within.sum() + var1_across.sum())).item(),
            "group1_frac_across_of_total": (var1_across.sum() / var1_total.sum()).item(),
            "group2_frac_across_of_signal": (var2_across.sum() / (var2_within.sum() + var2_across.sum())).item(),
            "group2_frac_across_of_total": (var2_across.sum() / var2_total.sum()).item(),
        }
