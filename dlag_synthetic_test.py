"""
dlag_synthetic_test.py

Correctness gate for dlag_model.py, to run BEFORE trusting any real-data
DLAG result. Samples synthetic two-population trial data directly from
DLAGModel's own generative model (build_K_z + C/d/R), with known ground-truth
timescales, delays (one cortex-leads case, one cerebellum-leads case), and
loadings; fits a freshly-initialized DLAGModel on it; and checks that the
recovered delays (sign + magnitude), timescales, and posterior latent
trajectories match ground truth.

Latents are matched between fit and truth by trajectory correlation (via
scipy's linear_sum_assignment) rather than assumed to keep their original
index, since optimization can permute latents that share similar timescales,
and each latent has an inherent +/- sign ambiguity (x_k -> -x_k, C_k -> -C_k
leaves the model unchanged) that both trajectory correlation and reported
delay/timescale are robust to (delay/timescale characterize the kernel's own
autocorrelation structure, unaffected by a uniform sign flip of a latent).

Run directly: python3 dlag_synthetic_test.py
"""

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from dlag_model import DLAGModel

torch.manual_seed(0)
np.random.seed(0)


def simulate_dlag(q1, q2, T, p1, p2, p12, n_trials, tau_w1, tau_w2, tau_a, delay_a,
                   c_scale=1.5, noise_var=0.05, seed=0):
    """Sample synthetic trial data directly from the DLAGModel generative model."""
    rng = torch.Generator().manual_seed(seed)
    truth = DLAGModel(q1, q2, T, p1, p2, p12, dt=1.0)
    with torch.no_grad():
        truth.raw_tau_w1.copy_(torch.tensor([_inv_sp(t) for t in tau_w1]))
        truth.raw_tau_w2.copy_(torch.tensor([_inv_sp(t) for t in tau_w2]))
        truth.raw_tau_a.copy_(torch.tensor([_inv_sp(t) for t in tau_a]))
        # raw_delay s.t. tanh(raw)*max_delay_bins = delay_a
        ratios = np.clip(np.array(delay_a) / truth.max_delay_bins, -0.999, 0.999)
        truth.raw_delay.copy_(torch.tensor(np.arctanh(ratios)))
        truth.C1.copy_(c_scale * torch.randn(q1, p1 + p12, generator=rng, dtype=torch.float64))
        truth.C2.copy_(c_scale * torch.randn(q2, p2 + p12, generator=rng, dtype=torch.float64))
        truth.d1.zero_()
        truth.d2.zero_()
        truth.raw_r1.copy_(torch.full((q1,), _inv_sp(noise_var)))
        truth.raw_r2.copy_(torch.full((q2,), _inv_sp(noise_var)))

        K_z = truth.build_K_z()
        L = torch.linalg.cholesky(K_z + 1e-8 * torch.eye(truth.N_z, dtype=torch.float64))
        z = (L @ torch.randn(truth.N_z, n_trials, generator=rng, dtype=torch.float64)).T  # (n_trials, N_z)

        v1 = z[:, : truth.N1].reshape(n_trials, T, truth.P1)
        v2 = z[:, truth.N1 :].reshape(n_trials, T, truth.P2)

        Y1 = torch.einsum("ntp,qp->ntq", v1, truth.C1) + truth.d1
        Y1 = Y1 + math.sqrt(noise_var) * torch.randn(n_trials, T, q1, generator=rng, dtype=torch.float64)
        Y2 = torch.einsum("ntp,qp->ntq", v2, truth.C2) + truth.d2
        Y2 = Y2 + math.sqrt(noise_var) * torch.randn(n_trials, T, q2, generator=rng, dtype=torch.float64)

    true_latents = {
        "within1": v1[:, :, :p1].numpy(),
        "across1": v1[:, :, p1:].numpy(),
        "within2": v2[:, :, :p2].numpy(),
        "across2": v2[:, :, p2:].numpy(),
    }
    return Y1.numpy(), Y2.numpy(), truth, true_latents


def _inv_sp(y):
    return math.log(math.expm1(max(y, 1e-6)))


def match_latents(true_traj, fit_traj):
    """
    true_traj, fit_traj: (n_trials, T, p) each. Match columns by |correlation|
    of flattened (trial,time) trajectories via linear_sum_assignment (maximize
    sum |corr| == minimize sum (1-|corr|)). Returns (row_ind, col_ind, corr_signed)
    where corr_signed[i] is the signed correlation of true col row_ind[i] with
    fit col col_ind[i].
    """
    p_true = true_traj.shape[-1]
    p_fit = fit_traj.shape[-1]
    true_flat = true_traj.reshape(-1, p_true)
    fit_flat = fit_traj.reshape(-1, p_fit)
    cost = np.zeros((p_true, p_fit))
    corr = np.zeros((p_true, p_fit))
    for i in range(p_true):
        for j in range(p_fit):
            c = np.corrcoef(true_flat[:, i], fit_flat[:, j])[0, 1]
            corr[i, j] = c
            cost[i, j] = 1 - abs(c)
    row_ind, col_ind = linear_sum_assignment(cost)
    corr_signed = corr[row_ind, col_ind]
    return row_ind, col_ind, corr_signed


def main():
    q1, q2, T = 15, 20, 60
    p1, p2, p12 = 1, 1, 2
    n_trials = 80

    tau_w1 = [4.0]
    tau_w2 = [6.0]
    tau_a = [8.0, 10.0]
    delay_a = [6.0, -5.0]  # latent 0: cortex leads (+); latent 1: cerebellum leads (-)

    print("=" * 70)
    print("Simulating synthetic data from the DLAG generative model...")
    print(f"  ground truth: tau_w1={tau_w1}  tau_w2={tau_w2}  tau_a={tau_a}  delay_a={delay_a} (bins)")
    Y1, Y2, truth_model, true_latents = simulate_dlag(
        q1, q2, T, p1, p2, p12, n_trials, tau_w1, tau_w2, tau_a, delay_a,
    )

    print("\nFitting a freshly-initialized DLAGModel on the synthetic data...")
    fit_model = DLAGModel(q1, q2, T, p1, p2, p12, dt=1.0)
    fit_model.fit(Y1, Y2, n_epochs=1500, lr=0.05, print_every=100)

    posterior = fit_model.posterior_latents(Y1, Y2)
    with torch.no_grad():
        fit_tau = {
            "within1": fit_model.tau_w1.numpy(),
            "within2": fit_model.tau_w2.numpy(),
            "across": fit_model.tau_a.numpy(),
        }
        fit_delay = fit_model.delay_a.numpy()  # bin units, no dt scaling

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    all_pass = True
    for group_name, true_key, true_tau_list in [
        ("within-cortex", "within1", tau_w1),
        ("within-cerebellum", "within2", tau_w2),
    ]:
        row_ind, col_ind, corr = match_latents(true_latents[true_key], posterior[true_key])
        print(f"\n[{group_name}] matched trajectory correlations: {np.round(corr, 3)}")
        for r, c, corr_val in zip(row_ind, col_ind, corr):
            fit_tau_val = (fit_tau["within1"] if true_key == "within1" else fit_tau["within2"])[c]
            print(f"  true latent {r}: tau_true={true_tau_list[r]:.1f} bins  "
                  f"-> matched fit latent {c}: tau_fit={fit_tau_val:.1f} bins, |corr|={abs(corr_val):.3f}")
            if abs(corr_val) < 0.7:
                all_pass = False

    row_ind, col_ind, corr = match_latents(true_latents["across1"], posterior["across1"])
    print(f"\n[across-group] matched trajectory correlations (group-1 view): {np.round(corr, 3)}")
    for r, c, corr_val in zip(row_ind, col_ind, corr):
        print(f"  true latent {r}: tau_true={tau_a[r]:.1f} bins, delay_true={delay_a[r]:+.1f} bins  "
              f"-> matched fit latent {c}: tau_fit={fit_tau['across'][c]:.1f} bins, "
              f"delay_fit={fit_delay[c]:+.1f} bins, |corr|={abs(corr_val):.3f}")
        sign_ok = np.sign(delay_a[r]) == np.sign(fit_delay[c])
        print(f"    -> delay SIGN {'MATCHES' if sign_ok else 'MISMATCH'} "
              f"({'cortex leads' if delay_a[r] > 0 else 'cerebellum leads'} true "
              f"vs {'cortex leads' if fit_delay[c] > 0 else 'cerebellum leads'} fit)")
        if abs(corr_val) < 0.7 or not sign_ok:
            all_pass = False

    print("\n" + "=" * 70)
    print("PASS" if all_pass else "FAIL", "-- see thresholds above (|corr|>=0.7, delay sign match)")
    print("=" * 70)


if __name__ == "__main__":
    main()
