# xfads_latent_dynamics.py
"""
Post-hoc analyses of a fitted XFADS model's latents (the dict returned by
xfads_analysis.run_xfads), organized around four questions:

1. plot_latent_timecourses / pca_reduce_latents
       What do the latents look like -- how does population state evolve
       over the trial?

2. compare_condition_subspaces / condition_axis_stability
       Do e.g. Left and Right reaches occupy different regions of latent
       space, or a shared rotation? compare_condition_subspaces asks whether
       the two conditions' trajectories live in the same subspace (principal
       angles) vs. offset regions within it (centroid separation);
       condition_axis_stability asks whether the direction separating them
       is fixed over the trial or itself rotates.

3. decode_condition_over_time / plot_decoding_curve
       Forecasting: a cross-validated, permutation-tested decoding-accuracy-
       vs-time curve answers "how early can we predict reach direction from
       population state."

4. simulate_forward_from_ic
       Churchland/Shenoy's dynamical-systems proposal: does preparatory
       activity set an initial condition that a SHARED, condition-invariant
       dynamical system (XFADS pools all conditions into one dynamics_mod by
       construction) then autonomously plays out? Rolls the fitted dynamics
       forward from each trial's smoothed state at an early "IC" time with
       NO further observation input (ssm.predict_forward), producing a
       counterfactual "dynamics-only" trajectory you can compare against the
       real, observation-informed one -- e.g. by running (2) or (3) on both
       and seeing whether the dynamics-only rollout still separates
       conditions as well as the real trajectory does.

All functions operate on the plain dict `result` returned by run_xfads (or,
for decode_condition_over_time and plot_xfads_latents, on any
(n_trials, n_time, n_latents) array + matching labels -- including the
counterfactual array from simulate_forward_from_ic) -- nothing here refits
or mutates the model.

Typical usage (continuing from a notebook that already has `result` from
xfads_analysis.run_xfads and organised.time_zF as the shared time axis):

    from xfads_latent_dynamics import *

    # 1. look at the latents
    plot_latent_timecourses(result, time=organised.time_zF)
    pca_res = pca_reduce_latents(result)
    plot_xfads_latents(pca_res, time=organised.time_zF)   # from xfads_analysis

    # 2. different regions or shared rotation?
    compare_condition_subspaces(result, "Left Cor", "Right Cor")
    condition_axis_stability(result, "Left Cor", "Right Cor", time=organised.time_zF)

    # 3. how early can we decode reach direction?
    dec = decode_condition_over_time(result["latents"], result["labels"],
                                      "Left Cor", "Right Cor", time=organised.time_zF)
    plot_decoding_curve(dec)

    # 4. does a shared dynamical system, from an early initial condition,
    #    reproduce that separation on its own?
    sim_latents, ic_idx = simulate_forward_from_ic(result, ic_time=-0.3, time=organised.time_zF)
    sim_dec = decode_condition_over_time(sim_latents, result["labels"],
                                          "Left Cor", "Right Cor", time=organised.time_zF)
    plot_decoding_curve(dec)       # observed
    plot_decoding_curve(sim_dec)   # dynamics-only from t=-0.3s
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# 1. What do the latents look like?
# ---------------------------------------------------------------------------

def pca_reduce_latents(result, n_components=None, fit_on="all"):
    """
    PCA-reduce XFADS latents for visualization/ranking. Raw XFADS latent
    dims aren't variance-ordered the way PCA/CCA components are -- there's
    no orthogonality constraint on the GRU dynamics' state -- so this gives
    a ranked, orthogonal view for state-space plots and picking which dims
    to look at.

    fit_on : "all" (default), or a condition name to fit the PCA on just
        that condition's trials (still transforms every trial either way).

    Returns a dict shaped like `result` (with "latents" and
    "latents_by_cond") so it can be passed straight to
    xfads_analysis.plot_xfads_latents, plus "pca" (the fitted sklearn PCA)
    and "explained_variance_ratio".
    """
    latents = result["latents"]
    n_trials, n_time, n_latents = latents.shape
    flat = latents.reshape(-1, n_latents)

    if fit_on == "all":
        fit_rows = flat
    else:
        fit_rows = latents[result["labels"] == fit_on].reshape(-1, n_latents)

    pca = PCA(n_components=n_components or n_latents)
    pca.fit(fit_rows)
    latents_pca = pca.transform(flat).reshape(n_trials, n_time, -1)

    labels = result["labels"]
    latents_pca_by_cond = {name: latents_pca[labels == name] for name in dict.fromkeys(labels)}

    return {
        "pca": pca,
        "labels": labels,
        "latents": latents_pca,
        "latents_by_cond": latents_pca_by_cond,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def plot_latent_timecourses(result, conditions=None, dims=None, time=None,
                             ncols=4, figsize=None, show_sem=True):
    """
    Grid of per-latent-dimension timecourses (trial mean +/- SEM), one line
    per condition -- the population-state-evolution view of XFADS's
    latents, analogous to CCA_analysis.plot_cca_component_timecourse but for
    every latent dim at once.
    """
    latents_by_cond = result["latents_by_cond"]
    names = list(latents_by_cond.keys()) if conditions is None else [
        c.name if hasattr(c, "name") else str(c) for c in conditions
    ]
    n_latents = next(iter(latents_by_cond.values())).shape[-1]
    dims = list(range(n_latents)) if dims is None else list(dims)

    ncols = max(1, min(ncols, len(dims)))
    nrows = int(np.ceil(len(dims) / ncols))
    if figsize is None:
        figsize = (3.2 * ncols, 2.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for i, d in enumerate(dims):
        ax = axes[i]
        for name in names:
            trace = latents_by_cond[name][:, :, d]
            mean = np.nanmean(trace, axis=0)
            t = np.asarray(time) if time is not None else np.arange(mean.shape[0])
            line, = ax.plot(t, mean, label=name, linewidth=1.5)
            if show_sem:
                n_valid = np.sum(np.isfinite(trace), axis=0)
                sem = np.nanstd(trace, axis=0) / np.sqrt(np.maximum(n_valid, 1))
                ax.fill_between(t, mean - sem, mean + sem, alpha=0.2, color=line.get_color())
        ax.axhline(0, linewidth=0.8, linestyle="--", alpha=0.4, color="gray")
        if time is not None:
            ax.axvline(0, linewidth=0.8, linestyle=":", alpha=0.4, color="black")
        ax.set_title(f"Latent {d + 1}", fontsize=9)

    for ax in axes[len(dims):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Time (s)" if time is not None else "Timepoint")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 2. Different regions, or a shared rotation?
# ---------------------------------------------------------------------------

def compare_condition_subspaces(result, cond_a, cond_b, n_components=3, verbose=True):
    """
    Do two conditions' latent trajectories occupy different regions of
    state space, or the same subspace traversed via a shared rotation?

    For each condition, the trial-averaged trajectory (n_time, n_latents) is
    centered on its own time-mean and its top `n_components` directions of
    temporal variation (via SVD) are taken as that condition's "trajectory
    subspace". The principal (subspace) angles between the two conditions'
    subspaces are near 0 deg if they share a subspace (consistent with a
    shared rotation/dynamics acting on different initial conditions) and
    near 90 deg if the subspaces are essentially unrelated (consistent with
    genuinely different population states).

    Also reports centroid separation relative to within-condition trial
    spread, since two conditions can share a subspace while still occupying
    offset regions within it (rotation + translation, not pure rotation).
    """
    latents_by_cond = result["latents_by_cond"]
    traj_a = np.nanmean(latents_by_cond[cond_a], axis=0)  # (n_time, n_latents)
    traj_b = np.nanmean(latents_by_cond[cond_b], axis=0)

    def subspace_basis(traj, k):
        centered = traj - traj.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        return Vt[:k].T  # (n_latents, k), orthonormal columns

    basis_a = subspace_basis(traj_a, n_components)
    basis_b = subspace_basis(traj_b, n_components)
    angles_deg = np.degrees(subspace_angles(basis_a, basis_b))

    centroid_a = latents_by_cond[cond_a].mean(axis=(0, 1))
    centroid_b = latents_by_cond[cond_b].mean(axis=(0, 1))
    centroid_dist = np.linalg.norm(centroid_a - centroid_b)

    pooled = np.concatenate([
        latents_by_cond[cond_a].reshape(-1, latents_by_cond[cond_a].shape[-1]),
        latents_by_cond[cond_b].reshape(-1, latents_by_cond[cond_b].shape[-1]),
    ], axis=0)
    pooled_spread = np.linalg.norm(pooled.std(axis=0))
    separation_index = centroid_dist / (pooled_spread + 1e-8)

    if verbose:
        print(f"Principal angles between {cond_a!r}/{cond_b!r} trajectory subspaces "
              f"(top {n_components} dims): {np.round(angles_deg, 1)} deg")
        print(f"Centroid separation / pooled trial spread: {separation_index:.2f}")
        if np.all(angles_deg < 30):
            print("-> Small angles: conditions largely share a trajectory subspace "
                  "(consistent with a shared rotation/dynamics on different initial conditions).")
        elif np.all(angles_deg > 60):
            print("-> Large angles: conditions occupy largely distinct/orthogonal subspaces "
                  "(consistent with genuinely different population states).")
        else:
            print("-> Mixed: partial subspace overlap -- some shared directions of variation, some distinct.")

    return {
        "angles_deg": angles_deg,
        "centroid_distance": centroid_dist,
        "pooled_spread": pooled_spread,
        "separation_index": separation_index,
        "basis_a": basis_a,
        "basis_b": basis_b,
    }


def condition_axis_stability(result, cond_a, cond_b, time=None, ref_time=0.0, figsize=(6, 4), ax=None):
    """
    Track how the cond_a-vs-cond_b separating direction
    (mean_a(t) - mean_b(t)) rotates over the trial, via cosine similarity to
    the separating direction at `ref_time` (default t=0, e.g. touch/reach
    onset). A flat curve near 1 means a fixed separating axis throughout the
    trial; a curve that drops away from 1 means the separating direction
    itself changes over time (more consistent with a shared, rotating
    dynamics than a static offset between conditions).
    """
    latents_by_cond = result["latents_by_cond"]
    mean_a = np.nanmean(latents_by_cond[cond_a], axis=0)  # (n_time, n_latents)
    mean_b = np.nanmean(latents_by_cond[cond_b], axis=0)
    diff = mean_a - mean_b

    t = np.asarray(time) if time is not None else np.arange(diff.shape[0])
    ref_idx = int(np.argmin(np.abs(t - ref_time)))
    ref_vec = diff[ref_idx]
    ref_unit = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)

    diff_unit = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8)
    cos_sim = diff_unit @ ref_unit

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.plot(t, cos_sim, linewidth=2)
    ax.axhline(1, linestyle="--", color="gray", alpha=0.4)
    ax.axhline(0, linestyle="--", color="gray", alpha=0.4)
    ax.axvline(t[ref_idx], linestyle=":", color="black", alpha=0.5)
    ax.set_xlabel("Time (s)" if time is not None else "Timepoint")
    ax.set_ylabel(f"cos(axis(t), axis(t={t[ref_idx]:.2f}))")
    ax.set_title(f"Stability of {cond_a} vs {cond_b} separating axis")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()

    return {"time": t, "cos_sim": cos_sim, "diff_trajectory": diff, "ref_idx": ref_idx}, fig, ax


# ---------------------------------------------------------------------------
# 3. Forecasting -- how early can we decode behavior?
# ---------------------------------------------------------------------------

def decode_condition_over_time(
    latents, labels, cond_a, cond_b, time=None,
    n_folds=5, n_permutations=200, alpha=0.05, min_consecutive=3,
    random_state=0, verbose=True,
):
    """
    Time-resolved binary decoding of condition (e.g. reach direction) from
    latent state: one cross-validated logistic regression per timepoint,
    giving a decoding-accuracy-vs-time curve -- "how early can we predict
    behavior" from population state.

    latents : (n_trials, n_time, n_latents) -- e.g. result["latents"], or a
        counterfactual array such as simulate_forward_from_ic's output, to
        ask the same question of that.
    labels : (n_trials,) condition-name array aligned with `latents`.
    cond_a, cond_b : the two condition names to decode between (only trials
        from these two are used).

    Significance is via label permutation (n_permutations shuffles): a
    timepoint counts as decodable only if its real accuracy exceeds the
    (1-alpha) percentile of its own null distribution for at least
    `min_consecutive` consecutive timepoints (guards against single noisy
    timepoints). This is the slow part of this function -- lower
    n_permutations for a quick look, raise it for a number you'd report.

    Returns dict: time, accuracy, null_thresh (per timepoint), chance_level,
    first_decodable_time (or None).
    """
    mask = np.isin(labels, [cond_a, cond_b])
    X = latents[mask]
    y = (labels[mask] == cond_a).astype(int)
    n_time = X.shape[1]
    t = np.asarray(time) if time is not None else np.arange(n_time)

    n_folds_eff = min(n_folds, np.bincount(y).min())
    if n_folds_eff < 2:
        raise ValueError(f"Not enough trials per condition for cross-validation (counts: {np.bincount(y)}).")

    def cv_accuracy(X_t, y_):
        skf = StratifiedKFold(n_splits=n_folds_eff, shuffle=True, random_state=random_state)
        accs = []
        for train_idx, test_idx in skf.split(X_t, y_):
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_t[train_idx], y_[train_idx])
            accs.append(clf.score(X_t[test_idx], y_[test_idx]))
        return np.mean(accs)

    accuracy = np.array([cv_accuracy(X[:, ti, :], y) for ti in range(n_time)])

    rng = np.random.RandomState(random_state)
    null = np.zeros((n_permutations, n_time))
    for p in range(n_permutations):
        y_perm = rng.permutation(y)
        for ti in range(n_time):
            null[p, ti] = cv_accuracy(X[:, ti, :], y_perm)
    null_thresh = np.percentile(null, 100 * (1 - alpha), axis=0)

    suprathresh = accuracy > null_thresh
    first_decodable_time = None
    run = 0
    for ti in range(n_time):
        run = run + 1 if suprathresh[ti] else 0
        if run >= min_consecutive:
            first_decodable_time = t[ti - min_consecutive + 1]
            break

    if verbose:
        chance = max(np.mean(y), 1 - np.mean(y))
        print(f"Chance level (majority class): {chance:.2f}")
        print(f"Peak decoding accuracy: {accuracy.max():.2f} at t={t[np.argmax(accuracy)]:.3f}")
        print(f"First sustained above-chance decoding: "
              f"{'t=' + format(first_decodable_time, '.3f') if first_decodable_time is not None else 'none found'}")

    return {
        "time": t, "accuracy": accuracy, "null_thresh": null_thresh,
        "chance_level": max(np.mean(y), 1 - np.mean(y)),
        "first_decodable_time": first_decodable_time,
    }


def plot_decoding_curve(decode_result, ax=None, figsize=(7, 4), title=None, label="decoding accuracy"):
    t = decode_result["time"]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    line, = ax.plot(t, decode_result["accuracy"], linewidth=2, label=label)
    ax.plot(t, decode_result["null_thresh"], linestyle="--", color=line.get_color(), alpha=0.5)
    ax.axhline(decode_result["chance_level"], linestyle=":", color="black", alpha=0.5)
    if decode_result["first_decodable_time"] is not None:
        ax.axvline(decode_result["first_decodable_time"], color=line.get_color(), alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cross-validated accuracy")
    ax.set_title(title or "Decoding condition from latent state over time")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Does a shared dynamical system carry the initial condition forward?
# ---------------------------------------------------------------------------

def simulate_forward_from_ic(result, ic_time, time=None, n_forward_samples=50, seed=0):
    """
    Churchland/Shenoy-style test: does preparatory activity set an initial
    condition that a SHARED, condition-invariant dynamical system then
    autonomously plays out? XFADS already fits ONE dynamics_mod pooled
    across every condition (run_xfads's "shared" pooling), so the question
    isn't whether the dynamics are literally shared (they are, by
    construction) but whether that shared dynamics, evolved forward from
    each trial's own early state with NO further observation input, is
    enough on its own to reproduce what actually happened later in the
    trial.

    Takes each trial's SMOOTHED (observation-informed) latent state at
    `ic_time`, then rolls the dynamics forward via ssm.predict_forward --
    which uses only the learned mean_fn (z_t = f(z_t-1)) plus process noise,
    no observations -- for the rest of the trial. Averaging
    `n_forward_samples` stochastic rollouts approximates the deterministic
    mean flow.

    The result is a counterfactual "dynamics-only" trajectory, shaped like
    result["latents"], that you can feed to decode_condition_over_time or
    plot_xfads_latents (after building a matching latents_by_cond) to
    compare against the real, observation-informed trajectory:
        - if dynamics-only reproduces the real trajectory's condition
          separation, that supports the initial-condition hypothesis.
        - if dynamics-only collapses toward a shared path while the real
          trajectory stays separated, ongoing condition-specific input
          (beyond the t=ic_time state) is doing real work.

    Returns simulated, ic_idx
        simulated : (n_trials, n_time, n_latents), identical to
            result["latents"] up to and including ic_time, dynamics-only
            beyond it.
        ic_idx : the timepoint index ic_time resolved to.
    """
    ssm = result["ssm"]
    latents = result["latents"]
    n_trials, n_time_total, n_latents = latents.shape
    t = np.asarray(time) if time is not None else np.arange(n_time_total)
    ic_idx = int(np.argmin(np.abs(t - ic_time)))
    n_bins_forward = n_time_total - ic_idx - 1
    if n_bins_forward <= 0:
        raise ValueError(f"ic_time={ic_time} (index {ic_idx}) leaves no timepoints left to forecast.")

    torch.manual_seed(seed)
    z_ic = torch.tensor(latents[:, ic_idx, :], dtype=torch.float32)
    z_ic_rep = z_ic.unsqueeze(0).repeat(n_forward_samples, 1, 1)  # (n_forward_samples, n_trials, n_latents)

    ssm.eval()
    with torch.no_grad():
        z_forward = ssm.predict_forward(z_ic_rep, n_bins_forward)  # (n_forward_samples, n_trials, n_bins_forward, n_latents)
    z_forward_mean = z_forward.mean(dim=0).cpu().numpy()

    simulated = latents.copy()
    simulated[:, ic_idx + 1:, :] = z_forward_mean
    return simulated, ic_idx


def latents_by_cond_from_array(latents, labels):
    """Small helper: split any (n_trials, n_time, n_latents) array (e.g.
    simulate_forward_from_ic's output) by condition label, for use with
    xfads_analysis.plot_xfads_latents (which expects result["latents_by_cond"])."""
    return {"latents": latents, "labels": labels,
            "latents_by_cond": {name: latents[labels == name] for name in dict.fromkeys(labels)}}
