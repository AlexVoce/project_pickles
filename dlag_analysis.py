"""
dlag_analysis.py

Entry point for fitting DLAG (dlag_model.DLAGModel) between cortex (npix,
group 1) and cerebellum (zF, group 2) on this project's touch-trial data,
mirroring CCA_analysis.py / xfads_analysis.py's data-loading and results-dict
conventions.

DLAG requires TRUE simultaneously-recorded, trial-paired populations (not
CCA_analysis's cross-animal build_pseudopopulation), so this operates on a
single organisedDataset from one organise_dataset(touchDict) call.

    from pathlib import Path
    import jsonpickle
    from organise_touch_trials import organise_dataset
    from dlag_analysis import collect_dlag_trials, run_dlag, plot_dlag_loss

    touchDict = jsonpickle.decode(open(dataPath/"sg024/sg024_2026-02-24_v2_touchDict_aligned.json").read())
    organised = organise_dataset(touchDict)
    cond = collect_dlag_trials(organised)          # pools every "*_cor" condition
    result = run_dlag(cond, p1=2, p2=4, p12=3)
    plot_dlag_loss(result)

Dimensionality (p1, p2, p12) is HAND-CHOSEN here, not cross-validated --
cross-validated dimensionality selection (as in the DLAG paper) is a
follow-up, not built in this pass. Always run dlag_synthetic_test.py first to
confirm the model implementation itself is correct before trusting outputs
from this module on real data.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from CCA_analysis import finite_feature_mask, fit_scaler, apply_scaler, flatten_trials
from dlag_model import DLAGModel


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_dlag_trials(organised, cor_only=True, name="All Cor"):
    """
    Pool trials from `organised` (an organisedDataset, one session/animal --
    NOT a build_pseudopopulation result) into a single ConditionData for a
    DLAG fit. By default pools every condition whose name ends in "_cor"
    (all correct trials, regardless of cue/port side); per-trial
    cueSide/portSide/rewardSide metadata is preserved on the returned
    ConditionData for post-hoc grouping of latent trajectories.
    """
    if cor_only:
        cond_names = [c for c in organised.keys() if c.endswith("_cor")]
    else:
        cond_names = list(organised.keys())
    return organised.pool(cond_names, name=name)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_dlag(
    cond,                    # ConditionData, e.g. from collect_dlag_trials
    p1=2,                    # within-cortex latents
    p2=4,                    # within-cerebellum latents
    p12=3,                   # across-group (shared) latents
    scale=True,              # z-score npix (X) only; zF (Y) is already z-scored
    test_size=0.2,
    n_epochs=1000,
    lr=0.05,
    max_delay_bins=None,     # defaults to DLAGModel's T/4
    init_tau_bins=5.0,
    time=None,               # real time axis (organised.time_zF), for dt; sample-index dt=1 if None
    random_state=42,
    verbose=True,
):
    """
    Fit DLAG between cond.npix_ds (group 1, cortex) and cond.zF (group 2,
    cerebellum). Standardizes npix (train-trials-only StandardScaler, zF left
    unscaled -- same convention as CCA_analysis.py), splits trials into
    train/test, fits DLAGModel on train, then computes posterior latents for
    ALL trials (train+test) so downstream trajectory plots aren't limited to
    the training set.

    Returns a dict:
        model               fitted DLAGModel
        scaler_x            fitted StandardScaler for npix (None if scale=False)
        good_npix, good_zF  finite-feature masks applied before fitting
        train_idx, test_idx trial indices into `cond`
        posterior           dict from DLAGModel.posterior_latents, ALL trials
        loss_history         training NLL/trial per epoch
        test_nll            held-out NLL/trial (generalization sanity check)
        delays_ms, timescales_ms, shared_variance    fitted-model summaries
        dt                  seconds per time bin used for ms conversions
        cond, config
    """
    X = np.asarray(cond.npix_ds, dtype=float)   # (n_trials, T, q1) cortex
    Y = np.asarray(cond.zF, dtype=float)         # (n_trials, T, q2) cerebellum
    n_trials, T, _ = X.shape

    dt = float(time[1] - time[0]) if time is not None else 1.0

    train_idx, test_idx = train_test_split(
        np.arange(n_trials), test_size=test_size, random_state=random_state,
    )

    X_train_flat = flatten_trials(X[train_idx])
    Y_train_flat = flatten_trials(Y[train_idx])
    good_npix = finite_feature_mask(X_train_flat)
    good_zF = finite_feature_mask(Y_train_flat)

    scaler_x, X_train_z = fit_scaler(X_train_flat[:, good_npix], scale=scale)
    Y_train_z = Y_train_flat[:, good_zF]  # not scaling zF -- already z-scored

    q1, q2 = X_train_z.shape[1], Y_train_z.shape[1]
    Y1_train = X_train_z.reshape(len(train_idx), T, q1)
    Y2_train = Y_train_z.reshape(len(train_idx), T, q2)

    if verbose:
        print(f"DLAG fit: cortex(npix) q1={q1}, cerebellum(zF) q2={q2}, T={T}, "
              f"n_train={len(train_idx)}, n_test={len(test_idx)}, "
              f"latents p1={p1} p2={p2} p12={p12}")

    model = DLAGModel(
        q1=q1, q2=q2, T=T, p1=p1, p2=p2, p12=p12, dt=dt,
        max_delay_bins=max_delay_bins, init_tau_bins=init_tau_bins,
    )
    loss_history = model.fit(Y1_train, Y2_train, n_epochs=n_epochs, lr=lr, verbose=verbose)

    # apply the SAME feature masks/scaler to every trial (train+test) for posterior inference
    X_all_flat = flatten_trials(X)[:, good_npix]
    Y_all_flat = flatten_trials(Y)[:, good_zF]
    X_all_z = apply_scaler(X_all_flat, scaler_x)
    Y1_all = X_all_z.reshape(n_trials, T, q1)
    Y2_all = Y_all_flat.reshape(n_trials, T, q2)

    posterior = model.posterior_latents(Y1_all, Y2_all)

    test_nll = None
    if len(test_idx) > 0:
        Y1_test, Y2_test = Y1_all[test_idx], Y2_all[test_idx]
        test_nll = model.neg_log_likelihood(
            __import__("torch").as_tensor(Y1_test, dtype=model._dtype),
            __import__("torch").as_tensor(Y2_test, dtype=model._dtype),
        ).item()
        if verbose:
            print(f"Held-out test NLL/trial: {test_nll:.3f} "
                  f"(train final: {loss_history[-1]:.3f})")

    config = dict(p1=p1, p2=p2, p12=p12, scale=scale, test_size=test_size,
                  n_epochs=n_epochs, lr=lr, random_state=random_state, dt=dt)

    return {
        "model": model,
        "scaler_x": scaler_x,
        "good_npix": good_npix,
        "good_zF": good_zF,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "posterior": posterior,
        "loss_history": loss_history,
        "test_nll": test_nll,
        "delays_ms": model.delays_ms(),
        "timescales_ms": model.timescales_ms(),
        "shared_variance": model.shared_variance_explained(),
        "dt": dt,
        "cond": cond,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_dlag_summary(result):
    """Print fitted timescales, across-group delays (with direction), and shared-variance fractions."""
    ts = result["timescales_ms"]
    delays = result["delays_ms"]
    sv = result["shared_variance"]
    dt = result["dt"]
    unit = "ms" if dt != 1.0 else "bins"

    print(f"Within-cortex timescales ({unit}):     {np.round(ts['within1'], 1)}")
    print(f"Within-cerebellum timescales ({unit}): {np.round(ts['within2'], 1)}")
    print(f"Across-group timescales ({unit}):      {np.round(ts['across'], 1)}")
    print(f"Across-group delays ({unit}) [+ = cortex leads, - = cerebellum leads]:")
    for k, d in enumerate(delays):
        direction = "cortex leads" if d > 0 else "cerebellum leads"
        print(f"  latent {k}: {d:+.1f} {unit}  ({direction})")
    print(f"Fraction of cortex signal variance from across-group latents:      "
          f"{sv['group1_frac_across_of_signal']:.3f}")
    print(f"Fraction of cerebellum signal variance from across-group latents:  "
          f"{sv['group2_frac_across_of_signal']:.3f}")


def plot_dlag_loss(result, ax=None, figsize=(6, 4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.plot(result["loss_history"])
    if result["test_nll"] is not None:
        ax.axhline(result["test_nll"], color="orange", linestyle="--", label="held-out test NLL")
        ax.legend(frameon=False)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL / trial")
    ax.set_title("DLAG training loss")
    fig.tight_layout()
    return fig, ax


def plot_dlag_delay_summary(result, figsize=(6, 4), ax=None):
    """Bar chart of fitted across-group delays, one bar per across-group latent."""
    delays = result["delays_ms"]
    dt = result["dt"]
    unit = "ms" if dt != 1.0 else "bins"
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    colors = ["tab:blue" if d > 0 else "tab:red" for d in delays]
    ax.bar(np.arange(len(delays)), delays, color=colors)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(np.arange(len(delays)))
    ax.set_xticklabels([f"latent {k}" for k in range(len(delays))])
    ax.set_ylabel(f"Delay ({unit})\n+ cortex leads / - cerebellum leads")
    ax.set_title("DLAG across-group delays")
    fig.tight_layout()
    return fig, ax


def plot_dlag_latent_timecourse(
    result,
    which="across1",       # "within1", "within2", "across1", "across2"
    component=0,
    group_by="cueSide",    # attribute on result["cond"] to group/color trials by, or None
    time=None,
    show_sem=True,
    figsize=(7, 5),
    title=None,
    ax=None,
):
    """
    Plot posterior mean latent trajectory over time, averaged across trials
    (optionally split by a per-trial metadata field on result["cond"], e.g.
    "cueSide"/"portSide" -- mirrors CCA_analysis.plot_cca_component_timecourse).
    """
    traj = result["posterior"][which][:, :, component]  # (n_trials, T)
    cond = result["cond"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    t = np.asarray(time) if time is not None else np.arange(traj.shape[1])

    if group_by is None:
        groups = {"all": np.arange(traj.shape[0])}
    else:
        labels = np.asarray(getattr(cond, group_by))
        groups = {lbl: np.where(labels == lbl)[0] for lbl in np.unique(labels) if lbl is not None}

    for name, idx in groups.items():
        if len(idx) == 0:
            continue
        sub = traj[idx]
        mean = np.nanmean(sub, axis=0)
        line, = ax.plot(t, mean, label=str(name), linewidth=2)
        if show_sem:
            sem = np.nanstd(sub, axis=0) / np.sqrt(len(idx))
            ax.fill_between(t, mean - sem, mean + sem, alpha=0.2, color=line.get_color())

    ax.axhline(0, linewidth=1, linestyle="--", alpha=0.4, color="gray")
    ax.set_xlabel("Time (s)" if time is not None else "Timepoint")
    ax.set_ylabel(f"{which} latent {component}")
    ax.set_title(title or f"DLAG {which} latent {component}")
    if group_by is not None:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax
