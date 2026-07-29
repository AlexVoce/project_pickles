# xfads_analysis.py
"""
Fit XFADS (catniplab/xfads -- a deep nonlinear Gaussian state-space model,
https://github.com/catniplab/xfads) to cortical (npix) trial data.

Mirrors CCA_analysis.py's data-loading conventions so the two analyses can be
run side by side from the same organised dataset / notebook:

    conditions : dict {name: ConditionData}, or a list of ConditionData
                 (each object's `.name` is used as the key), or a single
                 ConditionData. Trial-level arrays are
                 (n_trials, n_timepoints, n_features).

Unlike CCA (which relates cortex X to cerebellum Y), XFADS models a single
population's own latent dynamics -- only the X side (npix_ds by default) is
used here.

Main entry point: run_xfads(conditions, ...)

Install (once -- not done automatically by this script):
    pip install git+https://github.com/catniplab/xfads.git
This pulls in its own pinned torch / pytorch-lightning / hydra-core
(omegaconf) / scikit-learn / etc. If that conflicts with packages already
installed in this environment, use a fresh virtualenv/conda env as the xfads
README recommends (it targets Python >=3.9).

Observation model: npix_ds is continuous (already /100-scaled and
downsampled onto zF's time grid in organise_touch_trials.py), not raw
integer spike counts, so this module fits a GaussianLikelihood. If you want
a Poisson count model instead, feed it raw npix_touchMat trial counts
(before the /100 scaling and downsampling) and swap in
xfads.ssm_modules.likelihoods.PoissonLikelihood -- the assembly code below
would need the corresponding readout/likelihood swap.
"""
try:
    import torch
    import pytorch_lightning as lightning
    from omegaconf import OmegaConf
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    import xfads.utils as xfads_utils
    from xfads.ssm_modules.dynamics import DenseGaussianDynamics, DenseGaussianInitialCondition
    from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
    from xfads.ssm_modules.likelihoods import GaussianLikelihood
    from xfads.smoothers.nonlinear_smoother import NonlinearFilter, LowRankNonlinearStateSpaceModel
    from xfads.smoothers.lightning_trainers import LightningNonlinearSSM
except ImportError as e:
    raise ImportError(
        "xfads_analysis.py needs the xfads package, which isn't installed. Install it with:\n"
        "    pip install git+https://github.com/catniplab/xfads.git\n"
        "(pulls in pytorch-lightning, hydra-core/omegaconf, etc. -- use a fresh "
        "environment if that conflicts with what's already installed here)."
    ) from e

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from CCA_analysis import as_named_conditions, finite_feature_mask


# ---------------------------------------------------------------------------
# Data collection (X/npix side only -- see CCA_analysis.collect_trials for
# the X+Y version this is adapted from)
# ---------------------------------------------------------------------------

def collect_npix_trials(conditions, x_attr="npix_ds", verbose=True):
    """
    Pull trial-level cortical (X) arrays off one or more ConditionData
    objects and concatenate them, tagging each trial with the name of the
    condition it came from.

    conditions : a single ConditionData, dict {name: ConditionData}, or list
                 of ConditionData.

    Returns X_trials, labels
        X_trials : (n_trials_total, n_timepoints, n_features)
        labels   : (n_trials_total,) object array of condition names
    """
    if hasattr(conditions, "name") and not isinstance(conditions, dict):
        conditions = {conditions.name: conditions}
    conditions = as_named_conditions(conditions)
    if len(conditions) < 1:
        raise ValueError("collect_npix_trials needs at least 1 condition.")

    X_list, label_list = [], []
    for name, cond in conditions.items():
        X_c = np.asarray(getattr(cond, x_attr), dtype=float)
        X_list.append(X_c)
        label_list.append(np.full(X_c.shape[0], name, dtype=object))

    X_trials = np.concatenate(X_list, axis=0)
    labels = np.concatenate(label_list)

    if verbose:
        print(f"X trials: {X_trials.shape}")
        print(", ".join(f"{name}: {np.sum(labels == name)} trials" for name in conditions))

    return X_trials, labels


# ---------------------------------------------------------------------------
# Model assembly -- follows xfads's own examples/vdp_example (Gaussian
# likelihood, generic LightningNonlinearSSM trainer) rather than the
# monkey_reaching example, which is Poisson + behavior-decoding specific.
# ---------------------------------------------------------------------------

def _build_ssm(cfg, n_neurons):
    readout_fn = torch.nn.Linear(cfg.n_latents, n_neurons, device=cfg.device)
    R_diag = 0.5 * torch.ones(n_neurons, device=cfg.device)
    likelihood_pdf = GaussianLikelihood(readout_fn, n_neurons, R_diag, device=cfg.device)

    dynamics_fn = xfads_utils.build_gru_dynamics_function(
        cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device, use_layer_norm=cfg.use_layer_norm,
    )
    Q_diag = torch.ones(cfg.n_latents, device=cfg.device)
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
    Q_0_diag = torch.ones(cfg.n_latents, device=cfg.device)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

    backward_encoder = BackwardEncoderLRMvn(
        cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
        rank_local=cfg.rank_local, rank_backward=cfg.rank_backward, device=cfg.device,
    )
    local_encoder = LocalEncoderLRMvn(
        cfg.n_latents, n_neurons, cfg.n_hidden_local, cfg.n_latents,
        rank=cfg.rank_local, device=cfg.device, dropout=cfg.p_local_dropout,
    )
    nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device=cfg.device)

    return LowRankNonlinearStateSpaceModel(
        dynamics_mod, likelihood_pdf, initial_condition_pdf,
        backward_encoder, local_encoder, nl_filter, device=cfg.device,
    )


def _smooth_all_trials(ssm, X, n_samples, device, chunk_size=32):
    """
    Run the fitted (acausal) smoother over every trial in X, in chunks to
    bound memory. Returns posterior mean and std over the n_samples draws,
    each (n_trials, n_timepoints, n_latents).
    """
    ssm.eval()
    means, stds = [], []
    with torch.no_grad():
        for start in range(0, X.shape[0], chunk_size):
            chunk = torch.tensor(X[start:start + chunk_size], dtype=torch.float32, device=device)
            z_s, _ = ssm.fast_smooth_1_to_T(chunk, n_samples, get_kl=False)  # (n_samples, n_chunk, n_time, n_latents)
            means.append(z_s.mean(dim=0).cpu().numpy())
            stds.append(z_s.std(dim=0).cpu().numpy())
    return np.concatenate(means, axis=0), np.concatenate(stds, axis=0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_xfads(
    conditions,             # dict {name: ConditionData}, list, or single ConditionData
    x_attr="npix_ds",
    n_latents=12,
    rank_local=6,
    rank_backward=4,
    n_hidden_dynamics=64,
    n_hidden_local=64,
    n_hidden_backward=64,
    use_layer_norm=True,
    n_samples=5,             # posterior samples per forward pass (ELBO + inference) -- vectorized, cheap to lower
    p_local_dropout=0.3,
    scale=True,              # z-score each npix feature (fit on train trials only) before fitting
    val_size=0.15,           # held-out fraction of TRIALS for early stopping / checkpoint selection
    batch_sz=512,            # clamped to len(train_idx) below -- default is "full batch" for small trial counts.
                              # The filter loops sequentially over timepoints *per batch*, so fewer/bigger
                              # batches means far fewer loop invocations per epoch; lower this only if you
                              # run out of memory.
    lr=1e-3,
    lr_gamma_decay=0.999,
    n_epochs=500,
    early_stopping_patience=100,  # epochs of no valid_loss improvement before stopping; None to disable
    device=None,              # None = auto ("cuda" if available else "cpu")
    cpu_threads=4,            # torch.set_num_threads() when device == "cpu"; None to leave PyTorch's default.
                              # XFADS's filter loop is many small (rank ~4-6) sequential ops -- on Apple
                              # Silicon in particular, PyTorch's default (one thread per core) spends more
                              # time synchronizing the thread pool than it saves per tiny op. A small fixed
                              # count measured ~2.5x faster than the 10-thread default on an M4. This is a
                              # PROCESS-GLOBAL setting (affects any other torch code in the same session).
    seed=42,
    ckpt_dir="xfads_ckpts",
    log_dir="xfads_logs",
    run_name="xfads_run",
    infer_chunk_size=32,
    verbose=True,
):
    """
    Fit an XFADS Gaussian state-space model to cortical (npix) trial data
    pooled across any number of conditions, then run the fitted (acausal)
    smoother over every trial to extract per-trial latent trajectories.

    Every condition is pooled into ONE model fit (like CCA_analysis's
    fit="shared"), so all conditions land in the same latent space and can
    be compared/plotted together -- see plot_xfads_latents.

    Returns a dict:
        ssm, cfg, trainer          fitted objects / the hydra-style config used
        scaler                     fitted StandardScaler, or None if scale=False
        good_X                     finite-feature mask applied before fitting
        labels                     condition name per trial, concatenated order
        is_train                   bool mask, True for trials used in the training split
        latents, latents_std       (n_trials, n_timepoints, n_latents) posterior
                                    mean / std from the acausal smoother, all trials
        latents_by_cond            {name: array} split of `latents` by condition
        best_ckpt_path             path Lightning saved the best checkpoint to
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    lightning.seed_everything(seed, workers=True)
    torch.set_default_dtype(torch.float32)

    X_trials, labels = collect_npix_trials(conditions, x_attr=x_attr, verbose=verbose)
    n_trials_total, n_time, n_features_raw = X_trials.shape

    good_X = finite_feature_mask(X_trials.reshape(-1, n_features_raw))
    X_trials = X_trials[:, :, good_X]
    n_neurons = X_trials.shape[-1]
    if verbose:
        print(f"Using {n_neurons}/{n_features_raw} finite npix features.")

    names, counts = np.unique(labels, return_counts=True)
    can_stratify = len(names) > 1 and counts.min() >= 2
    has_val = val_size is not None and val_size > 0 and n_trials_total >= 4
    if has_val:
        train_idx, val_idx = train_test_split(
            np.arange(n_trials_total), test_size=val_size, random_state=seed,
            stratify=labels if can_stratify else None,
        )
    else:
        if verbose:
            print("Too few trials for a held-out split -- training on all trials, no early-stopping monitor.")
        train_idx, val_idx = np.arange(n_trials_total), np.array([], dtype=int)

    is_train = np.zeros(n_trials_total, dtype=bool)
    is_train[train_idx] = True

    if scale:
        scaler = StandardScaler()
        scaler.fit(X_trials[train_idx].reshape(-1, n_neurons))
        X_z = scaler.transform(X_trials.reshape(-1, n_neurons)).reshape(n_trials_total, n_time, n_neurons)
    else:
        scaler = None
        X_z = X_trials

    cfg = OmegaConf.create(dict(
        n_latents=n_latents, rank_local=rank_local, rank_backward=rank_backward,
        use_layer_norm=use_layer_norm, n_hidden_dynamics=n_hidden_dynamics,
        n_samples=n_samples, n_hidden_local=n_hidden_local, n_hidden_backward=n_hidden_backward,
        p_mask_a=0.0, p_mask_b=0.0, p_mask_apb=0.0, p_mask_y_in=0.0, p_local_dropout=p_local_dropout,
        lr=lr, lr_gamma_decay=lr_gamma_decay, n_epochs=n_epochs,
        batch_sz=max(1, min(batch_sz, len(train_idx))),
        seed=seed, device=device, data_device=device,
    ))

    y_train = torch.tensor(X_z[train_idx], dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(y_train), batch_size=cfg.batch_sz, shuffle=True,
    )
    val_loader = None
    if has_val:
        y_val = torch.tensor(X_z[val_idx], dtype=torch.float32)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(y_val), batch_size=max(1, len(val_idx)), shuffle=False,
        )

    ssm = _build_ssm(cfg, n_neurons)
    seq_vae = LightningNonlinearSSM(ssm, cfg)

    monitor = "valid_loss" if has_val else "train_loss"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt_callback = ModelCheckpoint(
        save_top_k=1, monitor=monitor, mode="min", dirpath=ckpt_dir,
        filename="{epoch:0}_{" + monitor + ":.2f}",
    )
    callbacks = [ckpt_callback]
    if early_stopping_patience is not None:
        callbacks.append(EarlyStopping(monitor=monitor, mode="min", patience=early_stopping_patience))

    csv_logger = CSVLogger(log_dir, name=run_name)
    trainer = lightning.Trainer(
        max_epochs=cfg.n_epochs,
        gradient_clip_val=1.0,
        default_root_dir=str(Path(ckpt_dir) / "lightning"),
        callbacks=callbacks,
        logger=csv_logger,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
    )
    trainer.fit(model=seq_vae, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt_path = ckpt_callback.best_model_path or None
    if best_ckpt_path:
        seq_vae = LightningNonlinearSSM.load_from_checkpoint(best_ckpt_path, ssm=ssm, cfg=cfg)
        ssm = seq_vae.ssm
    elif verbose:
        print("No checkpoint was saved (training may have been interrupted); using the last-epoch weights.")

    latents, latents_std = _smooth_all_trials(
        ssm, X_z, n_samples=cfg.n_samples, device=device, chunk_size=infer_chunk_size,
    )

    unique_names = list(dict.fromkeys(labels))
    latents_by_cond = {name: latents[labels == name] for name in unique_names}

    return {
        "ssm": ssm,
        "cfg": cfg,
        "trainer": trainer,
        "scaler": scaler,
        "good_X": good_X,
        "labels": labels,
        "is_train": is_train,
        "latents": latents,
        "latents_std": latents_std,
        "latents_by_cond": latents_by_cond,
        "best_ckpt_path": best_ckpt_path,
    }


def save_xfads_result(result, path):
    """
    Pickle the numpy/config parts of an run_xfads() result (everything
    except the live torch model/trainer, which are already checkpointed by
    Lightning at result["best_ckpt_path"]).
    """
    payload = {
        "cfg": OmegaConf.to_container(result["cfg"], resolve=True),
        "scaler": result["scaler"],
        "good_X": result["good_X"],
        "labels": result["labels"],
        "is_train": result["is_train"],
        "latents": result["latents"],
        "latents_std": result["latents_std"],
        "latents_by_cond": result["latents_by_cond"],
        "best_ckpt_path": result["best_ckpt_path"],
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved XFADS result to {path}")


# ---------------------------------------------------------------------------
# Plotting -- mean per-condition latent trajectory, styled like
# CCA_analysis.plot_cca_state_space (start/touch/end markers, time-colored line)
# ---------------------------------------------------------------------------

def plot_xfads_latents(
    result,
    conditions=None,
    dims=(0, 1),
    time=None,
    smooth_sigma=None,
    cmap="viridis",
    figsize=(6, 6),
    title=None,
    ax=None,
):
    """
    Plot the mean trial trajectory through latent space (dims[0] vs dims[1]),
    one path per condition, colored along its length by time.

    result : output of run_xfads.
    conditions : sequence of condition objects (with `.name`) or name
                 strings; defaults to every condition in result.
    dims : which two latent dimensions to use as axes.
    time : (n_timepoints,) real time axis (e.g. organised.time_zF); sample
           index if None. If given, touch onset (t=0) is marked with a star.
    """
    latents_by_cond = result["latents_by_cond"]
    if conditions is None:
        names = list(latents_by_cond.keys())
    else:
        names = [c.name if hasattr(c, "name") else str(c) for c in conditions]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    d0, d1 = dims
    last_lc = None
    for name in names:
        if name not in latents_by_cond:
            raise KeyError(f"Condition {name!r} not found in result['latents_by_cond'].")
        traj = np.nanmean(latents_by_cond[name], axis=0)  # (n_time, n_latents)
        if smooth_sigma is not None:
            from scipy.ndimage import gaussian_filter1d
            traj = gaussian_filter1d(traj, sigma=smooth_sigma, axis=0, mode="nearest")
        x, y = traj[:, d0], traj[:, d1]
        t = np.asarray(time) if time is not None else np.arange(len(x))

        points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, linewidth=2.5)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        last_lc = lc

        ax.scatter(x[0], y[0], marker="o", s=60, color="green", zorder=3)  # start
        if time is not None:
            touch_idx = np.argmin(np.abs(t))
            ax.scatter(x[touch_idx], y[touch_idx], marker="*", s=120, color="orange", zorder=3)  # touch
        ax.scatter(x[-1], y[-1], marker="s", s=60, color="gray", zorder=3)  # end
        ax.annotate(name, (x[-1], y[-1]))

    ax.relim()
    ax.autoscale_view()
    ax.set_xlabel(f"Latent {d0 + 1}")
    ax.set_ylabel(f"Latent {d1 + 1}")
    ax.set_title(title or "XFADS latent trajectories")
    if last_lc is not None:
        fig.colorbar(last_lc, ax=ax, label="Time (s)" if time is not None else "Timepoint")
    fig.tight_layout()

    return fig, ax
