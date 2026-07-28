# CCA_analysis.py
"""
Flexible shared-CCA pipeline between npix (X) and cerebellar zF (Y) trial
data, pooled across any number of conditions (e.g. Left vs Right, or Left
Correct / Left Incorrect / Right Correct / Right Incorrect, ...).

Main entry point: run_shared_cca(conditions, ...)

    conditions : dict {name: ConditionData}, or a list of ConditionData
                 (each object's `.name` is used as the key). ConditionData
                 objects come from organise_touch_trials.py, e.g.
                 organised['iCueL_1port_cor'] or a .pool(...) result.
                 Trial-level arrays are (n_trials, n_timepoints, n_features).

A single shared CCA is fit on ALL conditions' trials pooled together (so
every condition is projected into the same component space), but every
correlation this module reports -- both the final fitted model's and the
cross-validated diagnostic's -- is computed SEPARATELY per condition, never
pooled across conditions. Pooling would let between-condition mean
differences inflate/deflate the correlation independent of how well X and Y
actually covary within a condition.

Every preprocessing step is an optional switch:
    scale=False              skip z-scoring X/Y before CCA
    n_pca_components=None    skip PCA, fit CCA directly on (scaled) features
    test_size=None            fit on all trials, no held-out split
    cv=False                   skip the k-fold cross-validated diagnostic

Microzone selection: pass `zone` (a microzone id, or list of ids) plus
`neuron_zone` (organised.neuron_zone) to restrict Y to the cerebellar
neurons in that zone instead of the whole population. X (npix) is never
affected by `zone` -- only which zF columns feed the CCA.

run_shared_cca returns a dict with the fitted CCA/scaler/PCA objects, the
feature masks used, and every trial's projection onto the shared components
-- both in original concatenated order and split back out by condition name
-- so trajectories/correlations can be plotted downstream without refitting
anything.
"""
import numpy as np

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles


# ---------------------------------------------------------------------------
# Small composable helpers
# ---------------------------------------------------------------------------

def flatten_trials(X):
    """(n_trials, n_timepoints, n_features) -> (n_trials * n_timepoints, n_features)."""
    return X.reshape(-1, X.shape[-1])


def finite_feature_mask(X):
    """Boolean mask over columns that are finite in every row of X."""
    return np.isfinite(X).all(axis=0)


def fit_scaler(X, Y, scale=True):
    if not scale:
        return None, None, X, Y
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    return scaler_x, scaler_y, scaler_x.fit_transform(X), scaler_y.fit_transform(Y)


def apply_scaler(X, Y, scaler_x, scaler_y):
    X_z = scaler_x.transform(X) if scaler_x is not None else X
    Y_z = scaler_y.transform(Y) if scaler_y is not None else Y
    return X_z, Y_z


def fit_pca(X, Y, n_components=10):
    if n_components is None:
        return None, None, X, Y
    n_x = min(n_components, X.shape[0] - 1, X.shape[1])
    n_y = min(n_components, Y.shape[0] - 1, Y.shape[1])
    pca_x, pca_y = PCA(n_components=n_x), PCA(n_components=n_y)
    return pca_x, pca_y, pca_x.fit_transform(X), pca_y.fit_transform(Y)


def apply_pca(X, Y, pca_x, pca_y):
    X_p = pca_x.transform(X) if pca_x is not None else X
    Y_p = pca_y.transform(Y) if pca_y is not None else Y
    return X_p, Y_p


def canonical_corrs(X_scores, Y_scores):
    """Per-component Pearson correlation between X and Y canonical variates."""
    n_components = X_scores.shape[1]
    return np.array([
        np.corrcoef(X_scores[:, k], Y_scores[:, k])[0, 1]
        for k in range(n_components)
    ])


def select_zone_neurons(neuron_zone, zone):
    """Indices into the neuron axis whose neuron_zone label is in `zone` (scalar or list)."""
    neuron_zone = np.asarray(neuron_zone)
    zone = np.atleast_1d(zone)
    idx = np.where(np.isin(neuron_zone, zone))[0]
    if idx.size == 0:
        raise ValueError(f"No neurons found for zone(s) {zone.tolist()}.")
    return idx


def as_named_conditions(conditions):
    """
    Normalize `conditions` (a dict {name: ConditionData}, or a list of
    ConditionData objects) into an ordered dict {name: ConditionData}. List
    entries are keyed by their own `.name` attribute (falling back to
    "cond{i}" if unnamed).
    """
    if isinstance(conditions, dict):
        return dict(conditions)
    out = {}
    for i, cond in enumerate(conditions):
        name = getattr(cond, "name", None) or f"cond{i}"
        out[name] = cond
    return out


# ---------------------------------------------------------------------------
# Collecting trial-level data from any number of conditions
# ---------------------------------------------------------------------------

def collect_trials(
    conditions,        # dict {name: ConditionData}, or list of ConditionData
    zone=None,         # microzone id, or list of ids -- restricts Y (zF) only
    neuron_zone=None,  # per-neuron microzone label, required if zone is given
    x_attr="npix_ds",
    y_attr="zF",
    verbose=True,
):
    """
    Pull trial-level X/Y arrays off any number of ConditionData objects and
    concatenate them into one dataset, tagging each trial with the name of
    the condition it came from.

    Returns X_trials, Y_trials, labels, neuron_idx
        X_trials, Y_trials : (n_trials_total, n_timepoints, n_features)
        labels             : (n_trials_total,) object array of condition names
        neuron_idx         : neuron-axis indices used for Y, or None if `zone`
                              was not given (i.e. the whole population was kept)
    """
    conditions = as_named_conditions(conditions)
    if len(conditions) < 2:
        raise ValueError("collect_trials needs at least 2 conditions.")

    if zone is not None:
        if neuron_zone is None:
            raise ValueError("neuron_zone must be provided when zone is specified.")
        neuron_idx = select_zone_neurons(neuron_zone, zone)
    else:
        neuron_idx = None

    X_list, Y_list, label_list = [], [], []
    for name, cond in conditions.items():
        X_c = np.asarray(getattr(cond, x_attr), dtype=float)
        if zone is not None:
            Y_c = np.asarray(cond.zF, dtype=float)[:, :, neuron_idx]
        else:
            Y_c = np.asarray(getattr(cond, y_attr), dtype=float)
        X_list.append(X_c)
        Y_list.append(Y_c)
        label_list.append(np.full(X_c.shape[0], name, dtype=object))

    X_trials = np.concatenate(X_list, axis=0)
    Y_trials = np.concatenate(Y_list, axis=0)
    labels = np.concatenate(label_list)

    if verbose:
        zone_str = f"zone {zone}" if zone is not None else "all neurons"
        print(f"X trials: {X_trials.shape}  Y trials: {Y_trials.shape}  ({zone_str})")
        counts = ", ".join(f"{name}: {np.sum(labels == name)} trials" for name in conditions)
        print(counts)

    return X_trials, Y_trials, labels, neuron_idx


def _aggregate_corrs_by_fold(corrs_by_fold, names, n_components):
    """
    corrs_by_fold: list (one per fold) of {name: array_or_None}.
    Returns (raw, mean, std), each a dict {name: array}; raw is (n_folds, n_components)
    with NaN rows/entries where a condition had <2 rows in that fold.
    """
    raw, mean, std = {}, {}, {}
    for name in names:
        arr = np.full((len(corrs_by_fold), n_components), np.nan)
        for f, fold_corrs in enumerate(corrs_by_fold):
            c = fold_corrs.get(name)
            if c is not None:
                arr[f, :len(c)] = c
        raw[name] = arr
        mean[name] = np.nanmean(arr, axis=0)
        std[name] = np.nanstd(arr, axis=0)
    return raw, mean, std


# ---------------------------------------------------------------------------
# Cross-validated CCA diagnostic (complete trials held out per fold)
# ---------------------------------------------------------------------------

def trial_cv_cca(
    X_trials,
    Y_trials,
    condition_labels=None,
    n_cca_components=3,
    n_pca_components=10,
    scale=True,
    max_folds=5,
    verbose=True,
):
    """
    Cross-validated CCA with complete trials held out per fold.

    Expected input shape: (n_trials, n_timepoints, n_features) for both.
    Scaling and PCA are refit within each fold (train-only) and are optional
    (scale=False / n_pca_components=None to skip either).

    `condition_labels`, if given, is a per-trial array (len == n_trials)
    naming which condition each trial belongs to. Canonical correlations are
    computed SEPARATELY per condition (never pooled across conditions) --
    pooling would let between-condition mean differences inflate/deflate the
    correlation independent of how well X and Y covary within a condition.
    If omitted, all trials are treated as one group ("all").
    """
    X_trials = np.asarray(X_trials, dtype=float)
    Y_trials = np.asarray(Y_trials, dtype=float)

    if X_trials.ndim != 3 or Y_trials.ndim != 3:
        raise ValueError("Inputs must have shape (n_trials, n_timepoints, n_features).")
    if X_trials.shape[0] != Y_trials.shape[0]:
        raise ValueError("X and Y have different numbers of trials.")
    if X_trials.shape[1] != Y_trials.shape[1]:
        raise ValueError("X and Y have different numbers of timepoints.")

    n_trials, n_timepoints, _ = X_trials.shape
    if n_trials < 2:
        raise ValueError("At least two trials are required.")

    if condition_labels is None:
        condition_labels = np.full(n_trials, "all", dtype=object)
    else:
        condition_labels = np.asarray(condition_labels, dtype=object)
        if len(condition_labels) != n_trials:
            raise ValueError("condition_labels must have one entry per trial.")
    names = list(np.unique(condition_labels))

    X = flatten_trials(X_trials)
    Y = flatten_trials(Y_trials)
    groups = np.repeat(np.arange(n_trials), n_timepoints)
    row_labels = condition_labels[groups]

    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y, groups, row_labels = X[valid], Y[valid], groups[valid], row_labels[valid]

    if max_folds is not None and max_folds < 2:
        raise ValueError("max_folds must be at least 2.")
    n_splits = 2 if n_trials < 4 else min(max_folds, n_trials)

    cv = GroupKFold(n_splits=n_splits)

    train_corrs_by_fold = []
    test_corrs_by_fold = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, Y, groups=groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        scaler_x, scaler_y, X_train_z, Y_train_z = fit_scaler(X_train, Y_train, scale=scale)
        X_test_z, Y_test_z = apply_scaler(X_test, Y_test, scaler_x, scaler_y)

        pca_x, pca_y, X_train_final, Y_train_final = fit_pca(
            X_train_z, Y_train_z, n_components=n_pca_components,
        )
        X_test_final, Y_test_final = apply_pca(X_test_z, Y_test_z, pca_x, pca_y)

        n_components = min(n_cca_components, X_train_final.shape[1], Y_train_final.shape[1])
        cca = CCA(n_components=n_components, max_iter=2000)
        cca.fit(X_train_final, Y_train_final)

        X_train_score, Y_train_score = cca.transform(X_train_final, Y_train_final)
        X_test_score, Y_test_score = cca.transform(X_test_final, Y_test_final)

        train_labels_fold = row_labels[train_idx]
        test_labels_fold = row_labels[test_idx]

        fold_train_corrs, fold_test_corrs = {}, {}
        for name in names:
            tr_mask = train_labels_fold == name
            te_mask = test_labels_fold == name
            fold_train_corrs[name] = (
                canonical_corrs(X_train_score[tr_mask], Y_train_score[tr_mask])
                if tr_mask.sum() >= 2 else None
            )
            fold_test_corrs[name] = (
                canonical_corrs(X_test_score[te_mask], Y_test_score[te_mask])
                if te_mask.sum() >= 2 else None
            )

        train_corrs_by_fold.append(fold_train_corrs)
        test_corrs_by_fold.append(fold_test_corrs)

        if verbose:
            held_out_trials = np.unique(groups[test_idx])
            printable = {name: (np.round(c, 3) if c is not None else None)
                         for name, c in fold_test_corrs.items()}
            print(f"Fold {fold}: held-out trials {held_out_trials}; test correlations {printable}")

    _, mean_train_corrs, _ = _aggregate_corrs_by_fold(train_corrs_by_fold, names, n_cca_components)
    test_corrs_raw, mean_test_corrs, std_test_corrs = _aggregate_corrs_by_fold(
        test_corrs_by_fold, names, n_cca_components,
    )

    return {
        "train_corrs": train_corrs_by_fold,       # list of {name: array_or_None}, one per fold
        "test_corrs": test_corrs_raw,             # {name: (n_folds, n_components)}, NaN-padded
        "mean_train_corrs": mean_train_corrs,     # {name: (n_components,)}
        "mean_test_corrs": mean_test_corrs,       # {name: (n_components,)}
        "std_test_corrs": std_test_corrs,         # {name: (n_components,)}
        "n_trials": n_trials,
        "n_folds": n_splits,
    }


# ---------------------------------------------------------------------------
# Core fit-and-project block, reused for both "shared" and "separate" modes
# ---------------------------------------------------------------------------

def _fit_and_project(
    X_trials, Y_trials, labels, n_timepoints,
    scale, n_pca_components, n_cca_components,
    test_size, stratify, cv, max_folds, random_state, verbose,
):
    """
    Fit one CCA (scaler -> PCA -> CCA) on the given trials, then project all
    of them back through it. `labels` names each trial (may be a single
    repeated name, or several -- correlations are always broken out per
    unique name present). Returns the per-run result dict documented on
    run_shared_cca (everything except `neuron_idx`/`config`, added by the caller).
    """
    n_trials_total = X_trials.shape[0]
    names = list(dict.fromkeys(labels))  # unique, order of first appearance

    can_stratify = stratify and len(set(labels)) > 1
    if test_size is not None:
        train_idx, test_idx = train_test_split(
            np.arange(n_trials_total),
            test_size=test_size,
            random_state=random_state,
            stratify=labels if can_stratify else None,
        )
    else:
        train_idx = np.arange(n_trials_total)
        test_idx = np.array([], dtype=int)

    # fit scaler/PCA/CCA on the training trials only
    X_train_flat = flatten_trials(X_trials[train_idx])
    Y_train_flat = flatten_trials(Y_trials[train_idx])

    good_X = finite_feature_mask(X_train_flat)
    good_Y = finite_feature_mask(Y_train_flat)
    X_train_flat = X_train_flat[:, good_X]
    Y_train_flat = Y_train_flat[:, good_Y]

    scaler_x, scaler_y, X_train_z, Y_train_z = fit_scaler(X_train_flat, Y_train_flat, scale=scale)
    pca_x, pca_y, X_train_final, Y_train_final = fit_pca(
        X_train_z, Y_train_z, n_components=n_pca_components,
    )

    n_components = min(n_cca_components, X_train_final.shape[1], Y_train_final.shape[1])
    cca = CCA(n_components=n_components, max_iter=2000)
    cca.fit(X_train_final, Y_train_final)

    cv_results = None
    if cv:
        if verbose:
            print("Cross-validated diagnostic (training trials only):")
        cv_results = trial_cv_cca(
            X_trials[train_idx][:, :, good_X],
            Y_trials[train_idx][:, :, good_Y],
            condition_labels=labels[train_idx],
            n_cca_components=n_cca_components,
            n_pca_components=n_pca_components,
            scale=scale,
            max_folds=max_folds,
            verbose=verbose,
        )

    # project every trial (train + held-out) back through the fitted pipeline
    X_all_flat = flatten_trials(X_trials)[:, good_X]
    Y_all_flat = flatten_trials(Y_trials)[:, good_Y]
    X_all_z, Y_all_z = apply_scaler(X_all_flat, Y_all_flat, scaler_x, scaler_y)
    X_all_final, Y_all_final = apply_pca(X_all_z, Y_all_z, pca_x, pca_y)
    X_scores_flat, Y_scores_flat = cca.transform(X_all_final, Y_all_final)

    X_scores = X_scores_flat.reshape(n_trials_total, n_timepoints, n_components)
    Y_scores = Y_scores_flat.reshape(n_trials_total, n_timepoints, n_components)

    is_test = np.zeros(n_trials_total, dtype=bool)
    is_test[test_idx] = True

    # canonical correlations of the actual final model (not the per-fold refits
    # inside cv_results), computed separately per condition -- never pooled.
    trial_of_row = np.repeat(np.arange(n_trials_total), n_timepoints)
    row_is_train = np.isin(trial_of_row, train_idx)
    row_labels = labels[trial_of_row]

    final_corrs = {}
    for name in names:
        cond_rows = row_labels == name
        train_rows = cond_rows & row_is_train
        test_rows = cond_rows & ~row_is_train
        final_corrs[name] = {
            "train": (
                canonical_corrs(X_scores_flat[train_rows], Y_scores_flat[train_rows])
                if train_rows.sum() >= 2 else None
            ),
            "test": (
                canonical_corrs(X_scores_flat[test_rows], Y_scores_flat[test_rows])
                if test_rows.sum() >= 2 else None
            ),
        }

    if verbose:
        print("Final model canonical correlations (per condition):")
        for name, corrs in final_corrs.items():
            train_str = np.round(corrs["train"], 3) if corrs["train"] is not None else None
            test_str = np.round(corrs["test"], 3) if corrs["test"] is not None else None
            print(f"  {name}: train={train_str}" + (f", test={test_str}" if test_idx.size > 0 else ""))

    def split_by_cond(scores):
        return {name: scores[labels == name] for name in names}

    return {
        "cca": cca,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "pca_x": pca_x,
        "pca_y": pca_y,
        "good_X": good_X,
        "good_Y": good_Y,
        "n_components": n_components,
        "labels": labels,
        "is_test": is_test,
        "X_scores": X_scores,
        "Y_scores": Y_scores,
        "X_scores_by_cond": split_by_cond(X_scores),
        "Y_scores_by_cond": split_by_cond(Y_scores),
        "final_corrs": final_corrs,
        "cv_results": cv_results,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_shared_cca(
    conditions,             # dict {name: ConditionData}, or list of ConditionData
    fit="shared",           # "shared": one CCA pooled across all conditions (comparable
                             #   component space -- use this to see how e.g. Left/Right
                             #   trajectories separate within CC space).
                             # "separate": an independent CCA fit per condition (own
                             #   scaler/PCA/CCA, own components -- use this to ask how
                             #   correlated X and Y are within a single condition, since
                             #   pooling other conditions into the fit would change that
                             #   answer).
    zone=None,              # microzone id, or list of ids -- None = whole cerebellum
    neuron_zone=None,       # organised.neuron_zone, required if zone is given
    x_attr="npix_ds",
    y_attr="zF",
    scale=True,
    n_pca_components=10,
    n_cca_components=3,
    test_size=None,         # e.g. 0.2 to hold out trials from the final fit; None = fit on all
    stratify=True,          # stratify the held-out split by condition label (shared mode only)
    temporal_window=None,      # (start, end) in seconds; if given, only use that time window for CCA
    time=None,              # (n_timepoints,) real time axis, e.g. organised.time_zF; required for temporal_window
                             #   to mean "seconds relative to touch" -- without it, falls back to sample index
    cv=True,                # run the k-fold cross-validated diagnostic
    max_folds=5,
    random_state=42,
    verbose=True,
):
    """
    Fit CCA between X (npix) and Y (zF) trial data across any number of
    conditions (2, 3, 4, ...), then project every trial back through the
    fitted scaler/PCA/CCA pipeline -- split back out by original condition
    name -- so downstream code can plot per-condition trajectories/
    correlations without refitting anything.

    `fit="shared"` (default) pools all conditions into one CCA fit, so every
    condition lands in the SAME component space -- what you want to compare
    trajectories (e.g. does the Left trajectory separate from Right in CC
    space?). `fit="separate"` instead fits an independent CCA per condition
    (its own scaler/PCA/CCA/components) -- what you want if you're asking
    how correlated X and Y are WITHIN one condition, since mixing in other
    conditions' trials would change that correlation and the two component
    spaces are then not comparable across conditions.

    Regardless of `fit`, every correlation reported here -- `final_corrs`
    and `cv_results` -- is computed per condition, never pooled across
    conditions (pooling would let between-condition mean differences
    inflate/deflate the correlation independent of the true within-condition
    relationship).

    Set `zone` (+ `neuron_zone`) to fit Y from a single microzone's neurons
    instead of the whole cerebellar population; X is unaffected.

    Set `temporal_window=(start, end)` (seconds) to restrict the CCA fit to
    a peri-touch window, e.g. (-0.3, 0.3) for +/-300ms around touch. Pass
    `time` (e.g. organised.time_zF) so `start`/`end` are interpreted against
    the real, touch-relative time axis -- ConditionData objects don't carry
    their own time axis, so without `time` this silently falls back to
    sample index and the window will almost certainly not mean what you
    expect.

    Returns a dict. If fit="shared":
        cca, scaler_x, scaler_y, pca_x, pca_y   fitted objects (None if skipped)
        good_X, good_Y                          finite-feature masks applied before fitting
        neuron_idx                              neuron-axis indices used for Y (None if zone is None)
        n_components                            actual n CCA components used
        labels                                  condition name per trial, concatenated order
        is_test                                 bool mask, True for held-out trials (all False if test_size is None)
        X_scores, Y_scores                      (n_trials_total, n_timepoints, n_components), concatenated order
        X_scores_by_cond, Y_scores_by_cond       same, split into {name: array} per condition
        final_corrs                             {name: {"train": arr, "test": arr_or_None}}
        cv_results                              {"mean_test_corrs": {name: arr}, ...} or None if cv=False
        config                                   the settings this run used

    If fit="separate":
        neuron_idx, config                      as above
        per_condition                           {name: {cca, scaler_x, scaler_y, pca_x, pca_y,
                                                  good_X, good_Y, n_components, is_test, X_scores,
                                                  Y_scores, final_corrs (a plain {"train":.., "test":..}
                                                  for this condition), cv_results}} -- one independent
                                                  fit per condition
    """
    if fit not in ("shared", "separate"):
        raise ValueError("fit must be 'shared' or 'separate'.")

    conditions = as_named_conditions(conditions)
    if len(conditions) < 2:
        raise ValueError("run_shared_cca needs at least 2 conditions.")

    X_trials, Y_trials, labels, neuron_idx = collect_trials(
        conditions, zone=zone, neuron_zone=neuron_zone,
        x_attr=x_attr, y_attr=y_attr, verbose=verbose,
    )
    n_timepoints = X_trials.shape[1]
    if temporal_window is not None:
        start, end = temporal_window
        if time is not None:
            time_axis = np.asarray(time)
            if time_axis.shape[0] != n_timepoints:
                raise ValueError(
                    f"time has {time_axis.shape[0]} samples but trials have "
                    f"{n_timepoints} timepoints."
                )
        else:
            time_axis = np.arange(n_timepoints)  # sample index -- pass `time` for a real peri-touch window
        mask = (time_axis >= start) & (time_axis <= end)
        if not np.any(mask):
            raise ValueError(f"No timepoints found in the requested window {temporal_window}.")
        X_trials = X_trials[:, mask, :]
        Y_trials = Y_trials[:, mask, :]
        n_timepoints = X_trials.shape[1]
        if verbose:
            print(f"Restricting to temporal window {temporal_window}: {n_timepoints} timepoints remain.")

    config = dict(
        condition_names=list(conditions.keys()), fit=fit, zone=zone,
        scale=scale, n_pca_components=n_pca_components,
        n_cca_components=n_cca_components, test_size=test_size,
        max_folds=max_folds, random_state=random_state,
    )

    if fit == "shared":
        block = _fit_and_project(
            X_trials, Y_trials, labels, n_timepoints,
            scale=scale, n_pca_components=n_pca_components, n_cca_components=n_cca_components,
            test_size=test_size, stratify=stratify, cv=cv, max_folds=max_folds,
            random_state=random_state, verbose=verbose,
        )
        block["neuron_idx"] = neuron_idx
        block["config"] = config
        block["n_timepoints"] = n_timepoints
        return block

    # fit == "separate": one independent CCA per condition, no shared component space
    per_condition = {}
    for name in conditions:
        if verbose:
            print(f"\n=== Fitting separately: {name} ===")
        mask = labels == name
        block = _fit_and_project(
            X_trials[mask], Y_trials[mask], labels[mask], n_timepoints,
            scale=scale, n_pca_components=n_pca_components, n_cca_components=n_cca_components,
            test_size=test_size, stratify=False, cv=cv, max_folds=max_folds,
            random_state=random_state, verbose=verbose,
        )
        block["final_corrs"] = block["final_corrs"][name]
        block["X_scores_by_cond"] = block["X_scores_by_cond"][name]
        block["Y_scores_by_cond"] = block["Y_scores_by_cond"][name]
        del block["labels"]
        per_condition[name] = block

    return {
        "neuron_idx": neuron_idx,
        "per_condition": per_condition,
        "config": config,
        "n_timepoints": n_timepoints,
    }


def run_shared_cca_per_zone(
    conditions,         # dict {name: ConditionData}, or list of ConditionData
    neuron_zone,
    zone_ids=None,      # defaults to all unique labels in neuron_zone
    **kwargs,
):
    """
    Convenience wrapper: call run_shared_cca once per microzone (X stays the
    full npix population each time; Y is restricted to that zone's neurons),
    returning {zone_id: run_shared_cca(...) result}.

    Any extra kwargs (scale, n_pca_components, n_cca_components, test_size,
    cv, max_folds, random_state, verbose, ...) are forwarded to every call.
    """
    if zone_ids is None:
        zone_ids = np.unique(np.asarray(neuron_zone))

    return {
        zone_id: run_shared_cca(
            conditions, zone=zone_id, neuron_zone=neuron_zone, **kwargs,
        )
        for zone_id in zone_ids
    }


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d


def _scores_by_condition(cca_res, which="X", fit=None):
    """
    Normalize shared/separate cca_res into {name: (n_trials, n_time, n_components)}
    for the requested variate (`which` = "X" i.e. npix, or "Y" i.e. zF).
    """
    which = which.upper()
    if which not in ("X", "Y"):
        raise ValueError("which must be 'X' or 'Y'.")

    if fit is None:
        fit = "separate" if "per_condition" in cca_res else "shared"

    if fit == "shared":
        return cca_res[f"{which}_scores_by_cond"]

    score_key = f"{which}_scores"
    return {name: block[score_key] for name, block in cca_res["per_condition"].items()}


def plot_cca_component_timecourse(
    cca_res,
    which="X",
    fit=None,
    conditions=None,
    component=0,
    time=None,
    show_sem=True,
    figsize=(7, 5),
    title=None,
    ax=None,
):
    """
    Plot how strongly each condition's trials load onto one canonical
    component over the trial timecourse: mean canonical-variate score per
    timepoint (across trials), one line per condition, +/- SEM shading.

    Parameters
    ----------
    cca_res : dict
        Output of run_shared_cca (fit="shared" or "separate").
    which : {"X", "Y"}, default "X"
        Plot the npix (X) or zF (Y) canonical variate.
    fit : {"shared", "separate"}, optional
        Inferred from cca_res if None.
    conditions : sequence, optional
        Conditions to plot (objects with `.name`, or name strings). Defaults
        to every condition in cca_res.
    component : int, default=0
        Zero-based canonical-component index (0 = CC1, 1 = CC2, ...).
    time : array-like, optional
        (n_timepoints,) real time axis (e.g. condition.time_zF, in seconds).
        Defaults to sample index if not given.
    show_sem : bool, default=True
        Shade mean +/- SEM across trials.

    Returns
    -------
    fig, ax, plot_data
    """
    scores_by_cond = _scores_by_condition(cca_res, which=which, fit=fit)

    if conditions is None:
        names = list(scores_by_cond.keys())
    else:
        names = [c.name if hasattr(c, "name") else str(c) for c in conditions]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    traces = {}
    for name in names:
        if name not in scores_by_cond:
            raise KeyError(f"Condition {name!r} not found in cca_res.")
        scores = scores_by_cond[name]  # (n_trials, n_time, n_components)
        if component < 0 or component >= scores.shape[-1]:
            raise IndexError(
                f"component={component} is invalid for {name!r}: "
                f"only {scores.shape[-1]} components available."
            )
        trace = scores[:, :, component]  # (n_trials, n_time)
        mean = np.nanmean(trace, axis=0)
        t = np.asarray(time) if time is not None else np.arange(mean.shape[0])

        line, = ax.plot(t, mean, label=name, linewidth=2)
        if show_sem:
            n_valid = np.sum(np.isfinite(trace), axis=0)
            sem = np.nanstd(trace, axis=0) / np.sqrt(np.maximum(n_valid, 1))
            ax.fill_between(t, mean - sem, mean + sem, alpha=0.2, color=line.get_color())

        traces[name] = mean

    ax.axhline(0, linewidth=1, linestyle="--", alpha=0.4, color="gray")
    ax.set_xlabel("Time (s)" if time is not None else "Timepoint")
    ax.set_ylabel(f"{which.upper()} canonical variate, CC{component + 1}")
    ax.set_title(title or f"CC{component + 1} loading over time ({which.upper()})")
    ax.legend(frameon=False)
    fig.tight_layout()

    plot_data = {"which": which, "component": component, "traces": traces}
    return fig, ax, plot_data


def _resolve_per_condition(value, names, default_cycle):
    """Resolve a per-condition style spec into a {name: value} dict.

    `value` may be a single item (applied to every condition), a list/tuple
    (cycled by position), or a dict {name: value} (looked up by name, with
    any missing names falling back to `default_cycle`).
    """
    if isinstance(value, dict):
        return {
            name: value[name] if name in value else default_cycle[i % len(default_cycle)]
            for i, name in enumerate(names)
        }
    if value is None:
        seq = default_cycle
    elif isinstance(value, (list, tuple)):
        seq = value
    else:
        seq = [value]
    return {name: seq[i % len(seq)] for i, name in enumerate(names)}


def _plot_cca_state_space_one(
    cca_res,
    which,
    fit,
    conditions,
    components,
    time,
    cond_colours,
    cond_linestyles,
    cmap,
    smooth_sigma,
    title,
    ax,
):
    """Single-axis worker behind plot_cca_state_space; see that docstring."""
    scores_by_cond = _scores_by_condition(cca_res, which=which, fit=fit)

    if conditions is None:
        names = list(scores_by_cond.keys())
    else:
        names = [c.name if hasattr(c, "name") else str(c) for c in conditions]

    cond_colours = _resolve_per_condition(cond_colours, names, [cmap])
    cond_colours = {name: plt.get_cmap(c) if isinstance(c, str) else c for name, c in cond_colours.items()}
    cond_linestyles = _resolve_per_condition(cond_linestyles, names, ["-", "--", "-.", ":"])

    c0, c1 = components
    fig = ax.figure

    trajectories = {}
    last_lc = None
    for name in names:
        if name not in scores_by_cond:
            raise KeyError(f"Condition {name!r} not found in cca_res.")
        scores = scores_by_cond[name]  # (n_trials, n_time, n_components)
        n_components = scores.shape[-1]
        if max(c0, c1) >= n_components:
            raise IndexError(
                f"components={components} invalid for {name!r}: "
                f"only {n_components} components available."
            )
        mean_traj = np.nanmean(scores, axis=0)  # (n_time, n_components)
        if smooth_sigma is not None:
            mean_traj = gaussian_filter1d(mean_traj, sigma=smooth_sigma, axis=0, mode="nearest")
        x, y = mean_traj[:, c0], mean_traj[:, c1]
        t = np.asarray(time) if time is not None else np.arange(len(x))

        points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments, cmap=cond_colours[name], linestyle=cond_linestyles[name], linewidth=2.5,
        )
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        last_lc = lc

        ax.scatter(x[0], y[0], marker="o", s=60, color='green', zorder=3) # start
        # add scatterpoint for touch time if time is provided
        if time is not None:
            touch_idx = np.argmin(np.abs(t))
            ax.scatter(x[touch_idx], y[touch_idx], marker="*", s=120, color="orange", zorder=3) # touch
        ax.scatter(x[-1], y[-1], marker="s", s=60, color="gray", zorder=3) # end

        trajectories[name] = mean_traj[:, [c0, c1]]

    ax.relim()
    ax.autoscale_view()
    ax.set_xlabel(f"{which.upper()} CC{c0 + 1}")
    ax.set_ylabel(f"{which.upper()} CC{c1 + 1}")
    ax.set_title(title or f"{which.upper()} trajectory through CC space")

    legend_handles = [
        Line2D([0], [0], color=cond_colours[name](0.7), linestyle=cond_linestyles[name], label=name)
        for name in names
    ]
    ax.legend(handles=legend_handles, frameon=False)
    if last_lc is not None:
        fig.colorbar(last_lc, ax=ax, label="Time (s)" if time is not None else "Timepoint")

    plot_data = {"which": which, "components": components, "trajectories": trajectories}
    return ax, plot_data


def plot_cca_state_space(
    cca_res,
    which="both",
    fit=None,
    conditions=None,
    components=(0, 1),
    time=None,
    cond_colours=None,  # colourmap(s) per condition: single cmap, [cmap1, cmap2, ...], or {cond_name: cmap}
    cond_linestyles=None,  # linestyle(s) per condition: single style, [ls1, ls2, ...], or {cond_name: ls}
    cmap=None,
    smooth_sigma=None,  # if given, gaussian-smooth (sigma in samples) each trajectory's timecourse before plotting
    figsize=None,
    title=None,
    ax=None,
):
    """
    Plot the mean trial trajectory through CC-space (components[0] vs
    components[1]), one path per condition, colored along its length by
    time. Requires fit="shared" (or comparable component spaces) for the
    conditions to be meaningfully overlaid on the same axes.

    Start of each trajectory is marked with a circle, end with a square, and
    (if `time` is given) touch onset with a star; condition identity is
    given by linestyle (color encodes time instead).

    Parameters
    ----------
    cca_res : dict
        Output of run_shared_cca.
    which : {"X", "Y", "both"}, default "both"
        "both" plots X and Y side by side as two subplots in one figure;
        pass "X" or "Y" for a single axis (usable with a pre-existing `ax`).
    fit : {"shared", "separate"}, optional
    conditions : sequence, optional
    components : (int, int), default (0, 1)
        Which two canonical components to use as the two axes.
    time : array-like, optional
        (n_timepoints,) real time axis for the colorbar; sample index if None.
    cond_colours : colormap, list, or dict, optional
        Per-condition colormap for the time-colored trajectory line. A
        single cmap (name or Colormap) applies to all conditions; a list is
        cycled by condition order; a dict {cond_name: cmap} looks up by
        name (falling back to `cmap` for any condition not listed).
    cond_linestyles : str, list, or dict, optional
        Per-condition linestyle, same shape rules as `cond_colours`.
        Defaults to cycling through ["-", "--", "-.", ":"].
    cmap : str, optional
        Fallback colormap used where `cond_colours` doesn't specify one;
        None uses matplotlib's default cmap.
    smooth_sigma : float, optional
        Gaussian smoothing sigma (in samples/timepoints) applied to each
        condition's mean trajectory before plotting. None = no smoothing.

    Returns
    -------
    which="both": fig, (ax_x, ax_y), {"X": plot_data, "Y": plot_data}
    which="X" or "Y": fig, ax, plot_data
    """
    if which == "both":
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize or (11, 5))
        else:
            axes = ax
            fig = axes[0].figure

        plot_data = {}
        for w, a in zip(("X", "Y"), axes):
            _, plot_data[w] = _plot_cca_state_space_one(
                cca_res, which=w, fit=fit, conditions=conditions, components=components,
                time=time, cond_colours=cond_colours, cond_linestyles=cond_linestyles,
                cmap=cmap, smooth_sigma=smooth_sigma, title=None, ax=a,
            )
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        return fig, axes, plot_data

    if which not in ("X", "Y"):
        raise ValueError("which must be 'X', 'Y', or 'both'.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6, 6))
    else:
        fig = ax.figure

    _, plot_data = _plot_cca_state_space_one(
        cca_res, which=which, fit=fit, conditions=conditions, components=components,
        time=time, cond_colours=cond_colours, cond_linestyles=cond_linestyles,
        cmap=cmap, smooth_sigma=smooth_sigma, title=title, ax=ax,
    )
    fig.tight_layout()
    return fig, ax, plot_data

def plot_cca_scatter(
    cca_res,
    which="X",
    fit=None,
    conditions=None,
    components=(0, 1),
    agg="trial_mean",     # "trial_mean": one point per trial, averaged over time; "all_timepoints": one point per (trial, timepoint)
    cond_colours=None,    # per-condition colour: single colour, [c1, c2, ...], or {cond_name: colour}
    alpha=0.7,
    s=40,
    show_centroids=False,     # also mark each condition's centroid (mean of its plotted points)
    centroid_marker="X",
    centroid_size=200,
    centroid_edgecolor="black",
    centroid_linewidth=1.5,
    figsize=(6, 6),
    title=None,
    ax=None,
):
    """
    Scatter plot of trials in CC-space (components[0] vs components[1], or
    components[0] vs [1] vs [2] for a 3D scatter), coloured by condition.
    Requires fit="shared" (or comparable component spaces) for the
    conditions to be meaningfully overlaid on the same axes.

    Unlike plot_cca_state_space (one averaged trajectory per condition,
    coloured along its length by time), this shows the actual spread of
    individual trials -- how separated the conditions' point clouds are,
    not just their mean paths.

    Parameters
    ----------
    cca_res : dict
        Output of run_shared_cca.
    which : {"X", "Y"}, default "X"
    fit : {"shared", "separate"}, optional
    conditions : sequence, optional
    components : (int, int) or (int, int, int), default (0, 1)
        Which canonical components to use as plot axes. Pass two indices
        for a 2D scatter or three (e.g. (0, 1, 2)) for a 3D scatter. If
        `ax` is given for the 3D case, it must already have a 3D
        projection (e.g. created with subplot_kw={"projection": "3d"}).
    agg : {"trial_mean", "all_timepoints"}, default "trial_mean"
        "trial_mean": one point per trial, averaged across its time window
        -- good for overall condition separation.
        "all_timepoints": one point per (trial, timepoint) -- denser, shows
        how much the per-timepoint cloud overlaps across conditions.
    cond_colours : color, list, or dict, optional
        Per-condition colour, same shape rules as elsewhere in this module
        (single value, list cycled by order, or {cond_name: colour}).
    show_centroids : bool, default False
        If True, also plot each condition's centroid (mean of its plotted
        points, computed after `agg`) as a larger marker on top of the
        point cloud, in the same colour with a contrasting edge.
    centroid_marker, centroid_size, centroid_edgecolor, centroid_linewidth :
        Styling for the centroid markers (only used if show_centroids=True).

    Returns
    -------
    fig, ax, plot_data
        plot_data["points"] holds the per-condition plotted points, and
        (if show_centroids=True) plot_data["centroids"] holds the
        per-condition centroid coordinates.
    """
    if agg not in ("trial_mean", "all_timepoints"):
        raise ValueError("agg must be 'trial_mean' or 'all_timepoints'.")
    if len(components) not in (2, 3):
        raise ValueError("components must have length 2 (2D) or 3 (3D).")
    is_3d = len(components) == 3

    scores_by_cond = _scores_by_condition(cca_res, which=which, fit=fit)

    if conditions is None:
        names = list(scores_by_cond.keys())
    else:
        names = [c.name if hasattr(c, "name") else str(c) for c in conditions]

    default_colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cond_colours = _resolve_per_condition(cond_colours, names, default_colours)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"} if is_3d else None)
    else:
        fig = ax.figure
        if is_3d and not hasattr(ax, "zaxis"):
            raise ValueError("3D components requires `ax` to have a 3D projection.")

    points_by_cond = {}
    for name in names:
        if name not in scores_by_cond:
            raise KeyError(f"Condition {name!r} not found in cca_res.")
        scores = scores_by_cond[name]  # (n_trials, n_time, n_components)
        n_components = scores.shape[-1]
        if max(components) >= n_components:
            raise IndexError(
                f"components={components} invalid for {name!r}: "
                f"only {n_components} components available."
            )

        idx = list(components)
        if agg == "trial_mean":
            points = np.nanmean(scores[:, :, idx], axis=1)  # (n_trials, n_dims)
        else:
            points = scores[:, :, idx].reshape(-1, len(idx))  # (n_trials * n_time, n_dims)

        ax.scatter(
            *[points[:, d] for d in range(points.shape[1])], color=cond_colours[name],
            alpha=alpha, s=s, label=name, edgecolor="none",
        )
        points_by_cond[name] = points

    centroids_by_cond = None
    if show_centroids:
        centroids_by_cond = {name: np.nanmean(points_by_cond[name], axis=0) for name in names}
        for name in names:
            centroid = centroids_by_cond[name]
            ax.scatter(
                *[[centroid[d]] for d in range(len(centroid))], color=cond_colours[name],
                marker=centroid_marker, s=centroid_size, edgecolor=centroid_edgecolor,
                linewidth=centroid_linewidth, zorder=5,
            )

    axis_labels = [f"{which.upper()} CC{c + 1}" for c in components]
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    if is_3d:
        ax.set_zlabel(axis_labels[2])
    ax.set_title(title or " vs ".join(axis_labels))
    ax.legend(frameon=False)
    fig.tight_layout()

    plot_data = {"which": which, "components": components, "agg": agg, "points": points_by_cond}
    if centroids_by_cond is not None:
        plot_data["centroids"] = centroids_by_cond
    return fig, ax, plot_data


def plot_cca_cv_correlations(
    cca_res,
    conditions=None,
    fit=None,
    component=0,
    show_individual=True,
    show_errorbars=True,
    jitter=0.06,
    figsize=(7, 5),
    title=None,
    ax=None,
):
    """
    Plot cross-validated CCA test correlations for shared or separate fits.

    Parameters
    ----------
    cca_res : dict
        Output returned by run_shared_cca.

    conditions : sequence, optional
        Conditions to plot. Elements can be:
            - condition objects with a `.name` attribute, or
            - condition-name strings.

        If None, condition names are inferred from cca_res.

    fit : {"shared", "separate"}, optional
        Type of CCA fit. If None, inferred from the structure of cca_res.

    component : int, default=0
        Zero-based canonical-component index:
            0 = CC1
            1 = CC2
            2 = CC3

    show_individual : bool, default=True
        Overlay each CV fold as a separate point.

    show_errorbars : bool, default=True
        Plot mean ± standard deviation across CV folds.

    jitter : float, default=0.06
        Horizontal jitter applied to individual fold points.

    figsize : tuple, default=(7, 5)
        Figure size, used only if ax is not supplied.

    title : str, optional
        Plot title. A default title is generated when None.

    ax : matplotlib.axes.Axes, optional
        Existing axes on which to draw.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    plot_data : dict
        Extracted means, standard deviations, and fold-level correlations.
    """

    # ---------------------------------------------------------
    # 1. Infer fit type
    # ---------------------------------------------------------
    if fit is None:
        if "per_condition" in cca_res:
            fit = "separate"
        elif "cv_results" in cca_res:
            fit = "shared"
        else:
            raise ValueError(
                "Could not infer fit type from cca_res. "
                "Pass fit='shared' or fit='separate'."
            )

    fit = fit.lower()

    if fit not in {"shared", "separate"}:
        raise ValueError("fit must be 'shared' or 'separate'.")

    # ---------------------------------------------------------
    # 2. Obtain condition names
    # ---------------------------------------------------------
    if conditions is None:
        if fit == "shared":
            names = list(
                cca_res["cv_results"]["mean_test_corrs"].keys()
            )
        else:
            names = list(cca_res["per_condition"].keys())
    else:
        names = [
            cond.name if hasattr(cond, "name") else str(cond)
            for cond in conditions
        ]

    if len(names) == 0:
        raise ValueError("No conditions were found.")

    # ---------------------------------------------------------
    # 3. Extract the relevant CV result for each condition
    # ---------------------------------------------------------
    means = []
    stds = []
    fold_values = {}

    for name in names:

        if fit == "shared":
            cv_results = cca_res["cv_results"]
        else:
            if name not in cca_res["per_condition"]:
                raise KeyError(
                    f"Condition {name!r} is not present in "
                    "cca_res['per_condition']."
                )

            cv_results = (
                cca_res["per_condition"][name]["cv_results"]
            )

        if cv_results is None:
            raise ValueError(
                f"No cross-validation results are stored for {name!r}."
            )

        try:
            condition_folds = np.asarray(
                cv_results["test_corrs"][name],
                dtype=float,
            )
        except KeyError as exc:
            raise KeyError(
                f"Could not find CV test correlations for {name!r}."
            ) from exc

        if condition_folds.ndim != 2:
            raise ValueError(
                f"Expected test correlations for {name!r} to have "
                "shape (n_folds, n_components), but received "
                f"{condition_folds.shape}."
            )

        if component < 0 or component >= condition_folds.shape[1]:
            raise IndexError(
                f"component={component} is invalid for {name!r}. "
                f"The result contains {condition_folds.shape[1]} "
                "canonical components."
            )

        values = condition_folds[:, component]

        # Calculate directly from fold values so that the plotted
        # quantities always agree with the displayed individual points.
        mean_value = np.nanmean(values)
        std_value = np.nanstd(values)

        fold_values[name] = values
        means.append(mean_value)
        stds.append(std_value)

    means = np.asarray(means)
    stds = np.asarray(stds)

    # ---------------------------------------------------------
    # 4. Create the plot
    # ---------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(names))

    # Mean and optional SD error bars
    if show_errorbars:
        ax.errorbar(
            x,
            means,
            yerr=stds,
            fmt="o",
            capsize=5,
            linewidth=1.5,
            markersize=7,
            label="Mean ± fold SD",
            zorder=3,
        )
    else:
        ax.scatter(
            x,
            means,
            s=55,
            label="Fold mean",
            zorder=3,
        )

    # Individual CV fold points
    if show_individual:
        rng = np.random.default_rng(42)

        for i, name in enumerate(names):
            values = fold_values[name]

            offsets = rng.uniform(
                -jitter,
                jitter,
                size=len(values),
            )

            ax.scatter(
                np.full(len(values), x[i]) + offsets,
                values,
                alpha=0.65,
                s=35,
                label="Individual folds" if i == 0 else None,
                zorder=2,
            )

    # Reference line at zero, useful when test correlations are negative
    ax.axhline(
        0,
        linewidth=1,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")

    ax.set_xlabel("Condition")
    ax.set_ylabel(
        f"Held-out canonical correlation, CC{component + 1}"
    )

    if title is None:
        title = (
            f"{fit.capitalize()} CCA: "
            f"cross-validated CC{component + 1}"
        )

    ax.set_title(title)

    if show_individual or show_errorbars:
        ax.legend(frameon=False)

    fig.tight_layout()

    plot_data = {
        "fit": fit,
        "component": component,
        "condition_names": names,
        "means": dict(zip(names, means)),
        "stds": dict(zip(names, stds)),
        "fold_values": fold_values,
    }

    return fig, ax, plot_data


# ---------------------------------------------------------------------------
# Linear decoding from CCA space (e.g. Left vs Right)
# ---------------------------------------------------------------------------

def decode_cca_condition(
    cca_res,
    group_a,               # condition name, or list of names, pooled as class 0 (e.g. "Left" side)
    group_b,               # condition name, or list of names, pooled as class 1 (e.g. "Right" side)
    which="X",
    fit=None,
    reg_method="logistic",  # "logistic", "ridge", "svm" (linear), "svm_rbf",
                             # "random_forest", or "xgboost"
    reg_kwargs=None,        # extra kwargs passed to the chosen estimator's constructor
    scale=True,             # z-score each component across trials before decoding
    components=None,       # canonical components to use as decoder features; None = every available one
    balance_conditions=True,  # if True, subsample the larger group to match the smaller one (per timepoint)
    n_splits=5,
    n_permutations=0,      # if > 0, also builds a label-shuffled null accuracy distribution per timepoint
    random_state=42,
    verbose=True,
):
    """
    Time-resolved linear decoding of a binary condition split (e.g. Left vs
    Right) from the CCA canonical-variate scores. At every timepoint, fits
    a cross-validated logistic regression on that timepoint's per-trial
    canonical scores (n_trials, n_components) and reports held-out
    accuracy/AUC -- i.e. how linearly separable the two groups are in CC
    space at that point in the trial. This is independent of (and answers a
    different question from) the X/Y canonical correlation itself: two
    conditions can be strongly correlated between X and Y while still being
    totally inseparable from each other in that same space, or vice versa.

    group_a / group_b : str or list of str
        Condition name(s) (as they appear in cca_res) to pool as each
        decoder class, e.g. group_a=["L_cor", "L_inc"], group_b=["R_cor",
        "R_inc"] to decode side regardless of trial outcome.
    which : {"X", "Y"}, default "X"
    fit : {"shared", "separate"}, optional
        Inferred from cca_res if None. Use "shared" (the default CCA fit
        mode) so group_a/group_b sit in the same component space --
        decoding across a "separate" fit's per-condition spaces isn't
        meaningful.
    reg_method : {"logistic", "ridge", "svm", "svm_rbf", "random_forest", "xgboost", "LDA"}, default "logistic"
        Classifier fit at each timepoint. "logistic", "ridge", and "svm"
        (linear-kernel SVC) are linear decoders -- their held-out
        accuracy/AUC is directly comparable to the linear CCA structure.
        "svm_rbf", "random_forest", and "xgboost" are nonlinear and can
        pick up separability a linear boundary would miss; a gap between a
        linear and nonlinear decoder's accuracy at the same timepoint is
        itself informative. "xgboost" requires the xgboost package.
    reg_kwargs : dict, optional
        Extra constructor kwargs for the chosen estimator (e.g.
        reg_method="random_forest", reg_kwargs={"n_estimators": 500}).
    components : sequence of int, optional
        Zero-based canonical-component indices to use as decoder features.
        None = every available component.
    n_splits : int, default 5
        StratifiedKFold folds for the held-out accuracy/AUC at each timepoint.
    n_permutations : int, default 0
        If > 0, also fits `n_permutations` label-shuffled decoders per
        timepoint to build a null accuracy distribution (for a chance band
        -- see plot_decode_accuracy). Cost scales linearly with this, so
        keep it modest (e.g. 100-500) unless you have time to spare.

    Returns
    -------
    dict with:
        time_accuracy, time_accuracy_std   (n_time,) held-out accuracy, mean/std across folds
        time_auc, time_auc_std             (n_time,) held-out ROC AUC, mean/std across folds
        null_accuracy                      (n_permutations, n_time) or None
        labels                              (n_trials,) 0/1 group labels used (0=group_a, 1=group_b)
        group_a, group_b, components, which, n_time
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def make_decoder():
        kwargs = dict(reg_kwargs) if reg_kwargs else {}
        method = reg_method.lower()
        if method == "logistic":
            base = LogisticRegression(max_iter=1000, random_state=random_state, **kwargs)
        elif method == "ridge":
            base = RidgeClassifier(random_state=random_state, **kwargs)
        elif method in ("svm", "svm_linear"):
            base = SVC(kernel="linear", random_state=random_state, **kwargs)
        elif method == "lda":
            base = LinearDiscriminantAnalysis(**kwargs)
        elif method == "svm_rbf":
            base = SVC(kernel="rbf", random_state=random_state, **kwargs)
        elif method in ("rf", "random_forest"):
            base = RandomForestClassifier(n_estimators=200, random_state=random_state, **kwargs)
        elif method == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError(
                    "reg_method='xgboost' requires the xgboost package (pip install xgboost)."
                ) from exc
            base = XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.1,
                eval_metric="logloss", random_state=random_state, n_jobs=1, **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown reg_method {reg_method!r}; expected 'logistic', 'ridge', "
                f"'svm', 'svm_rbf', 'random_forest', 'LDA', or 'xgboost'."
            )
        return make_pipeline(StandardScaler(), base) if scale else base

    scores_by_cond = _scores_by_condition(cca_res, which=which, fit=fit)

    group_a = [group_a] if isinstance(group_a, str) else list(group_a)
    group_b = [group_b] if isinstance(group_b, str) else list(group_b)
    for name in group_a + group_b:
        if name not in scores_by_cond:
            raise KeyError(f"Condition {name!r} not found in cca_res.")

    X_a = np.concatenate([scores_by_cond[name] for name in group_a], axis=0)
    X_b = np.concatenate([scores_by_cond[name] for name in group_b], axis=0)
    if balance_conditions:
        n_trials = min(len(X_a), len(X_b))
        rng = np.random.default_rng(random_state)
        X_a = rng.choice(X_a, size=n_trials, replace=False)
        X_b = rng.choice(X_b, size=n_trials, replace=False)
    if X_a.shape[1:] != X_b.shape[1:]:
        raise ValueError(
            f"group_a/group_b timepoints/components disagree: "
            f"{X_a.shape[1:]} vs {X_b.shape[1:]}."
        )

    X_all = np.concatenate([X_a, X_b], axis=0)  # (n_trials, n_time, n_components)
    y = np.concatenate([np.zeros(len(X_a)), np.ones(len(X_b))]).astype(int)

    n_time, n_components_avail = X_all.shape[1], X_all.shape[2]
    comp_idx = np.arange(n_components_avail) if components is None else np.asarray(components)

    if verbose:
        print(
            f"Decoding {group_a} (n={len(X_a)}) vs {group_b} (n={len(X_b)}) "
            f"from {which.upper()} CC{list(comp_idx + 1)}, {n_time} timepoints, "
            f"reg_method={reg_method!r}."
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)

    time_accuracy = np.full(n_time, np.nan)
    time_accuracy_std = np.full(n_time, np.nan)
    time_auc = np.full(n_time, np.nan)
    time_auc_std = np.full(n_time, np.nan)
    null_accuracy = np.full((n_permutations, n_time), np.nan) if n_permutations else None

    for ti in range(n_time):
        feats = X_all[:, ti, :][:, comp_idx]
        clf = make_decoder()

        cv_res = cross_validate(clf, feats, y, cv=cv, scoring=["accuracy", "roc_auc"])
        time_accuracy[ti] = cv_res["test_accuracy"].mean()
        time_accuracy_std[ti] = cv_res["test_accuracy"].std()
        time_auc[ti] = cv_res["test_roc_auc"].mean()
        time_auc_std[ti] = cv_res["test_roc_auc"].std()

        for p in range(n_permutations):
            y_perm = rng.permutation(y)
            perm_res = cross_validate(clf, feats, y_perm, cv=cv, scoring="accuracy")
            null_accuracy[p, ti] = perm_res["test_score"].mean()

    return {
        "time_accuracy": time_accuracy,
        "time_accuracy_std": time_accuracy_std,
        "time_auc": time_auc,
        "time_auc_std": time_auc_std,
        "null_accuracy": null_accuracy,
        "labels": y,
        "reg_method": reg_method,
        "group_a": group_a,
        "group_b": group_b,
        "components": comp_idx,
        "which": which,
        "n_time": n_time,
    }


def plot_decode_accuracy(
    decode_res,
    time=None,
    metric="accuracy",   # "accuracy" or "auc"
    chance=0.5,
    figsize=(7, 5),
    title=None,
    ax=None,
):
    """
    Plot the time-resolved decoding performance from decode_cca_condition:
    mean +/- std across CV folds, a dashed chance line, touch onset (if
    `time` is given), and -- if decode_cca_condition was run with
    n_permutations > 0 -- a shaded null band from the label-shuffled
    control (2.5th-97.5th percentile).

    Parameters
    ----------
    decode_res : dict
        Output of decode_cca_condition.
    time : array-like, optional
        (n_timepoints,) real time axis (e.g. organised.time_zF); sample
        index if None.
    metric : {"accuracy", "auc"}, default "accuracy"
    chance : float, default 0.5
        Reference chance line (0.5 for a balanced binary decode).

    Returns
    -------
    fig, ax
    """
    if metric not in ("accuracy", "auc"):
        raise ValueError("metric must be 'accuracy' or 'auc'.")

    mean = decode_res[f"time_{metric}"]
    std = decode_res[f"time_{metric}_std"]
    n_time = len(mean)
    t = np.asarray(time) if time is not None else np.arange(n_time)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if metric == "accuracy" and decode_res.get("null_accuracy") is not None:
        lo, hi = np.nanpercentile(decode_res["null_accuracy"], [2.5, 97.5], axis=0)
        ax.fill_between(t, lo, hi, color="orange", alpha=0.3, label="Null (shuffled labels)")

    ax.plot(t, mean, color="black", linewidth=2, label=metric.upper())
    ax.fill_between(t, mean - std, mean + std, color="black", alpha=0.2)
    ax.axhline(chance, linestyle="--", color="gray", linewidth=1, label="Chance")
    if time is not None:
        touch_t = t[int(np.argmin(np.abs(t)))]
        ax.axvline(touch_t, linestyle=":", color="red", linewidth=1, label="Touch")

    ax.set_xlabel("Time (s)" if time is not None else "Timepoint")
    ax.set_ylabel("Accuracy" if metric == "accuracy" else "ROC AUC")

    group_a, group_b = decode_res["group_a"], decode_res["group_b"]
    reg_method = decode_res.get("reg_method")
    method_suffix = f", {reg_method}" if reg_method else ""
    ax.set_title(
        title or f"Decoding {'/'.join(group_a)} vs {'/'.join(group_b)} ({decode_res['which']}{method_suffix})"
    )
    ax.legend(frameon=False)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Scree plot: how much X-Y correlation each canonical component captures
# ---------------------------------------------------------------------------

def _final_corrs_by_condition(cca_res, fit=None):
    """Normalize shared/separate cca_res into {name: {"train": arr, "test": arr_or_None}}."""
    if fit is None:
        fit = "separate" if "per_condition" in cca_res else "shared"

    if fit == "shared":
        return cca_res["final_corrs"]

    return {name: block["final_corrs"] for name, block in cca_res["per_condition"].items()}


def plot_cca_scree(
    cca_res,
    fit=None,
    conditions=None,
    split="train",         # "train", "test", or "both" -- which fitted correlations to show
    cond_colours=None,     # per-condition line colour: single colour, [c1, c2, ...], or {cond_name: colour}
    figsize=(7, 5),
    title=None,
    ax=None,
):
    """
    Scree-style plot of the final fitted canonical correlation per
    component (CC1, CC2, ...), one line per condition -- how much X-Y
    correlation each successive canonical component captures, and how
    quickly that drops off. Reads `final_corrs` from run_shared_cca's
    output, so this is the fitted (and held-out, if test_size was set)
    correlation, not the cross-validated diagnostic (see
    plot_cca_cv_correlations for that).

    Parameters
    ----------
    cca_res : dict
        Output of run_shared_cca.
    fit : {"shared", "separate"}, optional
        Inferred from cca_res if None.
    conditions : sequence, optional
        Conditions to plot (objects with `.name`, or name strings). Defaults
        to every condition in cca_res.
    split : {"train", "test", "both"}, default "train"
        Plot the fitted (train) correlations, held-out (test) correlations
        (only available if run_shared_cca was called with test_size set),
        or both (test drawn dashed with square markers).
    cond_colours : color, list, or dict, optional
        Per-condition line colour, same shape rules as elsewhere in this
        module (single value, list cycled by order, or {cond_name: colour}).

    Returns
    -------
    fig, ax, plot_data
    """
    if split not in ("train", "test", "both"):
        raise ValueError("split must be 'train', 'test', or 'both'.")

    corrs_by_cond = _final_corrs_by_condition(cca_res, fit=fit)

    if conditions is None:
        names = list(corrs_by_cond.keys())
    else:
        names = [c.name if hasattr(c, "name") else str(c) for c in conditions]

    default_colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cond_colours = _resolve_per_condition(cond_colours, names, default_colours)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    corrs = {}
    max_components = 0
    for name in names:
        if name not in corrs_by_cond:
            raise KeyError(f"Condition {name!r} not found in cca_res.")
        block = corrs_by_cond[name]
        train_corrs = np.asarray(block["train"])
        test_corrs = block.get("test")
        test_corrs = np.asarray(test_corrs) if test_corrs is not None else None
        max_components = max(max_components, len(train_corrs))
        colour = cond_colours[name]
        cc_idx = np.arange(1, len(train_corrs) + 1)

        if split in ("train", "both"):
            ax.plot(
                cc_idx, train_corrs, marker="o", color=colour, linestyle="-",
                label=f"{name} (train)" if split == "both" else name,
            )
        if split in ("test", "both"):
            if test_corrs is None:
                if split == "test":
                    raise ValueError(
                        f"No test correlations for {name!r} -- run_shared_cca must be "
                        "called with test_size set to get held-out correlations."
                    )
            else:
                ax.plot(
                    cc_idx, test_corrs, marker="s", color=colour, linestyle="--",
                    label=f"{name} (test)" if split == "both" else name,
                )

        corrs[name] = {"train": train_corrs, "test": test_corrs}

    ax.set_xticks(np.arange(1, max_components + 1))
    ax.set_xlabel("Canonical component")
    ax.set_ylabel("Canonical correlation")
    ax.set_title(title or "CCA scree plot: canonical correlation per component")
    ax.legend(frameon=False)
    fig.tight_layout()

    plot_data = {"fit": fit, "split": split, "corrs": corrs}
    return fig, ax, plot_data