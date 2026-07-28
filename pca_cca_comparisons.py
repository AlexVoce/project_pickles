# pca cca comparisons? 

# ---------------------------------------------------------------------------
# PCA vs CCA eigenvector comparison
# ---------------------------------------------------------------------------
# Answers: do the directions CCA picks to maximise X-Y correlation also
# happen to be the directions that account for the most variance in X (or
# Y) alone? Reuses the exact same preprocessing helpers as the rest of this
# module (flatten_trials, finite_feature_mask, fit_scaler, fit_pca,
# canonical_corrs) so results are directly comparable to run_shared_cca.

def _cca_effective_directions(pca, cca, n_features_z, which="x", eps=1e-4):
    """
    Map CCA's canonical directions back into the PCA's INPUT space (i.e.
    the same scaled-feature space the PCA eigenvectors live in), so the two
    can be compared directly.

    sklearn's CCA re-standardizes its input internally before fitting, so
    its exposed weight vectors (`x_rotations_` etc.) are not directly
    comparable to PCA eigenvectors without undoing that internal scaling.
    Rather than depend on CCA's private standardization attributes, this
    finite-differences the actual fitted `pca.transform -> cca.transform`
    composition (both are affine, so this is exact up to float precision)
    -- correct regardless of sklearn version or internal scaling choices.

    Returns (n_features_z, n_cca_components) unit-norm direction vectors.
    """
    which = which.lower()
    if which not in ("x", "y"):
        raise ValueError("which must be 'x' or 'y'.")

    zero = np.zeros((1, n_features_z))
    basis = np.eye(n_features_z) * eps
    mat = np.vstack([zero, basis])
    p = pca.transform(mat)  # (n_features_z + 1, n_pca_components)

    if which == "x":
        scores = cca.transform(p)
    else:
        dummy_x = np.zeros((p.shape[0], cca.x_rotations_.shape[0]))
        _, scores = cca.transform(dummy_x, p)

    grad = (scores[1:] - scores[0]) / eps  # (n_features_z, n_cca_components)
    norms = np.linalg.norm(grad, axis=0, keepdims=True)
    return grad / np.where(norms == 0, 1, norms)


def compare_pca_cca_directions(
    X_trials,
    Y_trials,
    n_pca_components=20,
    n_cca_components=5,
    n_compare=None,
    scale=True,
    verbose=True,
):
    """
    Compare the top PCA eigenvectors of X (npix) and Y (zF) against the
    directions a CCA fit on the same data uses to maximise X-Y correlation.

    Preprocessing matches run_shared_cca/_fit_and_project exactly:
    flatten_trials -> finite_feature_mask -> fit_scaler (StandardScaler,
    optional) -> PCA -> CCA (fit on the PCA-reduced features, mirroring
    run_shared_cca with n_pca_components set). Fits on ALL trials passed in
    (no train/test split) since this is a diagnostic about the geometry of
    the fitted directions, not a held-out performance estimate.

    IMPORTANT: n_pca_components (the PCA dimensionality CCA is fit on top
    of) must be larger than n_compare for the subspace-angle comparison to
    mean anything -- if CCA is given exactly as many PCA dimensions as
    directions being compared, its directions trivially span the whole PCA
    space and the "subspace angle" is 0 by construction, regardless of
    whether individual directions actually line up. Use a generous
    n_pca_components (e.g. 20-30) and a smaller n_compare (e.g. 3-5) to ask
    whether CCA's top few directions live within PCA's top few, or reach
    into lower-variance PCA dimensions PCA alone would have discarded.

    For each side (X, Y) reports:
      - pca_evr             PCA explained_variance_ratio_, ALL n_pca_components
                             (full scree spectrum, for reference)
      - cca_var_explained   fraction of that side's total (scaled) variance
                             captured by each CCA direction ALONE -- the
                             CCA analogue of pca_evr, directly comparable
      - cosine_sim          (n_pca_components, n_cca_components) |cosine
                             similarity| between every PCA eigenvector (row)
                             and CCA direction (column) -- e.g. a CCA
                             direction that loads onto a LOW-ranked PCA
                             eigenvector (not just PCA1) will show up here
      - subspace_angles_deg principal angles (degrees) between
                             span(top n_compare PCA eigenvectors) and
                             span(top n_compare CCA directions), both
                             within the n_pca_components-dim ambient PCA
                             space -- 0 deg means the two subspaces
                             coincide, 90 deg means CCA is using variance
                             PCA's top ranks would discard as noise

    Parameters
    ----------
    X_trials, Y_trials : (n_trials, n_timepoints, n_features)
        Same trial-level arrays returned by collect_trials.
    n_pca_components : int, default=20
        PCA dimensionality CCA is fit on top of (equivalent to
        run_shared_cca's n_pca_components). Should be noticeably larger
        than n_compare -- see note above.
    n_cca_components : int, default=5
        Number of CCA canonical directions to fit.
    n_compare : int, optional
        Number of top PCA eigenvectors / CCA directions used for the
        subspace-angle comparison and the variance bar-plot. Defaults to
        n_cca_components. Must be <= min(n_pca_components, n_cca_components).
    scale : bool, default=True
        z-score X/Y before PCA, matching run_shared_cca's `scale` switch.

    Returns
    -------
    dict with keys "X" and "Y" (each a dict as described above), plus:
        canonical_corrs      (n_cca_components,) this CCA's own X-Y
                              canonical correlations, for reference
        n_pca_components, n_cca_components, n_compare, good_X, good_Y
    """
    X_trials = np.asarray(X_trials, dtype=float)
    Y_trials = np.asarray(Y_trials, dtype=float)

    X = flatten_trials(X_trials)
    Y = flatten_trials(Y_trials)
    good_X, good_Y = finite_feature_mask(X), finite_feature_mask(Y)
    X, Y = X[:, good_X], Y[:, good_Y]

    scaler_x, scaler_y, X_z, Y_z = fit_scaler(X, Y, scale=scale)
    pca_x, pca_y, X_p, Y_p = fit_pca(X_z, Y_z, n_components=n_pca_components)

    n_cca = min(n_cca_components, X_p.shape[1], Y_p.shape[1])
    n_compare = min(n_compare or n_cca, n_cca, X_p.shape[1], Y_p.shape[1])

    cca = CCA(n_components=n_cca, max_iter=2000)
    cca.fit(X_z, Y_z)
    X_c, Y_c = cca.transform(X_z, Y_z)
    corrs = canonical_corrs(X_c, Y_c)

    def _side(Z, pca_obj, which):
        pca_dirs = pca_obj.components_.T  # (n_features_z, n_pca_components), unit columns
        cca_dirs = _cca_effective_directions(pca_obj, cca, Z.shape[1], which=which)
        cosine_sim = np.abs(pca_dirs.T @ cca_dirs)  # (n_pca_components, n_cca)

        total_var = Z.var(axis=0, ddof=1).sum()
        cca_var_explained = (Z @ cca_dirs).var(axis=0, ddof=1) / total_var

        angles_deg = np.degrees(subspace_angles(
            pca_dirs[:, :n_compare], cca_dirs[:, :n_compare],
        ))
        return {
            "pca_evr": pca_obj.explained_variance_ratio_,
            "cca_var_explained": cca_var_explained,
            "cosine_sim": cosine_sim,
            "subspace_angles_deg": angles_deg,
        }

    result = {
        "X": _side(X_z, pca_x, "x"),
        "Y": _side(Y_z, pca_y, "y"),
        "canonical_corrs": corrs,
        "n_pca_components": n_pca_components,
        "n_cca_components": n_cca,
        "n_compare": n_compare,
        "good_X": good_X,
        "good_Y": good_Y,
    }

    if verbose:
        print(f"Canonical correlations: {np.round(corrs, 3)}")
        for side in ("X", "Y"):
            print(f"[{side}] PCA explained variance ratio (top {n_compare} of {n_pca_components}): "
                  f"{np.round(result[side]['pca_evr'][:n_compare], 3)}")
            print(f"[{side}] Variance explained by CCA directions alone (top {n_compare}): "
                  f"{np.round(result[side]['cca_var_explained'][:n_compare], 3)}")
            print(f"[{side}] Subspace angles (deg), top-{n_compare} PCA vs top-{n_compare} CCA: "
                  f"{np.round(result[side]['subspace_angles_deg'], 1)}")

    return result


def plot_pca_cca_comparison(result, which="X", n_show=None, figsize=(11, 4.5), title=None):
    """
    Visualize the output of compare_pca_cca_directions for one side.

    Left panel: rank-matched PCA explained_variance_ratio_ vs how much of
    that side's variance each CCA direction captures alone -- NOT a claim
    that CCA direction k equals PCA eigenvector k (see the cosine-
    similarity panel for that), just two variance curves side by side,
    truncated to `n_show` (defaults to result["n_compare"], the same
    top-x used for the subspace-angle comparison). Right panel: |cosine
    similarity| heatmap between every PCA eigenvector (rows, full
    n_pca_components spectrum) and CCA direction (columns) -- this is
    where a CCA direction loading onto a low-ranked, low-variance PCA
    eigenvector would show up.

    Returns fig, (ax_bar, ax_heat), the `result[which]` dict that was plotted.
    """
    which = which.upper()
    side = result[which]
    n_show = n_show or result.get("n_compare", len(side["cca_var_explained"]))
    pca_evr = side["pca_evr"][:n_show]
    cca_var = side["cca_var_explained"][:n_show]
    cosine_sim = side["cosine_sim"]

    fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=figsize)

    n = len(pca_evr)
    x = np.arange(n)
    width = 0.35
    ax_bar.bar(x - width / 2, pca_evr, width, label="PCA eigenvector")
    ax_bar.bar(x + width / 2, cca_var, width, label="CCA direction")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(i + 1) for i in x])
    ax_bar.set_xlabel("Component rank")
    ax_bar.set_ylabel(f"Fraction of {which} variance")
    ax_bar.set_title("Variance captured, PCA vs CCA")
    ax_bar.legend(frameon=False)

    im = ax_heat.imshow(cosine_sim, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xlabel("CCA direction")
    ax_heat.set_ylabel("PCA eigenvector")
    ax_heat.set_xticks(np.arange(cosine_sim.shape[1]))
    ax_heat.set_xticklabels(np.arange(1, cosine_sim.shape[1] + 1))
    ax_heat.set_yticks(np.arange(cosine_sim.shape[0]))
    ax_heat.set_yticklabels(np.arange(1, cosine_sim.shape[0] + 1))
    ax_heat.set_title("|cosine similarity|")
    fig.colorbar(im, ax=ax_heat, label="|cos angle|")

    angles_str = np.round(side["subspace_angles_deg"], 1)
    fig.suptitle(title or f"PCA vs CCA ({which}) -- subspace angles (deg): {angles_str}")
    fig.tight_layout()

    return fig, (ax_bar, ax_heat), side