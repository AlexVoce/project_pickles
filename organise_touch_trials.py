"""
organise_touch_trials.py

organises touchDict's per-trial imaging time series (zF_touchMat, npix_touchMat)
by trial condition (cue side x correct/incorrect x 1port/2port, from
touchDict['trialTypeInds']), downsampling npix_touchMat onto the same time
grid as zF_touchMat, and (optionally) wrapping everything in pynapple objects
so you get restrict()/mean()/plotting for free.

ASSUMPTIONS:
  - zF_touchMat and npix_touchMat both have a trial axis of length n_trials
    (n_trials = len(seshParams['trialOnT'])).
  - Of the two remaining axes in each array, one is "time" and one is
    "feature" (neuron/ROI/pixel -- zF and npix do NOT need to share a feature
    count; they can be entirely different signal types, e.g. zF = per-neuron
    fluorescence with ~195 neurons, npix = a 61-channel touch signal). The
    time axis is identified by its expected sample count (`zF_n_time=120`,
    `npix_n_time=400` by default -- override if yours differ, or pass
    `zF_time_axis`/`npix_time_axis` directly to skip detection). Works
    whether your raw layout is (trial, time, feature) or (trial, feature,
    time), e.g. zF_touchMat as (n_trials, n_neurons, n_time). Everything
    downstream (ConditionData.zF, .npix_ds, mean_zF()[:, feature_idx], etc.)
    is canonicalized to (trial, time, feature), so `feature_idx` always
    indexes neurons/ROIs/channels.
  - Both time axes span the same real peri-touch window. Default window is
    [-(preT+baseSec), +(postT+plotT)] seconds relative to trialOnT, using
    seshParams['dlcSets'] preT/postT and seshParams baseSec/plotT. Override
    with `window=(t0, t1)` on build_time_axis() / organise_dataset() if this
    guess is wrong for your pipeline.
  - trialTypeInds values are trial *indices* (0-based, into the trial axis),
    consistent with cueSide/portSide/rewardSide ordering.
  - iZone is a per-NEURON label (e.g. microzone identity), aligned to zF's
    neuron axis -- NOT per-trial. It's attached as organisedDataset.neuron_zone
    (only if its length matches zF's neuron count) and used via
    organised.zF_by_zone(cond_name, zone_id) / organised.zone_ids(), not as
    per-trial metadata on ConditionData.

Core entry point: `organise_dataset(touchDict)` -> organisedDataset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    import pynapple as nap
    HAVE_PYNAPPLE = True
except ImportError:
    HAVE_PYNAPPLE = False


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------

def find_trial_axis(arr: np.ndarray, n_trials: int) -> int:
    """Return the index of the axis in `arr` whose length equals n_trials."""
    matches = [ax for ax, sz in enumerate(arr.shape) if sz == n_trials]
    if not matches:
        raise ValueError(
            f"No axis of length {n_trials} found in array with shape {arr.shape}. "
            "Check that n_trials (len(trialOnT)) matches your data."
        )
    if len(matches) > 1:
        # Ambiguous (e.g. square-ish array). Default to first axis but warn.
        print(
            f"[organise_touch_trials] WARNING: multiple axes of length {n_trials} "
            f"in shape {arr.shape}; using axis {matches[0]}. Pass trial_axis "
            "explicitly if this is wrong."
        )
    return matches[0]


def find_time_axis(arr: np.ndarray, trial_axis: int, expected_len: int) -> int:
    """Return the index of the axis (excluding trial_axis) matching expected_len."""
    candidates = [ax for ax, sz in enumerate(arr.shape)
                  if ax != trial_axis and sz == expected_len]
    if not candidates:
        raise ValueError(
            f"No axis of length {expected_len} found (excluding trial axis {trial_axis}) "
            f"in shape {arr.shape}."
        )
    return candidates[0]


def move_to_front(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(arr, axis, 0)


def _resolve_time_axis(arr: np.ndarray, expected_len: Optional[int], override: Optional[int],
                        name: str) -> int:
    """
    Pick which of arr's two post-trial axes (1 or 2) is "time". `override` wins
    if given. Otherwise, look for an axis whose length equals `expected_len`.
    """
    if override is not None:
        return override
    if expected_len is not None:
        candidates = [ax for ax in (1, 2) if arr.shape[ax] == expected_len]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 2:
            print(f"[organise_touch_trials] WARNING: both post-trial axes of {name} "
                  f"equal length {expected_len} ({arr.shape}); defaulting to axis 2 as time. "
                  f"Pass {name}_time_axis=1 or 2 explicitly if this is wrong.")
            return 2
        raise ValueError(
            f"Could not find an axis of length {expected_len} (expected n_time) in "
            f"{name} post-trial shape {arr.shape[1:]}. zF and npix don't have to share "
            f"a feature count (e.g. neurons vs. touch pixels), so pass "
            f"{name}_n_time=<actual timepoints> or {name}_time_axis=1/2 explicitly."
        )
    raise ValueError(
        f"No expected_len or override given to resolve the time axis for {name} "
        f"(post-trial shape {arr.shape[1:]})."
    )


def orient_trial_time_feature(zF: np.ndarray, npixMat: np.ndarray, n_trials: int,
                               zF_n_time: Optional[int] = 120, npix_n_time: Optional[int] = 400,
                               zF_time_axis: Optional[int] = None,
                               npix_time_axis: Optional[int] = None):
    """
    Canonicalize zF/npixMat to shape (n_trials, n_time, n_features), regardless
    of whether the raw layout is (trial, time, feature) or (trial, feature, time)
    (e.g. zF_touchMat as (n_trials, n_neurons, n_time)).

    zF and npix_touchMat do NOT need to share a feature count -- they can be
    entirely different signal types (e.g. zF = per-neuron fluorescence,
    npix = per-pixel/touch-sensor signal). The time axis for each is instead
    identified by its expected number of samples (`zF_n_time`, `npix_n_time`,
    default 120/400 to match this pipeline), or pass `zF_time_axis`/
    `npix_time_axis` (1 or 2) directly to skip detection entirely.
    """
    zF = move_to_front(zF, find_trial_axis(zF, n_trials))
    npixMat = move_to_front(npixMat, find_trial_axis(npixMat, n_trials))

    if zF.ndim != 3 or npixMat.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays (trial, time, feature) in some order; got zF shape "
            f"{zF.shape}, npix shape {npixMat.shape}."
        )

    zF_time_ax = _resolve_time_axis(zF, zF_n_time, zF_time_axis, 'zF')
    npix_time_ax = _resolve_time_axis(npixMat, npix_n_time, npix_time_axis, 'npix')

    zF_feat_ax = 2 if zF_time_ax == 1 else 1
    npix_feat_ax = 2 if npix_time_ax == 1 else 1

    print(
        f"[organise_touch_trials] Detected layout: zF time axis={zF_time_ax} "
        f"(n_time={zF.shape[zF_time_ax]}, n_features={zF.shape[zF_feat_ax]}); "
        f"npix time axis={npix_time_ax} (n_time={npixMat.shape[npix_time_ax]}, "
        f"n_features={npixMat.shape[npix_feat_ax]}). Canonicalizing both to (trial, time, feature)."
    )

    zF = np.moveaxis(zF, zF_time_ax, 1)      # -> (trial, time, feature)
    npixMat = np.moveaxis(npixMat, npix_time_ax, 1)
    return zF, npixMat


_warned_meta_mismatch = set()


def safe_index_meta(arr: np.ndarray, idx: np.ndarray, n_trials: int, name: str) -> np.ndarray:
    """
    Index a per-trial metadata array (cueSide, iZone, ...) at `idx`, but only if
    `arr` actually has one entry per trial (len == n_trials). Some sessions have
    metadata arrays that are shorter than n_trials/trialOnT (e.g. trials dropped
    from a downstream computation like iZone); in that case, indices beyond the
    array's length can't be resolved, so this returns NaN/None placeholders for
    those trials instead of crashing, and warns once per array name.
    """
    if len(arr) == n_trials:
        return arr[idx]

    if name not in _warned_meta_mismatch:
        print(
            f"[organise_touch_trials] WARNING: '{name}' has length {len(arr)}, "
            f"but n_trials (from trialOnT) is {n_trials}. Filling out-of-range "
            "entries with NaN/None for this field -- the actual signals (zF/npix) "
            "are unaffected. Double check whether this array is trial-aligned."
        )
        _warned_meta_mismatch.add(name)

    fill = np.nan if np.issubdtype(arr.dtype, np.number) else None
    out = np.full(idx.shape, fill, dtype=object if fill is None else float)
    in_range = idx < len(arr)
    out[in_range] = arr[idx[in_range]]
    return out


def build_time_axis(seshParams: dict, n_points: int, window: Optional[tuple] = None) -> np.ndarray:
    """
    Build a relative (seconds, 0 = touch/trial-on time) time axis with n_points
    samples spanning `window` = (t0, t1). If window is None, derive a default
    from seshParams (preT/postT/baseSec/plotT) -- see module docstring caveat.
    """
    if window is None:
        plotT = seshParams['plotT']
        preT = -plotT
        postT = plotT
        # baseSec = seshParams.get('baseSec', 0)
        # plotT = seshParams.get('plotT', 0)
        # t0 = -(preT + baseSec)
        # t1 = postT + plotT
        window = (preT, postT)
    return np.linspace(window[0], window[1], n_points)


def downsample_along_axis(arr: np.ndarray, axis: int, target_n: int,
                           sigma_scale: float = 0.5) -> np.ndarray:
    """
    Gaussian-smooth `arr` along `axis` (anti-aliasing lowpass sized to the
    decimation ratio n_in/target_n), then downsample onto `target_n` points
    via linear interpolation over the same normalized [0, 1] index range.
    `sigma_scale` scales the Gaussian sigma (in units of *input* samples);
    0.5 means sigma = half the decimation factor, a standard anti-alias
    heuristic. No-op if arr is already at target_n along `axis`.
    """
    n_in = arr.shape[axis]
    if n_in == target_n:
        return arr

    decim_factor = n_in / target_n
    if decim_factor > 1:
        # sigma = sigma_scale * decim_factor
        sigma=0.025
        arr = gaussian_filter1d(arr, sigma=sigma, axis=axis, mode='nearest')
    # else: upsampling (target_n > n_in) -- skip smoothing, just interpolate

    x_in = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, target_n)

    def interp_1d(y):
        return np.interp(x_out, x_in, y)

    return np.apply_along_axis(interp_1d, axis, arr)


# --------------------------------------------------------------------------
# Merging touchDicts split by task condition (e.g. 1-port vs 2-port) for the
# same animal/session
# --------------------------------------------------------------------------

def _is_per_trial_array(value, n_trials: int) -> bool:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        return arr.ndim >= 1 and arr.shape[0] == n_trials
    return False


def align_npix_units(arr: np.ndarray, unit_ids: np.ndarray, union_ids: np.ndarray,
                      unit_axis: int) -> np.ndarray:
    """
    Re-index `arr`'s unit axis (currently ordered as `unit_ids`) onto the shared
    `union_ids` ordering, filling NaN for units this array doesn't have. Used to
    align npix_touchMat across sessions/conditions where different subsets of
    Neuropixels units were recorded/sorted (so raw column position isn't
    comparable, but the unit ID is).
    """
    unit_ids = np.asarray(unit_ids)
    union_ids = np.asarray(union_ids)
    dst_pos = {uid: i for i, uid in enumerate(union_ids)}
    dst_idx = np.array([dst_pos[uid] for uid in unit_ids])

    out_shape = list(arr.shape)
    out_shape[unit_axis] = len(union_ids)
    out = np.full(out_shape, np.nan, dtype=float)

    arr_front = np.moveaxis(arr, unit_axis, 0)
    out_front = np.moveaxis(out, unit_axis, 0)  # view sharing memory with `out`
    out_front[dst_idx] = arr_front
    return np.moveaxis(out_front, 0, unit_axis)


def merge_touchDicts(touchDicts: list, labels: Optional[list] = None,
                      zF_key: str = 'zF_touchMat', npix_key: str = 'npix_touchMat') -> dict:
    """
    Merge touchDicts for the SAME animal/session that were split by task
    condition (e.g. sg024 2026-02-24 v2=1port / v3=2port touchDict_aligned.json)
    into one touchDict that organise_dataset() can consume directly.

    - zF_touchMat is assumed already aligned across dicts on its neuron axis
      (same neurons, same order -- checked via shape and via `iZone`, which is
      per-neuron and must match exactly). Trials are simply concatenated.
    - npix_touchMat's unit axis is NOT assumed aligned: different subsets of
      Neuropixels units can be sorted/held per file, so each dict's units are
      re-indexed onto the union of seshParams['npixUnitIds'] across all dicts
      (align_npix_units), filling NaN where a unit is absent, before trials
      are concatenated.
    - trialTypeInds are merged key-by-key, with each dict's trial indices
      offset by the number of trials preceding it in the concatenated trial
      axis (works whether condition names are disjoint, e.g. "..._1port_cor"
      vs "..._2port_cor", or shared).
    - Per-trial seshParams arrays (trialOnT, cueSide, portSide, rewardSide,
      touchOnT, sourceTrialIds, sourceTrialRows, ...) are concatenated in the
      same order as the trial axis; scalar/config seshParams entries (mouse,
      expDate, fov, iPlane, ...) are taken from the first dict. A per-trial
      'sourceLabel' array is added so you can tell which original dict each
      merged trial came from.

    NOTE: trialOnT is concatenated as-is (no offsetting), which is only
    meaningful if all dicts share one absolute clock (e.g. 1-port/2-port
    trials interleaved within one continuous recording, as opposed to two
    separate back-to-back recordings). If the trialOnT ranges across dicts
    don't overlap at all, this prints a warning -- double check in that case
    before relying on absolute-time reconstruction (build_pynapple_view).
    """
    if len(touchDicts) < 2:
        raise ValueError("merge_touchDicts needs at least 2 touchDicts to merge.")

    sps = [d['seshParams'] for d in touchDicts]
    n_trials_list = [len(np.asarray(sp['trialOnT'])) for sp in sps]
    if labels is None:
        labels = [sp.get('sourceVersion', f"dict{i}") for i, sp in enumerate(sps)]

    mice = {sp.get('mouse') for sp in sps}
    dates = {sp.get('expDate') for sp in sps}
    if len(mice) > 1 or len(dates) > 1:
        print(f"[organise_touch_trials] WARNING: merging touchDicts across different "
              f"mouse/expDate ({mice}, {dates}) -- make sure this is intended.")

    trialOnT_list = [np.asarray(sp['trialOnT']) for sp in sps]
    if not all(t2.min() <= t1.max() and t1.min() <= t2.max()
               for t1, t2 in zip(trialOnT_list[:-1], trialOnT_list[1:])):
        print("[organise_touch_trials] WARNING: trialOnT ranges don't overlap across "
              "touchDicts being merged -- confirm these dicts share one absolute clock "
              "before relying on build_pynapple_view's absolute-time reconstruction.")

    # --- zF: move each dict's trial axis to front, check the rest matches ---
    zF_list = []
    for d, n_trials in zip(touchDicts, n_trials_list):
        arr = np.asarray(d[zF_key])
        zF_list.append(move_to_front(arr, find_trial_axis(arr, n_trials)))
    ref_shape = zF_list[0].shape[1:]
    for lbl, arr in zip(labels, zF_list):
        if arr.shape[1:] != ref_shape:
            raise ValueError(
                f"'{zF_key}' feature/time shape mismatch across touchDicts to merge: "
                f"'{labels[0]}' has {ref_shape}, '{lbl}' has {arr.shape[1:]}. zF is expected "
                "to already be aligned (same neurons, same order) across merged dicts."
            )
    zF_merged = np.concatenate(zF_list, axis=0)

    # --- iZone: per-neuron, must match exactly (else the neuron axes aren't really aligned) ---
    iZone_ref = np.asarray(touchDicts[0]['iZone']) if 'iZone' in touchDicts[0] else None
    if iZone_ref is not None:
        for lbl, d in zip(labels, touchDicts):
            if 'iZone' not in d or not np.array_equal(np.asarray(d['iZone']), iZone_ref):
                raise ValueError(
                    f"'iZone' differs (or is missing) at '{lbl}' relative to '{labels[0]}'. "
                    "iZone is per-neuron and should be identical if zF's neuron axis is "
                    "truly aligned -- this suggests these dicts are NOT alignable as-is."
                )

    # --- npix: union of unit IDs, re-index each dict's unit axis onto that union ---
    unit_ids_list = [np.asarray(sp['npixUnitIds']) for sp in sps]
    union_ids = np.unique(np.concatenate(unit_ids_list))

    npix_list = []
    for lbl, d, n_trials, unit_ids in zip(labels, touchDicts, n_trials_list, unit_ids_list):
        arr = np.asarray(d[npix_key])
        arr = move_to_front(arr, find_trial_axis(arr, n_trials))  # -> (trial, ?, ?)
        unit_ax = [ax for ax in (1, 2) if arr.shape[ax] == len(unit_ids)]
        if len(unit_ax) != 1:
            raise ValueError(
                f"Could not uniquely identify the npix unit axis for '{lbl}' (post-trial "
                f"shape {arr.shape[1:]}, expected one axis of length {len(unit_ids)} "
                "matching seshParams['npixUnitIds'])."
            )
        npix_list.append(align_npix_units(arr, unit_ids, union_ids, unit_axis=unit_ax[0]))
    npix_merged = np.concatenate(npix_list, axis=0)

    # --- trialTypeInds: merge keys, offsetting each dict's local trial indices ---
    offsets = np.cumsum([0] + n_trials_list[:-1])
    trialTypeInds_merged: dict = {}
    for d, offset in zip(touchDicts, offsets):
        for cond, idx in d.get('trialTypeInds', {}).items():
            idx = np.asarray(idx, dtype=int) + offset
            if cond in trialTypeInds_merged:
                trialTypeInds_merged[cond] = np.concatenate([trialTypeInds_merged[cond], idx])
            else:
                trialTypeInds_merged[cond] = idx

    # --- seshParams: concatenate per-trial arrays, keep first dict's scalar/config entries ---
    merged_sp = dict(sps[0])
    for key, val in sps[0].items():
        if _is_per_trial_array(val, n_trials_list[0]):
            pieces = []
            for lbl, sp, n_trials in zip(labels, sps, n_trials_list):
                v = sp.get(key)
                if not _is_per_trial_array(v, n_trials):
                    raise ValueError(
                        f"seshParams['{key}'] is per-trial in '{labels[0]}' but missing or "
                        f"wrong-length at '{lbl}'; can't concatenate."
                    )
                pieces.append(np.asarray(v))
            merged_sp[key] = np.concatenate(pieces)

    merged_sp['sourceLabel'] = np.concatenate([
        np.full(n, lbl, dtype=object) for lbl, n in zip(labels, n_trials_list)
    ])
    merged_sp['npixUnitIds'] = union_ids

    merged = {
        'seshParams': merged_sp,
        'trialTypeInds': trialTypeInds_merged,
        zF_key: zF_merged,
        npix_key: npix_merged,
    }
    if iZone_ref is not None:
        merged['iZone'] = iZone_ref
    return merged


# --------------------------------------------------------------------------
# Main organizing structure
# --------------------------------------------------------------------------

@dataclass
class ConditionData:
    name: str
    trial_idx: np.ndarray            # original trial indices for this condition
    zF: np.ndarray                   # (n_trials_cond, n_time, n_neurons)
    npix_ds: np.ndarray              # (n_trials_cond, n_time, n_npix_features) downsampled to same n_time as zF
    cueSide: np.ndarray
    portSide: np.ndarray
    rewardSide: np.ndarray
    trialOnT: np.ndarray

    def mean_zF(self) -> np.ndarray:
        return np.nanmean(self.zF, axis=0)

    def sem_zF(self) -> np.ndarray:
        return np.nanstd(self.zF, axis=0) / np.sqrt(self.zF.shape[0])

    def mean_npix(self) -> np.ndarray:
        return np.nanmean(self.npix_ds, axis=0)

    def sem_npix(self) -> np.ndarray:
        return np.nanstd(self.npix_ds, axis=0) / np.sqrt(self.npix_ds.shape[0])

    def zF_in_zone(self, neuron_zone: np.ndarray, zone_id) -> np.ndarray:
        """zF restricted to neurons whose `neuron_zone` label equals zone_id -> (n_trials_cond, n_time, n_neurons_in_zone)."""
        mask = np.asarray(neuron_zone) == zone_id
        return self.zF[:, :, mask]

    def __repr__(self):
        return (f"ConditionData('{self.name}', n_trials={len(self.trial_idx)}, "
                f"zF={self.zF.shape}, npix_ds={self.npix_ds.shape})")


@dataclass
class organisedDataset:
    conditions: dict                 # name -> ConditionData
    time_zF: np.ndarray               # (n_time,) shared relative time axis (seconds)
    time_npix_ds: np.ndarray          # same as time_zF after downsampling (kept for clarity)
    seshParams: dict

    # per-neuron label (e.g. microzone identity), aligned to zF's feature/neuron
    # axis -- NOT per-trial. length should equal zF.shape[-1] (n_neurons).
    neuron_zone: Optional[np.ndarray] = field(default=None, repr=False)

    # populated only if pynapple is used (see build_pynapple_view)
    tsdtensor_zF: Optional["nap.TsdTensor"] = field(default=None, repr=False)
    tsdtensor_npix: Optional["nap.TsdTensor"] = field(default=None, repr=False)
    condition_intervals: Optional[dict] = field(default=None, repr=False)

    def __getitem__(self, name):
        return self.conditions[name]

    def __iter__(self):
        return iter(self.conditions.values())

    def keys(self):
        return self.conditions.keys()

    def zone_ids(self):
        return None if self.neuron_zone is None else np.unique(self.neuron_zone)

    def zF_by_zone(self, cond_name: str, zone_id) -> np.ndarray:
        """zF for one condition, restricted to neurons in `zone_id` (see neuron_zone)."""
        if self.neuron_zone is None:
            raise ValueError("No neuron_zone available (iZone length didn't match n_neurons in zF).")
        return self.conditions[cond_name].zF_in_zone(self.neuron_zone, zone_id)

    def mean_zF_by_zone(self, cond_name: str, zone_id) -> np.ndarray:
        return np.nanmean(self.zF_by_zone(cond_name, zone_id), axis=0)

    def pool(self, cond_names: list, name: Optional[str] = None) -> ConditionData:
        """
        Concatenate trials across several named conditions into one ConditionData,
        e.g. to combine all "actually went Left, rewarded" trials regardless of
        whether the cue was congruent (iCueL_1port_cor, iCueL_2port_cor) or
        incongruent/mixed (iCueLR_Lport_cor) -- useful for finding neurons with a
        general directional response rather than being confined to the 10 raw
        trialTypeInds buckets. Conditions with 0 trials are skipped automatically.
        """
        present = [c for c in cond_names if c in self.conditions and len(self.conditions[c].trial_idx) > 0]
        if not present:
            raise ValueError(f"None of {cond_names} exist with trials in this dataset. "
                              f"Available: {list(self.conditions.keys())}")
        missing = set(cond_names) - set(present)
        if missing:
            print(f"[organise_touch_trials] pool(): skipping empty/missing conditions {sorted(missing)}")

        conds = [self.conditions[c] for c in present]
        return ConditionData(
            name=name or "+".join(present),
            trial_idx=np.concatenate([c.trial_idx for c in conds]),
            zF=np.concatenate([c.zF for c in conds], axis=0),
            npix_ds=np.concatenate([c.npix_ds for c in conds], axis=0),
            cueSide=np.concatenate([c.cueSide for c in conds]),
            portSide=np.concatenate([c.portSide for c in conds]),
            rewardSide=np.concatenate([c.rewardSide for c in conds]),
            trialOnT=np.concatenate([c.trialOnT for c in conds]),
        )

    def summary(self):
        for name, cond in self.conditions.items():
            print(f"{name:20s} n_trials={len(cond.trial_idx):4d}  zF={cond.zF.shape}  npix_ds={cond.npix_ds.shape}")
        if self.neuron_zone is not None:
            zones, counts = np.unique(self.neuron_zone, return_counts=True)
            print(f"neuron_zone: {dict(zip(zones.tolist(), counts.tolist()))}")
        else:
            print("neuron_zone: not available")


def organise_dataset(touchDict: dict, window: Optional[tuple] = None,
                      zF_key: str = 'zF_touchMat', npix_key: str = 'npix_touchMat',
                      zF_n_time: Optional[int] = 120, npix_n_time: Optional[int] = 400,
                      zF_time_axis: Optional[int] = None,
                      npix_time_axis: Optional[int] = None) -> organisedDataset:
    """
    Main entry point. Slices zF_touchMat/npix_touchMat by touchDict['trialTypeInds'],
    downsamples npix onto zF's time grid, and returns an organisedDataset you can
    index by condition name, e.g. organised['iCueL_1port_cor'].mean_zF().

    zF and npix don't need matching feature counts (e.g. neurons vs. touch
    pixels) -- their time axes are identified by expected sample count
    (zF_n_time/npix_n_time, default 120/400) or you can force it directly with
    zF_time_axis/npix_time_axis (1 or 2) if auto-detection picks the wrong one.
    """
    sp = touchDict['seshParams']
    trialOnT = np.asarray(sp['trialOnT'])
    n_trials = len(trialOnT)

    cueSide = np.asarray(sp['cueSide'])
    portSide = np.asarray(sp['portSide'])
    rewardSide = np.asarray(sp['rewardSide'])

    zF = np.asarray(touchDict[zF_key])
    npixMat = np.asarray(touchDict[npix_key])

    # canonicalize both to (n_trials, n_time, n_features) regardless of raw
    # axis order (e.g. handles zF_touchMat stored as (n_trials, n_neurons, n_time))
    zF, npixMat = orient_trial_time_feature(
        zF, npixMat, n_trials,
        zF_n_time=zF_n_time, npix_n_time=npix_n_time,
        zF_time_axis=zF_time_axis, npix_time_axis=npix_time_axis,
    )

    n_time_zF = zF.shape[1]

    time_zF = build_time_axis(sp, n_time_zF, window=window) # should span -2 to 2 seconds in 120 steps

    # downsample npix's time axis (now axis=1 after canonicalization) onto n_time_zF samples
    npix_ds = downsample_along_axis(npixMat, axis=1, target_n=n_time_zF)
    time_npix_ds = time_zF  # same grid post-downsample

    conditions = {}
    for cond_name, idx in touchDict['trialTypeInds'].items():
        idx = np.asarray(idx, dtype=int)
        conditions[cond_name] = ConditionData(
            name=cond_name,
            trial_idx=idx,
            zF=zF[idx],
            npix_ds=npix_ds[idx],
            cueSide=safe_index_meta(cueSide, idx, n_trials, 'cueSide'),
            portSide=safe_index_meta(portSide, idx, n_trials, 'portSide'),
            rewardSide=safe_index_meta(rewardSide, idx, n_trials, 'rewardSide'),
            trialOnT=safe_index_meta(trialOnT, idx, n_trials, 'trialOnT'),
        )

    # iZone is a per-NEURON label (e.g. microzone identity), aligned to zF's
    # neuron/feature axis -- not per-trial. Only attach it if its length
    # actually matches n_neurons; otherwise leave it out rather than
    # mis-slice it as if it were per-trial.
    n_neurons = zF.shape[-1]
    neuron_zone = None
    if 'iZone' in touchDict:
        iZone = np.asarray(touchDict['iZone'])
        if len(iZone) == n_neurons:
            neuron_zone = iZone
        else:
            print(f"[organise_touch_trials] WARNING: 'iZone' has length {len(iZone)}, "
                  f"which matches neither n_trials ({n_trials}) nor n_neurons in zF "
                  f"({n_neurons}). Leaving organisedDataset.neuron_zone unset -- check "
                  "touchDict['iZone'] manually.")

    return organisedDataset(
        conditions=conditions,
        time_zF=time_zF,
        time_npix_ds=time_npix_ds,
        seshParams=sp,
        neuron_zone=neuron_zone,
    )

