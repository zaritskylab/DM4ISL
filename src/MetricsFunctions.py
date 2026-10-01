"""
Image-similarity and 2D (slice-level) instance-segmentation metrics between GT fluorescence and virtual-staining predictions.

Input stacks are (N, Z, Y, X): N patches, each a stack of Z slices. Every slice is segmented and matched in 2D.
Per slice, each GT instance gets its metrics (NaN if it has no match with IoU >= iou_threshold) and the slice value
is the nanmean over its instances. The patch value is the nanmean over its slices, so every metric returns one value
per patch. Instance ids are per slice (instances are not linked across z).

All segmentation knobs live in SegParams. Override any of them with a dict, e.g.
    calc_metrics(gt, pred, {'min_size': 40, 'iou_threshold': 0.5, 'exclude_edge_slivers': True})
"""
from dataclasses import dataclass, fields, replace
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import relabel_sequential, watershed

#########################################################
# Parameters
#########################################################

@dataclass(frozen=True)
class SegParams:
    pixel_size_um: float = 0.29          # isotropic pixel size (y, x) in microns
    # --- binary segmentation ---
    smooth_sigma: float = 1              # 2D Gaussian sigma (pixels, y and x only) applied before Otsu; 0 disables
    threshold_mode: str = 'independent'  # 'independent': Otsu per image | 'gt': GT's Otsu threshold for both images
    threshold_scope: str = 'slice'       # 'slice': Otsu per 2D slice | 'patch': one Otsu over the 3D patch, used for all its slices
    opening_radius: int = 1              # disk radius (pixels) of the binary opening; 0 disables
    min_size: int = 64*64*0.01           # pixels; smaller components are removed and smaller watershed instances merged
    # --- instance segmentation ---
    peak_min_distance: int = 10          # minimal distance (pixels) between watershed seeds
    edt_border_axes: tuple = ()          # slice axes (0=y, 1=x) on which the border counts as background for the EDT
    # --- matching ---
    iou_threshold: float = 0.6          # minimal 2D IoU for a GT/pred pair to count as a true positive
    # --- nearest-neighbour distance ---
    nn_mode: str = 'surface'             # 'centroid': to the nearest other centroid | 'surface': contour gap (touching = 0)
    # --- edge slivers ---
    exclude_edge_slivers: bool = False   # ignore small instances cut by the slice border (not counted as TP/FP/FN)
    edge_sliver_size: int = 64*64*0.03   # pixels; edge instances below this area are slivers
    edge_axes: tuple = (0, 1)            # slice axes (0=y, 1=x) whose borders define "edge"
    # --- reporting ---
    area_error: str = 'abs'              # 'abs': |dA| um^2 | 'rel': |dA| / A_gt | 'signed_rel': dA / A_gt
    hd_percentile: float = 100.0         # 100: Hausdorff distance | 95: HD95

def resolve_params(params=None) -> SegParams:
    """Accepts None, a SegParams or a dict of overrides."""
    if params is None:
        return SegParams()
    if isinstance(params, SegParams):
        return params
    unknown = set(params) - {f.name for f in fields(SegParams)}
    if unknown:
        raise TypeError(f'Unknown segmentation parameters: {sorted(unknown)}')
    return replace(SegParams(), **params)

def _nanmean(values) -> float:
    """Mean over finite values; NaN (without a warning) when there are none."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else np.nan

#########################################################
# Segmentation (2D)
#########################################################

def smooth(image: np.ndarray, p: SegParams) -> np.ndarray:
    """2D Gaussian on the last two axes; works on a single slice (Y, X) or a stack (Z, Y, X)."""
    image = image.astype(np.float64)
    if p.smooth_sigma <= 0:
        return image
    sigma = (0,) * (image.ndim - 2) + (p.smooth_sigma, p.smooth_sigma)
    return gaussian(image, sigma=sigma, preserve_range=True)

def otsu_threshold(image: np.ndarray) -> float:
    return threshold_otsu(image) if image.max() > image.min() else np.inf

def thresholds(gt_s: np.ndarray, pred_s: np.ndarray, p: SegParams):
    """(gt_threshold, pred_threshold) according to threshold_mode."""
    gt_t = otsu_threshold(gt_s)
    if p.threshold_mode == 'gt':
        return gt_t, gt_t
    if p.threshold_mode == 'independent':
        return gt_t, otsu_threshold(pred_s)
    raise ValueError(f'Unknown threshold_mode: {p.threshold_mode}')

def binary_segmentation(image: np.ndarray, threshold: float, p: SegParams) -> np.ndarray:
    """Threshold -> hole filling -> opening -> removal of components below min_size."""
    mask = ndimage.binary_fill_holes(image > threshold)
    if p.opening_radius > 0:
        mask = ndimage.binary_opening(mask, structure=disk(p.opening_radius))
    return remove_small_objects(mask, min_size=p.min_size)

def distance_transform(mask: np.ndarray, border_axes: tuple = ()) -> np.ndarray:
    """EDT to background; on border_axes the space beyond the slice is treated as background."""
    if not border_axes:
        return ndimage.distance_transform_edt(mask)
    pad = [(1, 1) if a in border_axes else (0, 0) for a in range(mask.ndim)]
    crop = tuple(slice(1, -1) if a in border_axes else slice(None) for a in range(mask.ndim))
    return ndimage.distance_transform_edt(np.pad(mask, pad))[crop]

def _seed_unseeded_components(markers: np.ndarray, components: np.ndarray, n_components: int) -> np.ndarray:
    """Every connected component gets at least one seed, otherwise watershed leaves it unlabeled."""
    seeded = np.unique(components[markers > 0])
    next_id = markers.max() + 1
    for c in np.setdiff1d(np.arange(1, n_components + 1), seeded):
        markers[components == c] = next_id
        next_id += 1
    return markers

def instance_segmentation(mask: np.ndarray, p: SegParams) -> np.ndarray:
    """Distance-transform watershed; instances below min_size are merged into their neighbours."""
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)
    distance = distance_transform(mask, p.edt_border_axes)
    components, n_components = ndimage.label(mask)
    peaks = peak_local_max(distance, min_distance=p.peak_min_distance, labels=components, exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    markers = _seed_unseeded_components(markers, components, n_components)
    labels = watershed(-distance, markers, mask=mask)

    # Drop the seeds of too-small instances and re-flood, so over-split fragments merge into a neighbour
    small = np.flatnonzero(np.bincount(labels.ravel()) < p.min_size)
    small = small[small > 0]
    if len(small):
        markers[np.isin(markers, small)] = 0
        markers = _seed_unseeded_components(markers, components, n_components)
        labels = watershed(-distance, markers, mask=mask)
    return relabel_sequential(labels)[0].astype(np.int32)

#########################################################
# Instance properties and matching
#########################################################

def instance_properties(labels: np.ndarray, p: SegParams) -> pd.DataFrame:
    """Per-instance area, centroid, edge contact and sliver flag, indexed by label id (1..n)."""
    ids = np.arange(1, labels.max() + 1)
    cols = ['area_px', 'cy', 'cx', 'edge', 'sliver']
    if len(ids) == 0:
        return pd.DataFrame(columns=cols)
    area = np.bincount(labels.ravel())[1:]
    centroids = np.array(ndimage.center_of_mass(np.ones_like(labels), labels, ids)).reshape(-1, 2)
    slices = ndimage.find_objects(labels)
    edge = np.array([any(sl[a].start == 0 or sl[a].stop == labels.shape[a] for a in p.edge_axes) for sl in slices])
    sliver = p.exclude_edge_slivers & edge & (area < p.edge_sliver_size)
    return pd.DataFrame({'area_px': area, 'cy': centroids[:, 0], 'cx': centroids[:, 1],
                         'edge': edge, 'sliver': sliver}, index=ids)

def iou_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """(n_gt, n_pred) IoU matrix from a single label co-occurrence count. Labels must be sequential."""
    n_gt, n_pred = gt_labels.max() + 1, pred_labels.max() + 1
    counts = np.bincount(gt_labels.ravel().astype(np.int64) * n_pred + pred_labels.ravel(),
                         minlength=n_gt * n_pred).reshape(n_gt, n_pred)
    inter = counts[1:, 1:]
    union = counts.sum(axis=1)[1:, None] + counts.sum(axis=0)[None, 1:] - inter
    return np.divide(inter, union, out=np.zeros(inter.shape), where=union > 0)

def match_instances(gt_labels: np.ndarray, pred_labels: np.ndarray, iou_threshold: float):
    """Hungarian matching maximizing IoU. Returns [(gt_id, pred_id, iou), ...] with iou >= threshold."""
    iou = iou_matrix(gt_labels, pred_labels)
    if iou.size == 0:
        return []
    iou = np.where(iou >= iou_threshold, iou, 0.0)  # sub-threshold overlaps must not steal an assignment
    rows, cols = linear_sum_assignment(-iou)
    return [(r + 1, c + 1, iou[r, c]) for r, c in zip(rows, cols) if iou[r, c] >= iou_threshold]

#########################################################
# Per-instance distance metrics
#########################################################

def nearest_neighbor_distances(labels: np.ndarray, props: pd.DataFrame, ids, p: SegParams) -> dict:
    """
    Distance (um) from each instance in ids to its nearest neighbour in the same slice; NaN if it has none.
    Slivers are not neighbours, consistent with their exclusion from matching.
    """
    neighbours = props.index[~props['sliver'].astype(bool)]
    out = {}
    for i in ids:
        others = neighbours[neighbours != i]
        if len(others) == 0:
            out[i] = np.nan
        elif p.nn_mode == 'centroid':
            c = props[['cy', 'cx']].to_numpy(float)
            out[i] = np.linalg.norm(c[others - 1] - c[i - 1], axis=1).min() * p.pixel_size_um
        elif p.nn_mode == 'surface':
            d = ndimage.distance_transform_edt(~np.isin(labels, others))[labels == i].min()
            out[i] = (d - 1) * p.pixel_size_um  # adjacent pixels are 1 apart -> touching instances have gap 0
        else:
            raise ValueError(f'Unknown nn_mode: {p.nn_mode}')
    return out

def _contour(mask: np.ndarray) -> np.ndarray:
    # border_value=1: pixels on the slice border are crop artefacts, not object contour
    return mask & ~ndimage.binary_erosion(mask, border_value=1)

def hausdorff_distance(a: np.ndarray, b: np.ndarray, pixel_size_um: float, percentile: float = 100.0) -> float:
    """Symmetric contour Hausdorff distance (um) between two 2D masks; percentile < 100 gives e.g. HD95."""
    ca, cb = _contour(a), _contour(b)
    if not ca.any() or not cb.any():
        return np.nan
    d = np.concatenate([ndimage.distance_transform_edt(~cb)[ca], ndimage.distance_transform_edt(~ca)[cb]])
    return (d.max() if percentile >= 100 else np.percentile(d, percentile)) * pixel_size_um

#########################################################
# Slice and patch evaluation
#########################################################

INSTANCE_COLUMNS = ['gt_id', 'matched', 'pred_id', 'iou', 'gt_edge', 'pred_edge', 'gt_area_um2', 'pred_area_um2',
                    'area_diff_um2', 'area_error', 'dy_um', 'dx_um', 'gt_nn_dist_um', 'pred_nn_dist_um', 'hausdorff_um']
COUNT_COLUMNS = ['n_gt', 'n_pred', 'tp', 'fp', 'fn']
MEAN_COLUMNS = ['precision', 'recall', 'f1_score', 'binary_iou', 'mean_iou', 'mean_centroid_dist_um',
                'mean_nn_dist_error_um', 'mean_area_error', 'mean_hausdorff_um']

def _area_error(a_gt, a_pred, mode):
    if mode == 'abs':
        return abs(a_pred - a_gt)
    if mode == 'rel':
        return abs(a_pred - a_gt) / a_gt
    if mode == 'signed_rel':
        return (a_pred - a_gt) / a_gt
    raise ValueError(f'Unknown area_error mode: {mode}')

def _evaluate_slice(gt_s: np.ndarray, pred_s: np.ndarray, gt_t: float, pred_t: float, p: SegParams) -> dict:
    """
    Segments and matches one pair of smoothed 2D slices.
    'instances' has one row per (non-sliver) GT instance; unmatched ones have NaN metrics.
    Edge slivers (if excluded) are neither TP, FP nor FN; a pred matched to a GT sliver is ignored too.
    """
    ps = p.pixel_size_um
    gt_mask, pred_mask = binary_segmentation(gt_s, gt_t, p), binary_segmentation(pred_s, pred_t, p)
    gt_labels, pred_labels = instance_segmentation(gt_mask, p), instance_segmentation(pred_mask, p)
    gt_props, pred_props = instance_properties(gt_labels, p), instance_properties(pred_labels, p)
    gt_sliver, pred_sliver = gt_props['sliver'].astype(bool), pred_props['sliver'].astype(bool)

    union = np.logical_or(gt_mask, pred_mask).sum()
    binary_iou = np.logical_and(gt_mask, pred_mask).sum() / union if union > 0 else np.nan

    all_matches = match_instances(gt_labels, pred_labels, p.iou_threshold)
    any_matched_gt, any_matched_pred = [m[0] for m in all_matches], [m[1] for m in all_matches]
    matches = [m for m in all_matches if not gt_props.at[m[0], 'sliver']]
    match_of = {g: (q, iou) for g, q, iou in matches}
    tp = len(matches)
    fn = int((~gt_sliver & ~gt_props.index.isin(any_matched_gt)).sum())
    fp = int((~pred_sliver & ~pred_props.index.isin(any_matched_pred)).sum())

    gt_nn = nearest_neighbor_distances(gt_labels, gt_props, list(match_of), p)
    pred_nn = nearest_neighbor_distances(pred_labels, pred_props, [q for q, _ in match_of.values()], p)
    rows = []
    for g in gt_props.index[~gt_sliver]:
        gp = gt_props.loc[g]
        a_gt = gp.area_px * ps ** 2
        row = {'gt_id': g, 'matched': g in match_of, 'gt_edge': bool(gp.edge), 'gt_area_um2': a_gt}
        if g in match_of:  # unmatched GT instances keep NaN in all metric columns
            q, iou = match_of[g]
            pp = pred_props.loc[q]
            a_pred = pp.area_px * ps ** 2
            row.update({
                'pred_id': q, 'iou': iou, 'pred_edge': bool(pp.edge), 'pred_area_um2': a_pred,
                'area_diff_um2': a_pred - a_gt, 'area_error': _area_error(a_gt, a_pred, p.area_error),
                'dy_um': (pp.cy - gp.cy) * ps, 'dx_um': (pp.cx - gp.cx) * ps,
                'gt_nn_dist_um': gt_nn[g], 'pred_nn_dist_um': pred_nn[q],
                'hausdorff_um': hausdorff_distance(gt_labels == g, pred_labels == q, ps, p.hd_percentile),
            })
        rows.append(row)
    inst = pd.DataFrame(rows, columns=INSTANCE_COLUMNS)  # missing keys -> NaN
    inst['centroid_dist_um'] = np.sqrt(inst.dy_um.astype(float) ** 2 + inst.dx_um.astype(float) ** 2)
    inst['nn_dist_error_um'] = inst.pred_nn_dist_um.astype(float) - inst.gt_nn_dist_um.astype(float)
    inst['abs_nn_dist_error_um'] = inst.nn_dist_error_um.abs()

    summary = {
        'n_gt': int((~gt_sliver).sum()), 'n_pred': int((~pred_sliver).sum()), 'tp': tp, 'fp': fp, 'fn': fn,
        'precision': tp / (tp + fp) if tp + fp > 0 else np.nan,
        'recall': tp / (tp + fn) if tp + fn > 0 else np.nan,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else np.nan,
        'binary_iou': binary_iou,
        # nanmean over the slice's instances: unmatched instances (NaN) are excluded
        'mean_iou': _nanmean(inst['iou']),
        'mean_centroid_dist_um': _nanmean(inst['centroid_dist_um']),
        'mean_nn_dist_error_um': _nanmean(inst['abs_nn_dist_error_um']),
        'mean_area_error': _nanmean(inst['area_error']),
        'mean_hausdorff_um': _nanmean(inst['hausdorff_um']),
    }
    return {'summary': summary, 'instances': inst, 'gt_labels': gt_labels, 'pred_labels': pred_labels}

def evaluate_patch(gt_img: np.ndarray, pred_img: np.ndarray, params=None) -> dict:
    """
    Evaluates every z-slice of a (Z, Y, X) patch in 2D.
    Returns {'summary': patch-level dict (nanmean over slices; counts summed),
             'slices': per-slice DataFrame, 'instances': per-instance DataFrame with a 'slice' column,
             'gt_labels', 'pred_labels': (Z, Y, X) stacks of per-slice 2D labels}.
    """
    p = resolve_params(params)
    gt_s, pred_s = smooth(gt_img, p), smooth(pred_img, p)
    if p.threshold_scope == 'patch':
        gt_t, pred_t = thresholds(gt_s, pred_s, p)
    elif p.threshold_scope != 'slice':
        raise ValueError(f'Unknown threshold_scope: {p.threshold_scope}')

    slice_summaries, instances, gt_labels, pred_labels = [], [], [], []
    for z in range(gt_s.shape[0]):
        if p.threshold_scope == 'slice':
            gt_t, pred_t = thresholds(gt_s[z], pred_s[z], p)
        res = _evaluate_slice(gt_s[z], pred_s[z], gt_t, pred_t, p)
        slice_summaries.append({'slice': z, **res['summary']})
        instances.append(res['instances'].assign(slice=z))
        gt_labels.append(res['gt_labels'])
        pred_labels.append(res['pred_labels'])

    slices = pd.DataFrame(slice_summaries)
    summary = {c: int(slices[c].sum()) for c in COUNT_COLUMNS}
    summary.update({c: _nanmean(slices[c]) for c in MEAN_COLUMNS})  # nanmean over slices
    tp, fp, fn = summary['tp'], summary['fp'], summary['fn']
    summary['f1_pooled'] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else np.nan  # from counts summed over slices
    return {'summary': summary, 'slices': slices, 'instances': pd.concat(instances, ignore_index=True),
            'gt_labels': np.stack(gt_labels), 'pred_labels': np.stack(pred_labels)}

def segmentation_metrics_table(FL_images, ISL_images, params=None):
    """Evaluates all patches. Returns (per-patch DataFrame, per-instance DataFrame), both with a 'patch' column."""
    p = resolve_params(params)
    summaries, instances = [], []
    for i in range(len(FL_images)):
        res = evaluate_patch(FL_images[i], ISL_images[i], p)
        summaries.append({'patch': i, **res['summary']})
        if not res['instances'].empty:
            instances.append(res['instances'].assign(patch=i))
    return pd.DataFrame(summaries), (pd.concat(instances, ignore_index=True) if instances else pd.DataFrame())

def seg_watershed_iou_metrics_func(FL_images, ISL_images, params=None):
    """One value per patch (NaN where undefined, e.g. no matched instance in any slice)."""
    df, _ = segmentation_metrics_table(FL_images, ISL_images, params)
    cols = ['binary_iou', 'f1_score', 'mean_centroid_dist_um', 'mean_nn_dist_error_um', 'mean_area_error',
            'mean_hausdorff_um']
    return tuple(df[c].tolist() for c in cols)


#########################################################
# Image-similarity metrics
#########################################################

_MS_SSIM = None

def jsd(imgs1, imgs2, rn=3, bins=256, value_range=(0.0, 1.0)):
    # JSD between intensity histograms on shared fixed bins (compares intensity distributions, not spatial layout)
    h1, _ = np.histogram(imgs1, bins=bins, range=value_range)
    h2, _ = np.histogram(imgs2, bins=bins, range=value_range)
    return np.round(jensenshannon(h1, h2, base=2), rn)

def mi(imgs1, imgs2, rn=3, bins=20):
    eps = 1e-12
    hgram, _, _ = np.histogram2d(imgs1.ravel(), imgs2.ravel(), bins=bins)
    pxy = hgram / hgram.sum()
    px, py = pxy.sum(axis=1), pxy.sum(axis=0)
    hx = -np.sum(px * np.log2(px + eps))
    hy = -np.sum(py * np.log2(py + eps))
    hxy = -np.sum(pxy * np.log2(pxy + eps))
    return np.round(hx + hy - hxy, rn)

def ms_ssim(imgs1, imgs2, rn=3):
    global _MS_SSIM
    if _MS_SSIM is None:
        from generative.metrics import MultiScaleSSIMMetric
        _MS_SSIM = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
    return np.round(_MS_SSIM(torch.tensor(imgs1), torch.tensor(imgs2)).mean().numpy(), rn)

def pcc(imgs1, imgs2, rn=3):
    return np.round(stats.pearsonr(imgs1.ravel(), imgs2.ravel())[0], rn)

def mse(imgs1, imgs2, rn=3):
    return np.round(np.mean((imgs1.ravel() - imgs2.ravel()) ** 2), rn)

#########################################################
# Entry point
#########################################################

def calc_metrics(all_GT_images__, all_Model_images__, organelle_seg_params, rn=4):
    """organelle_seg_params: None, a SegParams or a dict overriding SegParams defaults."""
    pccs, mses, msssims, jsds, mis = [], [], [], [], []
    for i in range(len(all_GT_images__)):
        pccs.append(pcc(all_GT_images__[i], all_Model_images__[i], rn))
        mses.append(mse(all_GT_images__[i], all_Model_images__[i], rn))
        msssims.append(ms_ssim(all_GT_images__[i:i + 1], all_Model_images__[i:i + 1], rn))
        jsds.append(jsd(all_GT_images__[i], all_Model_images__[i], rn))
        mis.append(mi(all_GT_images__[i], all_Model_images__[i], rn))
    pccs, mses, msssims, jsds, mis = map(np.array, (pccs, mses, msssims, jsds, mis))

    ious, f1s, centroid_errors, mindist_errors, area_errors, hds = seg_watershed_iou_metrics_func(all_GT_images__, all_Model_images__, organelle_seg_params)  # each of length N
    return pccs, mses, msssims, jsds, mis, ious, f1s, centroid_errors, mindist_errors, area_errors, hds