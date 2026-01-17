from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from rna_backbone_design.analysis import metrics


def _align_to_reference(
    coords: torch.Tensor, mask: torch.Tensor, ref: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if coords.ndim != 3:
        raise ValueError(f"coords must be [K, L, 3], got {tuple(coords.shape)}")
    if mask.ndim != 1:
        raise ValueError(f"mask must be [L], got {tuple(mask.shape)}")
    if ref is None:
        ref = coords[0]
    fixed = ref[None, ...].expand(coords.shape[0], -1, -1)
    m = mask[None, ...].expand(coords.shape[0], -1)
    try:
        return metrics.superimpose(fixed, coords, mask=m)
    except RuntimeError:
        return coords


def _pairwise_rmsd_matrix(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 3:
        raise ValueError(f"coords must be [K, L, 3], got {tuple(coords.shape)}")
    if mask.ndim != 1:
        raise ValueError(f"mask must be [L], got {tuple(mask.shape)}")
    k = coords.shape[0]
    m = mask.to(coords.device).float()
    num = m.sum().clamp_min(1.0)
    diffs = coords[:, None, :, :] - coords[None, :, :, :]
    sq = (diffs * diffs).sum(dim=-1) * m[None, None, :]
    rmsd = (sq.sum(dim=-1) / num).sqrt()
    rmsd[torch.arange(k, device=coords.device), torch.arange(k, device=coords.device)] = 0.0
    return rmsd


def pairwise_rmsd_mean(coords: torch.Tensor, mask: torch.Tensor) -> float:
    k = coords.shape[0]
    if k < 2:
        return float("nan")
    mat = _pairwise_rmsd_matrix(coords, mask)
    triu = torch.triu_indices(k, k, offset=1, device=coords.device)
    return float(mat[triu[0], triu[1]].mean().detach().cpu().numpy())


def wasserstein_distance_equal_weight(
    distmat: np.ndarray, p: int = 2
) -> float:
    if distmat.ndim != 2:
        raise ValueError(f"distmat must be 2D, got shape {distmat.shape}")
    if distmat.shape[0] != distmat.shape[1]:
        raise ValueError(f"distmat must be square, got shape {distmat.shape}")
    from scipy.optimize import linear_sum_assignment

    cost = distmat.astype(np.float64) ** p
    row_ind, col_ind = linear_sum_assignment(cost)
    return float((cost[row_ind, col_ind].mean()) ** (1.0 / p))


def pairwise_rmsd_pearson_r(gt_coords: torch.Tensor, pred_coords: torch.Tensor, mask: torch.Tensor) -> float:
    if gt_coords.shape != pred_coords.shape:
        raise ValueError(
            f"gt_coords and pred_coords must have same shape, got {tuple(gt_coords.shape)} vs {tuple(pred_coords.shape)}"
        )
    k = gt_coords.shape[0]
    if k < 2:
        return float("nan")
    gt_mat = _pairwise_rmsd_matrix(gt_coords, mask)
    pred_mat = _pairwise_rmsd_matrix(pred_coords, mask)
    triu = torch.triu_indices(k, k, offset=1, device=gt_coords.device)
    x = gt_mat[triu[0], triu[1]]
    y = pred_mat[triu[0], triu[1]]
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if float(denom.detach().cpu().numpy()) == 0.0:
        return float("nan")
    r = (x * y).mean() / denom
    return float(r.detach().cpu().numpy())


@dataclass(frozen=True)
class EnsembleMetrics:
    pairwise_rmsd: float
    w2_distance: float
    pairwise_rmsd_r: float


def compute_ensemble_metrics(
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    mask: torch.Tensor,
) -> EnsembleMetrics:
    if gt_coords.ndim != 3 or pred_coords.ndim != 3:
        raise ValueError("gt_coords and pred_coords must be [K, L, 3]")
    if mask.ndim != 1:
        raise ValueError("mask must be [L]")
    if mask.sum().item() < 3:
        return EnsembleMetrics(float("nan"), float("nan"), float("nan"))

    gt_aligned = _align_to_reference(gt_coords, mask)
    pred_aligned = _align_to_reference(pred_coords, mask, ref=gt_aligned[0])

    pairwise_rmsd = pairwise_rmsd_mean(pred_aligned, mask)

    distmat = _pairwise_rmsd_matrix(
        torch.cat([gt_aligned, pred_aligned], dim=0), mask
    ).detach().cpu().numpy()
    k = gt_aligned.shape[0]
    cross = distmat[:k, k:]
    w2 = wasserstein_distance_equal_weight(cross.T, p=2)

    pairwise_r = pairwise_rmsd_pearson_r(gt_aligned, pred_aligned, mask)
    return EnsembleMetrics(pairwise_rmsd=pairwise_rmsd, w2_distance=w2, pairwise_rmsd_r=pairwise_r)
