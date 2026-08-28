from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr

def _lorentz_score_only(X, D_wish, mask_if, c):
    iu, ju = np.where(np.triu(mask_if, 1))
    if iu.size == 0:
        return 0.0
    diff = X[iu] - X[ju]
    dist = np.linalg.norm(diff, axis=1) + 1e-8
    t = dist - D_wish[iu, ju]
    c2 = c * c
    return float(np.sum(c2 / (c2 + t * t)))

def seed_gaps_midpoints(X_init: np.ndarray, is_gap: np.ndarray) -> np.ndarray:
    n = len(X_init)
    left_near = np.full(n, -1, dtype=int)
    right_near = np.full(n, -1, dtype=int)

    last = -1
    for i in range(n):
        if not is_gap[i]:
            last = i
        left_near[i] = last

    last = -1
    for i in range(n - 1, -1, -1):
        if not is_gap[i]:
            last = i
        right_near[i] = last

    for i in range(n):
        if is_gap[i]:
            l = left_near[i]; r = right_near[i]
            if l >= 0 and r >= 0:
                X_init[i] = 0.5 * (X_init[l] + X_init[r])
            elif l >= 0:
                X_init[i] = X_init[l].copy()
            elif r >= 0:
                X_init[i] = X_init[r].copy()
            else:
                X_init[i] = np.zeros(3, dtype=float)
    return X_init

def lorentz_objective_and_grad(
    X: np.ndarray, D_wish: np.ndarray, mask_if: np.ndarray, c: float,
    continuity: Optional[Dict[str, Tuple[int, np.ndarray, float]]] = None,
    freeze_mask: Optional[np.ndarray] = None,
    pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[float, np.ndarray]:
    g = np.zeros_like(X)
    f = 0.0
    eps = 1e-8
    c2 = c * c

    if pairs is None:
        iu, ju = np.where(np.triu(mask_if, 1))
    else:
        iu, ju = pairs
    if iu.size:
        diff = X[iu] - X[ju]
        dist = np.linalg.norm(diff, axis=1) + eps
        t = dist - D_wish[iu, ju]
        denom = (c2 + t * t)
        val = c2 / denom
        f += float(np.sum(val))

        coeff = (-2.0 * c2 * t) / (denom * denom * dist)
        gi = coeff[:, None] * diff
        np.add.at(g, iu, gi)
        np.add.at(g, ju, -gi)

    if continuity:
        if "left" in continuity and continuity["left"] is not None:
            idx, anchor, lam = continuity["left"]
            dvec = X[idx] - anchor
            f -= lam * float(np.dot(dvec, dvec))
            g[idx] -= 2.0 * lam * dvec
        if "right" in continuity and continuity["right"] is not None:
            idx, anchor, lam = continuity["right"]
            dvec = X[idx] - anchor
            f -= lam * float(np.dot(dvec, dvec))
            g[idx] -= 2.0 * lam * dvec

    if freeze_mask is not None:
        g[freeze_mask] = 0.0

    return f, g

def line_search_alpha_seeded(
    abs_bins, hic3col, alpha_grid, X_seed, c_policy="mean"
):
    from mb_gap_utils import build_wish_distance_matrix
    m = len(abs_bins)
    if m < 3 or X_seed.shape[0] != m:
        return 0.3, None, None

    best_alpha = None
    best_score = -np.inf
    best_D = None
    best_M = None

    for a in alpha_grid:
        D, M = build_wish_distance_matrix(abs_bins, hic3col, alpha=a, c=1.0)
        sel = (D > 0) & M
        if np.count_nonzero(sel) < 3:
            continue

        if c_policy == "mean":
            c = float(np.mean(D[sel]))
        elif c_policy == "median":
            c = float(np.median(D[sel]))
        else:
            c = 1.0

        score = _lorentz_score_only(X_seed, D, M, c)
        if (score > best_score + 1e-9) or (abs(score - best_score) <= 1e-9 and best_alpha is not None and abs(a - np.median(alpha_grid)) < abs(best_alpha - np.median(alpha_grid))):
            best_score = score
            best_alpha = float(a)
            best_D = D
            best_M = M

    if best_alpha is None:
        return 0.3, None, None
    return best_alpha, best_D, best_M

def optimize_gaps(
    X_init: np.ndarray,
    is_gap_mask: np.ndarray,
    D_wish: np.ndarray,
    mask_if: np.ndarray,
    max_iter: int = 1500,
    step: float = 0.05,
    c: Optional[float] = None,
    anchor_w: float = 1e-3,
    continuity: Optional[Dict[str, Tuple[int, np.ndarray, float]]] = None,
):
    X = X_init.copy()
    if c is None:
        sel = (D_wish > 0) & mask_if
        c = float(np.mean(D_wish[sel])) if np.count_nonzero(sel) else 1.0

    X_seed = X_init.copy()
    freeze_mask = ~is_gap_mask

    best_f = -np.inf
    no_improve = 0
    hist = {"f": []}
    patience = 10 if np.count_nonzero(is_gap_mask) <= 10 else 20

    for _ in range(max_iter):
        f, g = lorentz_objective_and_grad(X, D_wish, mask_if, c,
                                          continuity=continuity,
                                          freeze_mask=freeze_mask)
        if anchor_w > 0:
            g[is_gap_mask] += anchor_w * (X[is_gap_mask] - X_seed[is_gap_mask])

        X_try = X + step * g
        f_try, _ = lorentz_objective_and_grad(X_try, D_wish, mask_if, c,
                                              continuity=continuity,
                                              freeze_mask=freeze_mask)
        if f_try > f:
            X = X_try
            best_f = max(best_f, f_try)
            no_improve = 0
            step *= 1.02
        else:
            no_improve += 1
            step *= 0.5
            if step < 1e-6:
                step = 1e-6

        hist["f"].append(float(f_try))
        if no_improve > patience:
            break

    return X, hist

def generate_structure_from_scratch(D_wish: np.ndarray, M_if: np.ndarray, n_points: int, c: float = 1.0, max_iter: int = 1000):
    if n_points <= 0:
        return np.zeros((0, 3), dtype=float)
    X = np.random.normal(0, 0.1, (n_points, 3))
    step = 0.05
    min_step = 1e-6
    no_improve = 0
    for _ in range(max_iter):
        f, g = lorentz_objective_and_grad(X, D_wish, M_if, c)
        g_norm = np.linalg.norm(g)
        if g_norm > 1.0:
            g /= max(g_norm, 1e-12)
        X_try = X + step * g
        f_try, _ = lorentz_objective_and_grad(X_try, D_wish, M_if, c)
        if f_try > f:
            X = X_try
            no_improve = 0
            step = min(step * 1.1, 0.1)
        else:
            no_improve += 1
            step = max(step * 0.7, min_step)
        if no_improve > 20 or step <= min_step:
            break
    return X - np.mean(X, axis=0, keepdims=True)

def optimize_gaps_improved(
    X_init: np.ndarray,
    is_gap_mask: np.ndarray,
    D_wish: np.ndarray,
    M_if: np.ndarray,
    c: float = 1.0,
    max_iter: int = 1000,
    repulsion_strength: float = 0.1,
    min_dist: float = 0.1,
    anchor_w: float = 1e-2,
    chain_w: float = 0.05,
    freeze_present: bool = False,
    continuity: Optional[Dict[str, Tuple[int, np.ndarray, float]]] = None,
):
    def _gap_repulsion_gradient(X, is_gap_mask, strength=0.1, min_dist=0.1):
        n = len(X)
        g = np.zeros_like(X)
        gap_idx = np.where(is_gap_mask)[0]
        for ix in range(len(gap_idx)):
            i = gap_idx[ix]
            for j in gap_idx[ix+1:]:
                diff = X[i] - X[j]
                d = np.linalg.norm(diff) + 1e-8
                if d < min_dist:
                    force = strength * diff / (d ** 3)
                    g[i] += force
                    g[j] -= force
        return g

    X = X_init.copy()
    step = 0.05
    no_improve = 0
    patience = 20
    for _ in range(max_iter):
        f, g = lorentz_objective_and_grad(X, D_wish, M_if, c, continuity=continuity)
        g += _gap_repulsion_gradient(X, is_gap_mask, strength=repulsion_strength, min_dist=min_dist)
        if anchor_w > 0:
            g[is_gap_mask] += anchor_w * (X[is_gap_mask] - X_init[is_gap_mask])
        if chain_w > 0 and len(X) > 1:
            for i in range(len(X) - 1):
                v = X[i] - X[i + 1]
                d = np.linalg.norm(v) + 1e-8
                init_d = np.linalg.norm(X_init[i] - X_init[i + 1])
                desired = init_d if init_d > 1e-8 else 1.0
                F = chain_w * (d - desired) * (v / d)
                if is_gap_mask[i]:
                    g[i] += F
                if is_gap_mask[i + 1]:
                    g[i + 1] -= F
        if freeze_present:
            g[~is_gap_mask] = 0.0

        g_norm = np.linalg.norm(g)
        if g_norm > 1.0:
            g /= max(g_norm, 1e-12)

        X_try = X + step * g
        f_try, _ = lorentz_objective_and_grad(X_try, D_wish, M_if, c, continuity=continuity)
        if f_try > f:
            X = X_try
            no_improve = 0
            step = min(step * 1.1, 0.1)
        else:
            no_improve += 1
            step = max(step * 0.7, 1e-6)
        if no_improve > patience:
            break
    return X
