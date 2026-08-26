import os
import sys
import argparse
import json
import multiprocessing as mp
import random
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.stats import spearmanr

GLOBAL_SEED = 42

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_all_seeds(GLOBAL_SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

_QUIET_LOG = True

def _log(*args, **kwargs):
    flush = kwargs.pop("flush", True)
    force = kwargs.pop("force", False)
    kwargs.pop("sep", None)
    kwargs.pop("end", None)
    kwargs.pop("file", None)

    msg = " ".join(str(a) for a in args)
    if force or not _QUIET_LOG:
        print(msg, flush=flush)
        return

    m = msg.strip()
    if not m:
        return

    mu = m.upper()
    if "[ERROR]" in mu or mu.startswith("ERROR"):
        print(m, flush=flush)
        return
    if "[WARN" in mu or "[WARNING]" in mu or mu.startswith("WARNING"):
        print(m, flush=flush)
        return

    step_prefixes = (
        "STEP ",
        "PHASE 1 (",
        "FRESH MB STRUCTURE GENERATION",
        "LOADING DOMAIN STRUCTURES",
        "PARSING BED FILE",
        "COMPUTING MB REGION BOUNDARIES",
        "LOADING HI-C MATRIX",
        "GENERATING MB STRUCTURES",
        "GENERATION COMPLETE",
        "BYPASSING HARD STOP",
    )
    mu_lstrip = m.lstrip().upper()
    if any(mu_lstrip.startswith(pref) for pref in step_prefixes):
        if set(m.replace(" ", "")) <= {"=", "-", "*"}:
            return
        print(m, flush=flush)

MB_MULTI_START_CONFIG = {
    "enabled": True,
    "num_starts": 5,
    "base_seed": 42,
    "n_rounds": 3,
    "max_iter": 200,
    "rot_std": 0.30,
    "trans_std": 25.0,
    "logscale_std": 0.10,
    "gap_noise_std": 5.0,
    "refine_only_best": True,
    "pre_refine_select": True,
    "sanity_penalty_max_coord": 50000.0,
    "max_reasonable_coord_penalty": 0.000001,
}

_MB_DOMAIN_COORDS = None
_MB_DOMAINS = None
_MB_MB_GROUPS = None
_MB_EFFECTIVE_RANGES = None
_MB_HIC = None
_MB_BED_LOOKUP = None
_MB_ARGS = None
_MB_REFINE_PARAMS = None
_MB_ALLOWED_BINS = None

def _mb_worker_init(domain_coords, domains, mb_groups, effective_ranges, hic, bed_lookup,
                    args_dict, refine_params, allowed_bins):
    global _MB_DOMAIN_COORDS, _MB_DOMAINS, _MB_MB_GROUPS, _MB_EFFECTIVE_RANGES
    global _MB_HIC, _MB_BED_LOOKUP, _MB_ARGS, _MB_REFINE_PARAMS, _MB_ALLOWED_BINS
    _MB_DOMAIN_COORDS = domain_coords
    _MB_DOMAINS = domains
    _MB_MB_GROUPS = mb_groups
    _MB_EFFECTIVE_RANGES = effective_ranges
    _MB_HIC = hic
    _MB_BED_LOOKUP = bed_lookup
    _MB_ARGS = args_dict
    _MB_REFINE_PARAMS = refine_params
    _MB_ALLOWED_BINS = allowed_bins

def _run_one_mb_worker(mb_idx: int) -> bool:
    if _MB_MB_GROUPS is None:
        raise RuntimeError("Worker not initialized")

    if mb_idx not in _MB_MB_GROUPS:
        _log(f"\nMB_{mb_idx:03d}: No domains assigned, skipping")
        return False

    domains_in_mb = _MB_MB_GROUPS[mb_idx]
    eff_range = _MB_EFFECTIVE_RANGES[mb_idx]

    return generate_mb_structure(
        mb_idx,
        domains_in_mb,
        _MB_DOMAIN_COORDS,
        eff_range,
        _MB_HIC,
        _MB_ARGS["domain_folder"],
        _MB_BED_LOOKUP,
        allowed_bins=_MB_ALLOWED_BINS,
        fine_res=int(_MB_ARGS["fine_res"]),
        output_dir=_MB_ARGS["output_dir"],
        adaptive_alpha=not bool(_MB_ARGS["disable_adaptive_alpha"]),
        post_refine=bool(_MB_ARGS["enable_refine"]),
        refine_params=_MB_REFINE_PARAMS,
    )

_script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_script_dir, '..')))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    from .domain_transforms import (
        apply_rigid_transform, check_coordinate_sanity,
        compute_median_alpha, build_initial_structure,
    )
except Exception:
    from domain_transforms import (
        apply_rigid_transform, check_coordinate_sanity,
        compute_median_alpha, build_initial_structure,
    )

def adaptive_alpha_selection(abs_bins, hic, X_init, alpha_list, default_alpha=0.3,
                             grid_min=0.1, grid_max=1.2, grid_step=0.05,
                             min_contacts=3):
    candidates = set()
    if alpha_list:
        candidates.update([float(a) for a in alpha_list if a > 0])
        candidates.add(float(np.median(alpha_list)))
    candidates.add(default_alpha)
    for a in np.arange(grid_min, grid_max + 1e-9, grid_step):
        candidates.add(round(float(a), 3))
    candidates = sorted(candidates)

    results = []
    best_alpha = default_alpha
    best_dscc = -np.inf
    median_anchor = float(np.median(alpha_list)) if alpha_list else default_alpha

    for alpha in candidates:
        D_wish, M_if = build_wish_distance_matrix(abs_bins, hic, alpha=alpha)
        n_contacts = int(np.count_nonzero(M_if & (D_wish > 0)))
        if n_contacts < min_contacts:
            results.append({"alpha": alpha, "dscc": None, "n_contacts": n_contacts})
            continue
        dscc_val = compute_dscc(X_init, D_wish, M_if)
        results.append({"alpha": alpha, "dscc": dscc_val, "n_contacts": n_contacts})

        if (dscc_val > best_dscc + 1e-9) or (
            abs(dscc_val - best_dscc) <= 1e-9 and
            abs(alpha - median_anchor) < abs(best_alpha - median_anchor)
        ):
            best_alpha = alpha
            best_dscc = dscc_val

    return best_alpha, results

def refine_structure(X, D_wish, M_if, present_mask,
                     lam_contact=1.0, lam_smooth=0.01, lam_chain=0.01,
                     max_iter=150, revert_threshold_drop=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X0 = X.copy()
    n = len(X0)
    if n < 3:
        return X0, {"refined": False, "reason": "too_few_beads"}

    iu, ju = np.where(np.triu(M_if, 1) & (D_wish > 0))
    if len(iu) < 3:
        return X0, {"refined": False, "reason": "too_few_contacts"}

    neighbor_d = [np.linalg.norm(X0[i] - X0[i + 1]) for i in range(n - 1)]
    d_ref = float(np.median(neighbor_d)) if neighbor_d else 1.0

    t_X = torch.tensor(X0, dtype=torch.float64, device=device, requires_grad=True)
    t_iu = torch.tensor(iu, dtype=torch.long, device=device)
    t_ju = torch.tensor(ju, dtype=torch.long, device=device)
    t_wish = torch.tensor(D_wish[iu, ju], dtype=torch.float64, device=device)

    bw_arr = np.where(present_mask, 0.5, 1.0)
    t_bead_weight = torch.tensor(bw_arr[1:-1], dtype=torch.float64, device=device)

    optimizer = torch.optim.LBFGS(
        [t_X],
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()

        diff_c = t_X[t_iu] - t_X[t_ju]
        dist_c = torch.norm(diff_c, dim=1) + 1e-8
        t_c = dist_c - t_wish
        E_contact = lam_contact * torch.sum(t_c * t_c)

        v_s = t_X[:-2] - 2.0 * t_X[1:-1] + t_X[2:]
        v_s_sq = torch.sum(v_s * v_s, dim=1)
        E_smooth = lam_smooth * torch.sum(t_bead_weight * v_s_sq)

        v_ch = t_X[:-1] - t_X[1:]
        dist_ch = torch.norm(v_ch, dim=1) + 1e-8
        t_ch = dist_ch - d_ref
        E_chain = lam_chain * torch.sum(t_ch * t_ch)

        loss = E_contact + E_smooth + E_chain
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception as e:
        _log(f"  [WARN] PyTorch optimization failed: {e}")
        return X0, {"refined": False, "reason": "optimizer_crashed"}

    Xr = t_X.detach().cpu().numpy()

    dscc_before = compute_dscc(X0, D_wish, M_if)
    dscc_after = compute_dscc(Xr, D_wish, M_if)

    dropped = dscc_after < dscc_before - revert_threshold_drop
    if dropped:
        return X0, {
            "refined": False,
            "reason": "dscc_drop",
            "dscc_before": dscc_before,
            "dscc_after": dscc_after,
            "opt_status": 0,
            "opt_message": "Reverted due to score drop"
        }

    final_obj = None
    try:
        with torch.no_grad():
            final_obj = float(closure().item())
    except Exception:
        final_obj = None

    return Xr, {
        "refined": True,
        "dscc_before": dscc_before,
        "dscc_after": dscc_after,
        "opt_status": 1,
        "opt_message": "Successfully converged on GPU",
        "final_objective": final_obj,
        "iterations": optimizer.state[optimizer._params[0]].get('n_iter', 0),
        "contact_pairs": int(len(iu)),
        "neighbor_ref_dist": d_ref
    }

def parse_bed(bed_path):
    domains = []
    with open(bed_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            domains.append((chrom, start, end))
    return domains

def compute_effective_ranges(domains, MB=1_000_000):
    mb_groups = {}
    effective_ranges = {}
    prev_end = 0

    for (chrom, start, end) in domains:
        mb_idx = start // MB

        if mb_idx not in mb_groups:
            mb_groups[mb_idx] = []
        mb_groups[mb_idx].append((chrom, start, end))

        if mb_idx not in effective_ranges:
            effective_ranges[mb_idx] = [prev_end, end]
        else:
            effective_ranges[mb_idx][1] = max(effective_ranges[mb_idx][1], end)

        prev_end = max(prev_end, end)

    effective_ranges = {k: tuple(v) for k, v in effective_ranges.items()}
    return mb_groups, effective_ranges

def load_domain_structures(domain_folder):
    domain_coords = {}

    if not os.path.exists(domain_folder):
        _log(f"[ERROR] Domain folder not found: {domain_folder}")
        return domain_coords

    entries = os.listdir(domain_folder)

    for entry in entries:
        if not entry.startswith('domain'):
            continue

        try:
            di = int(entry[6:]) - 1
        except Exception:
            continue

        pdb_path = os.path.join(domain_folder, entry, f"{entry}_trained_structure.pdb")

        if os.path.exists(pdb_path):
            coords = read_pdb_coords(pdb_path)
            domain_coords[di] = coords

    return domain_coords

def read_pdb_coords(pdb_path):
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except Exception:
                    continue
    return np.array(coords, dtype=float)

def write_pdb(coords, output_path):
    with open(output_path, 'w') as f:
        for i, (x, y, z) in enumerate(coords):
            f.write(
                f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )

def load_hic_matrix(hic_path):
    hic = {}
    with open(hic_path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
            hic[(i, j)] = v
            hic[(j, i)] = v
    return hic

def load_allowed_bins_from_mapping(mapping_path):
    allowed = set()
    with open(mapping_path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 1:
                continue
            try:
                allowed.add(int(parts[0]))
            except Exception:
                continue
    return allowed

def infer_allowed_bins_from_hic(hic):
    bins = set()
    for i, j in hic.keys():
        bins.add(int(i))
        bins.add(int(j))
    return bins

def build_wish_distance_matrix(abs_bins, hic, alpha=0.3):
    n = len(abs_bins)
    D_wish = np.zeros((n, n), dtype=float)
    M_if = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            bin_i = abs_bins[i]
            bin_j = abs_bins[j]
            freq = hic.get((bin_i, bin_j), 0.0)

            if freq > 0:
                distance = freq ** (-alpha)
                D_wish[i, j] = D_wish[j, i] = distance
                M_if[i, j] = M_if[j, i] = True

    return D_wish, M_if

def compute_dscc(coords, D_wish, M_if):
    n = len(coords)
    if n < 3:
        return 0.0

    D_geom = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            D_geom[i, j] = D_geom[j, i] = dist

    sel = M_if & (D_wish > 0)
    if np.count_nonzero(sel) < 3:
        return 0.0

    corr = spearmanr(D_geom[sel], D_wish[sel]).correlation
    if corr is None or not np.isfinite(corr):
        return 0.0

    return float(corr)

def resample_domain_coords(orig_coords, n_slots):
    n_coords = len(orig_coords)
    if n_coords == n_slots:
        return orig_coords
    if n_coords == 0:
        return np.zeros((n_slots, 3), dtype=float)
    idxs = np.clip(
        np.round(np.linspace(0, n_coords - 1, num=n_slots)).astype(int),
        0,
        n_coords - 1
    )
    return orig_coords[idxs]

def prepare_domain_slots_with_indices(domain_slots, abs_bins, fine_res):
    abs_bin_to_idx = {b: i for i, b in enumerate(abs_bins)}
    domain_slots_with_indices = []

    for (di, start_bp, end_bp, coords) in domain_slots:
        dom_bins = list(range(start_bp, end_bp, fine_res))
        slot_indices = [abs_bin_to_idx[b] for b in dom_bins if b in abs_bin_to_idx]
        domain_slots_with_indices.append((di, start_bp, end_bp, coords, slot_indices))

    return domain_slots_with_indices

def make_random_init_params(n_domains, rng,
                            rot_std=0.30,
                            trans_std=25.0,
                            logscale_std=0.10):
    params = []
    for _ in range(n_domains):
        rx, ry, rz = rng.normal(0.0, rot_std, size=3)
        log_scale = rng.normal(0.0, logscale_std)
        tx, ty, tz = rng.normal(0.0, trans_std, size=3)
        params.append(np.array([rx, ry, rz, log_scale, tx, ty, tz], dtype=float))
    return params

def score_structure(dscc_value, max_coord, cfg):
    if dscc_value is None or not np.isfinite(dscc_value):
        return -np.inf
    penalty = 0.0
    if max_coord is not None and np.isfinite(max_coord):
        penalty = cfg["max_reasonable_coord_penalty"] * float(max_coord)
    return float(dscc_value) - penalty

def build_gap_initialized_structure(X_init, present_mask):
    X_gap_init = X_init.copy()
    gap_idx = np.where(~present_mask)[0]
    present_idx = np.where(present_mask)[0]

    for i in gap_idx:
        left = present_idx[present_idx < i]
        right = present_idx[present_idx > i]
        if len(left) and len(right):
            X_gap_init[i] = 0.5 * (X_init[left[-1]] + X_init[right[0]])
        elif len(left):
            X_gap_init[i] = X_init[left[-1]]
        elif len(right):
            X_gap_init[i] = X_init[right[0]]
        else:
            X_gap_init[i] = np.zeros(3)

    return X_gap_init

def optimize_gap_coordinates(X_gap_init, gap_idx, D_wish, M_if):
    try:
        from .mb_gap_opt import lorentz_objective_and_grad
    except ImportError:
        from mb_gap_opt import lorentz_objective_and_grad

    iu_gap, ju_gap = np.where(np.triu(M_if, 1) & (D_wish > 0))
    pairs_gap = (iu_gap, ju_gap)

    def gap_objective(X_gap_flat):
        X = X_gap_init.copy()
        X[gap_idx] = X_gap_flat.reshape((-1, 3))
        f, _ = lorentz_objective_and_grad(X, D_wish, M_if, c=1.0, pairs=pairs_gap)
        return -f

    def gap_jac(X_gap_flat):
        X = X_gap_init.copy()
        X[gap_idx] = X_gap_flat.reshape((-1, 3))
        _, g = lorentz_objective_and_grad(X, D_wish, M_if, c=1.0, pairs=pairs_gap)
        return (-g[gap_idx]).reshape((-1, 3)).flatten()

    X_gap_flat = X_gap_init[gap_idx].reshape((-1, 3)).flatten()

    result = minimize(
        gap_objective,
        X_gap_flat,
        jac=gap_jac,
        method='L-BFGS-B',
        options={'maxiter': 400}
    )

    X_gap_opt = X_gap_init.copy()
    X_gap_opt[gap_idx] = result.x.reshape((-1, 3))
    return X_gap_opt, result

def optimize_mb_structure(X_init, domain_slots, D_wish, M_if,
                          max_iter=200, init_params=None, n_rounds=2):
    n_domains = len(domain_slots)

    if n_domains == 0:
        return X_init.copy(), []

    if init_params is None:
        params = [np.zeros(7, dtype=float) for _ in range(n_domains)]
    else:
        if len(init_params) != n_domains:
            raise ValueError(
                f"init_params length mismatch: got {len(init_params)}, expected {n_domains}"
            )
        params = [np.array(p, dtype=float).copy() for p in init_params]

    iu, ju = np.where(np.triu(M_if, 1) & (D_wish > 0))

    if len(iu) < 3:
        _log("  [WARN] Too few Hi-C contacts for optimization")
        return X_init.copy(), []

    bounds = Bounds(
        lb=[-np.pi, -np.pi, -np.pi, -2.3, -1000, -1000, -1000],
        ub=[ np.pi,  np.pi,  np.pi,  2.3,  1000,  1000,  1000]
    )

    domain_resampled = []
    for _, _, _, orig_coords, slot_indices in domain_slots:
        coords_resampled = resample_domain_coords(orig_coords, len(slot_indices))
        domain_resampled.append(coords_resampled)

    _log(f"  Optimizing {n_domains} domains...")
    for round_idx in range(n_rounds):
        _log(f"    Round {round_idx + 1}/{n_rounds}")

        for di, (domain_idx, start_bp, end_bp, orig_coords, slot_indices) in enumerate(domain_slots):
            coords_resampled = domain_resampled[di]

            X_current = X_init.copy()
            for dj, (_, _, _, _, indices_j) in enumerate(domain_slots):
                X_current[indices_j] = apply_rigid_transform(domain_resampled[dj], params[dj])

            def objective(p):
                X_test = X_current.copy()
                X_test[slot_indices] = apply_rigid_transform(coords_resampled, p)

                d_geom = np.linalg.norm(X_test[iu] - X_test[ju], axis=1)
                d_wish = D_wish[iu, ju]

                valid = np.isfinite(d_geom) & np.isfinite(d_wish) & (d_wish > 0)
                if np.count_nonzero(valid) < 3:
                    return 0.0

                corr = spearmanr(d_geom[valid], d_wish[valid]).correlation
                if corr is None or not np.isfinite(corr):
                    return 0.0

                return -float(corr)

            result = minimize(
                objective,
                params[di],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter}
            )
            params[di] = result.x

    X_final = X_init.copy()
    transforms_log = []

    for di, (domain_idx, start_bp, end_bp, _, slot_indices) in enumerate(domain_slots):
        coords_resampled = domain_resampled[di]
        X_final[slot_indices] = apply_rigid_transform(coords_resampled, params[di])

        rx, ry, rz, log_scale, tx, ty, tz = params[di]
        transforms_log.append({
            "domain_index": int(domain_idx),
            "start_bp": int(start_bp),
            "end_bp": int(end_bp),
            "rotation": [float(rx), float(ry), float(rz)],
            "scale": float(np.exp(log_scale)),
            "translation": [float(tx), float(ty), float(tz)]
        })

    return X_final, transforms_log

def generate_mb_structure(mb_idx, domains, domain_coords, effective_range,
                          hic, domain_folder, bed_lookup, fine_res=10000, output_dir='outputs',
                          adaptive_alpha=True, post_refine=True,
                          allowed_bins=None,
                          refine_params=None):
    _log(f"\n{'='*60}")
    _log(f"Generating MB_{mb_idx:03d}")
    _log(f"{'='*60}")

    eff_start, eff_end = effective_range
    _log(f"  Effective range: {eff_start:,} - {eff_end:,} bp")
    _log(f"  Domains: {len(domains)}")

    domain_slots = []
    domain_indices = []

    for (chrom, start_bp, end_bp) in domains:
        di = bed_lookup.get((chrom, start_bp, end_bp))
        if di is not None and di in domain_coords:
            coords = domain_coords[di]
            domain_indices.append(di)
            domain_slots.append((di, start_bp, end_bp, coords))

    if len(domain_slots) == 0:
        _log(f"  [ERROR] No domain coordinates found")
        return False

    _log(f"  Loaded {len(domain_slots)} domain structures")

    alpha_median, alpha_list = compute_median_alpha(domain_indices, domain_folder, default=0.3)
    _log(f"  Domain alpha median (baseline): {alpha_median:.3f}")
    if alpha_list:
        _log(f"  Domain alphas: {[f'{a:.2f}' for a in alpha_list]}")

    X_init, present_mask, abs_bins = build_initial_structure(
        domain_slots, eff_start, eff_end, fine_res=fine_res, allowed_bins=allowed_bins
    )
    _log(f"  Structure size: {len(X_init)} beads")
    if allowed_bins is not None:
        _log(f"  Allowed-bin filter active: {len(allowed_bins)} valid bins in reference")
    n_present = np.count_nonzero(present_mask)
    n_gaps = np.count_nonzero(~present_mask)
    _log(f"  Present: {n_present}, Gaps: {n_gaps}")

    if len(X_init) == 0:
        _log("  [ERROR] No valid bins in this MB after allowed-bin filtering")
        return False

    is_sane_init, max_val_init = check_coordinate_sanity(
        X_init[present_mask], label="Initial coordinates"
    )
    if not is_sane_init:
        _log(f"  [ERROR] Initial coordinates failed sanity check")
        return False
    _log(f"  Initial coordinate range: max={max_val_init:.2f}")

    if adaptive_alpha:
        chosen_alpha, alpha_scan = adaptive_alpha_selection(
            abs_bins, hic, X_init, alpha_list, default_alpha=alpha_median
        )
        _log(f"  Adaptive alpha chosen: {chosen_alpha:.3f} (scan {len(alpha_scan)} candidates)")
    else:
        chosen_alpha = alpha_median
        alpha_scan = []

    D_wish, M_if = build_wish_distance_matrix(abs_bins, hic, alpha=chosen_alpha)
    n_contacts = np.count_nonzero(M_if & (D_wish > 0))
    _log(f"  Hi-C contacts: {n_contacts}")
    if n_contacts < 3:
        _log(f"  [ERROR] Too few Hi-C contacts")
        return False

    dscc_initial = compute_dscc(X_init, D_wish, M_if)
    _log(f"  Initial dSCC: {dscc_initial:.4f}")

    domain_slots_with_indices = prepare_domain_slots_with_indices(domain_slots, abs_bins, fine_res)

    cfg = MB_MULTI_START_CONFIG
    multi_enabled = bool(cfg.get("enabled", True))
    num_starts = int(cfg.get("num_starts", 1)) if multi_enabled else 1
    base_seed = int(cfg.get("base_seed", GLOBAL_SEED))
    n_rounds = int(cfg.get("n_rounds", 2))
    max_iter = int(cfg.get("max_iter", 200))

    _log(f"  Multi-start enabled: {multi_enabled}")
    _log(f"  Number of starts: {num_starts}")

    has_mixed_gaps = (n_present > 0 and n_gaps > 0)
    gap_idx = np.where(~present_mask)[0] if has_mixed_gaps else np.array([], dtype=int)

    base_gap_frame = None
    gap_opt_result_message = None
    if has_mixed_gaps:
        _log("  Mixed gaps detected: using gap optimization + multi-start domain transform search")
        X_gap_init = build_gap_initialized_structure(X_init, present_mask)
        base_gap_frame, gap_opt_result = optimize_gap_coordinates(X_gap_init, gap_idx, D_wish, M_if)
        gap_opt_result_message = getattr(gap_opt_result, "message", None)

    candidate_records = []
    best_pre = None
    best_pre_score = -np.inf

    for start_idx in range(num_starts):
        start_seed = base_seed + 1000 * int(mb_idx) + start_idx
        rng = np.random.default_rng(start_seed)

        if has_mixed_gaps:
            X_start = base_gap_frame.copy()
            if start_idx > 0 and len(gap_idx) > 0:
                X_start[gap_idx] += rng.normal(
                    0.0, cfg["gap_noise_std"], size=X_start[gap_idx].shape
                )
        else:
            X_start = X_init.copy()

        if start_idx == 0:
            init_params = [np.zeros(7, dtype=float) for _ in range(len(domain_slots_with_indices))]
            init_type = "baseline_zero"
        else:
            init_params = make_random_init_params(
                n_domains=len(domain_slots_with_indices),
                rng=rng,
                rot_std=cfg["rot_std"],
                trans_std=cfg["trans_std"],
                logscale_std=cfg["logscale_std"]
            )
            init_type = "randomized"

        _log(f"  Start {start_idx + 1}/{num_starts} | seed={start_seed} | init={init_type}")

        X_candidate, transforms_candidate = optimize_mb_structure(
            X_start,
            domain_slots_with_indices,
            D_wish,
            M_if,
            max_iter=max_iter,
            init_params=init_params,
            n_rounds=n_rounds
        )

        sane_candidate, max_coord_candidate = check_coordinate_sanity(
            X_candidate, label=f"Candidate coordinates (start {start_idx + 1})"
        )

        if not sane_candidate:
            _log(f"    [WARN] Candidate {start_idx + 1} failed sanity check; reverting to start frame")
            X_candidate = X_start.copy()
            transforms_candidate = []
            _, max_coord_candidate = check_coordinate_sanity(
                X_candidate, label=f"Fallback candidate coordinates (start {start_idx + 1})"
            )

        dscc_candidate_pre = compute_dscc(X_candidate, D_wish, M_if)
        score_candidate_pre = score_structure(dscc_candidate_pre, max_coord_candidate, cfg)

        candidate_record = {
            "start_idx": int(start_idx),
            "seed": int(start_seed),
            "init_type": init_type,
            "pre_refine_dscc": float(dscc_candidate_pre),
            "pre_refine_score": float(score_candidate_pre),
            "max_coordinate": float(max_coord_candidate) if np.isfinite(max_coord_candidate) else None,
            "sane": bool(sane_candidate),
            "X_pre": X_candidate.copy(),
            "transforms_pre": transforms_candidate,
        }

        _log(
            f"    Pre-refine dSCC={dscc_candidate_pre:.4f} | "
            f"score={score_candidate_pre:.4f} | max_coord={max_coord_candidate:.2f}"
        )

        candidate_records.append(candidate_record)

        if score_candidate_pre > best_pre_score:
            best_pre_score = score_candidate_pre
            best_pre = candidate_record

    if best_pre is None:
        _log("  [ERROR] Multi-start search produced no valid candidates")
        return False

    _log(
        f"  Best pre-refine start: {best_pre['start_idx'] + 1}/{num_starts} "
        f"(seed={best_pre['seed']}, dSCC={best_pre['pre_refine_dscc']:.4f})"
    )

    X_final = best_pre["X_pre"].copy()
    transforms_log = best_pre["transforms_pre"]
    selected_start_idx = best_pre["start_idx"]
    selected_seed = best_pre["seed"]
    dscc_final = best_pre["pre_refine_dscc"]
    refine_metrics = {}

    if post_refine:
        _log("  Post-refinement: refining winning candidate only")
        rp = refine_params or {}
        X_refined, refine_metrics = refine_structure(
            X_final, D_wish, M_if, present_mask,
            lam_contact=rp.get('lam_contact', 1.0),
            lam_smooth=rp.get('lam_smooth', 0.01),
            lam_chain=rp.get('lam_chain', 0.01),
            max_iter=rp.get('max_iter', 150),
            revert_threshold_drop=rp.get('revert_threshold_drop', 0.01)
        )

        if refine_metrics.get("refined"):
            _log(
                f"    Refinement applied. dSCC "
                f"{refine_metrics['dscc_before']:.4f} -> {refine_metrics['dscc_after']:.4f}"
            )
            X_final = X_refined
            dscc_final = refine_metrics['dscc_after']
        else:
            _log(f"    Refinement skipped (reason: {refine_metrics.get('reason')})")
            dscc_final = best_pre["pre_refine_dscc"]
    else:
        _log("  Post-refinement: disabled")

    is_sane_final, max_val_final = check_coordinate_sanity(X_final, label="Final coordinates")
    if not is_sane_final:
        _log(f"  [ERROR] Final coordinates failed sanity check, reverting to selected pre-refine candidate")
        X_final = best_pre["X_pre"].copy()
        transforms_log = best_pre["transforms_pre"]
        dscc_final = best_pre["pre_refine_dscc"]
        refine_metrics = {"refined": False, "reason": "final_sanity_fail_revert"}
        is_sane_final, max_val_final = check_coordinate_sanity(
            X_final, label="Reverted final coordinates"
        )

    _log(f"  Final coordinate range: max={max_val_final:.2f}")
    _log(f"  Final dSCC: {dscc_final:.4f}")
    _log(f"  dSCC improvement: {dscc_final - dscc_initial:+.4f}")

    os.makedirs(output_dir, exist_ok=True)

    output_pdb = os.path.join(output_dir, f"MB_{mb_idx:03d}_structure.pdb")
    output_log = os.path.join(output_dir, f"MB_{mb_idx:03d}.log")

    write_pdb(X_final, output_pdb)

    multi_start_summary = {
        "enabled": bool(multi_enabled),
        "num_starts": int(num_starts),
        "selected_start_idx": int(selected_start_idx),
        "selected_seed": int(selected_seed),
        "selected_pre_refine_dscc": float(best_pre["pre_refine_dscc"]),
        "selected_pre_refine_score": float(best_pre["pre_refine_score"]),
        "settings": {
            "base_seed": int(cfg["base_seed"]),
            "n_rounds": int(cfg["n_rounds"]),
            "max_iter": int(cfg["max_iter"]),
            "rot_std": float(cfg["rot_std"]),
            "trans_std": float(cfg["trans_std"]),
            "logscale_std": float(cfg["logscale_std"]),
            "gap_noise_std": float(cfg["gap_noise_std"]),
            "refine_only_best": bool(cfg["refine_only_best"]),
            "pre_refine_select": bool(cfg["pre_refine_select"]),
        },
        "candidates": [
            {
                "start_idx": int(c["start_idx"]),
                "seed": int(c["seed"]),
                "init_type": c["init_type"],
                "pre_refine_dscc": float(c["pre_refine_dscc"]),
                "pre_refine_score": float(c["pre_refine_score"]),
                "max_coordinate": float(c["max_coordinate"]) if c["max_coordinate"] is not None else None,
                "sane": bool(c["sane"]),
            }
            for c in candidate_records
        ]
    }

    log_data = {
        "mb_index": int(mb_idx),
        "effective_start_bp": int(eff_start),
        "effective_end_bp": int(eff_end),
        "n_beads": int(len(X_final)),
        "n_domains": len(domain_slots),
        "alpha_used": float(chosen_alpha),
        "alpha_baseline_median": float(alpha_median),
        "adaptive_alpha_enabled": bool(adaptive_alpha),
        "domain_alphas": [float(a) for a in alpha_list],
        "n_hic_contacts": int(n_contacts),
        "initial_dscc": float(dscc_initial),
        "final_dscc": float(dscc_final),
        "dscc_improvement": float(dscc_final - dscc_initial),
        "coordinate_sanity": bool(is_sane_final),
        "max_coordinate": float(max_val_final),
        "post_refine_enabled": bool(post_refine),
        "gap_optimization": {
            "used": bool(has_mixed_gaps),
            "optimizer_message": str(gap_opt_result_message) if gap_opt_result_message is not None else None,
        },
        "refine_summary": {
            "refined": bool(refine_metrics.get("refined", False)),
            "reason": refine_metrics.get("reason"),
            "dscc_before": float(refine_metrics.get("dscc_before", best_pre["pre_refine_dscc"])),
            "dscc_after": float(refine_metrics.get("dscc_after", dscc_final)),
            "opt_status": refine_metrics.get("opt_status"),
            "iterations": (
                int(refine_metrics.get("iterations", 0))
                if isinstance(refine_metrics.get("iterations", 0), (int, np.integer))
                else None
            )
        },
        "multi_start": multi_start_summary,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_log, 'w') as f:
        json.dump(log_data, f, indent=2)

    _log(f"  Saved: {output_pdb}")
    _log(f"  Saved: {output_log}")

    return True

def main():
    parser = argparse.ArgumentParser(description='Generate fresh MB structures from domain structures')
    parser.add_argument('--domain-folder', required=True, help='Folder with domain subfolders')
    parser.add_argument('--bed', required=True, help='BED file with domain boundaries')
    parser.add_argument('--hic', required=True, help='Hi-C 3-column file')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--start-mb', type=int, default=0, help='Start MB index')
    parser.add_argument('--end-mb', type=int, default=-1, help='End MB index (inclusive); -1 = auto from BED')
    parser.add_argument('--fine-res', type=int, default=10000, help='Resolution in bp')
    parser.add_argument('--disable-adaptive-alpha', action='store_true', help='Disable adaptive alpha selection')
    parser.add_argument('--enable-refine', action='store_true', help='Enable post-assembly refinement')
    parser.add_argument('--refine-lam-contact', type=float, default=1.0, help='Contact term weight for refinement')
    parser.add_argument('--refine-lam-smooth', type=float, default=0.01, help='Smoothness weight for refinement')
    parser.add_argument('--refine-lam-chain', type=float, default=0.01, help='Chain neighbor regularization weight')
    parser.add_argument('--refine-max-iter', type=int, default=150, help='Max iterations for refinement optimizer')
    parser.add_argument('--refine-revert-drop', type=float, default=0.01, help='Max allowed dSCC drop before revert')
    parser.add_argument('--mb-workers', type=int, default=1, help='Number of parallel workers over MB regions (1 = sequential)')
    parser.add_argument('--coordinate-mapping', default=None, help='Optional coordinate mapping file (first column = allowed genomic bins). If provided, MB beads are restricted to these bins only.')

    args = parser.parse_args()

    _log("Phase 1 (generate_mb_structures) started", flush=True)
    _log("FRESH MB STRUCTURE GENERATION")

    _log("STEP 1: Loading domain structures")
    domain_coords = load_domain_structures(args.domain_folder)

    if len(domain_coords) == 0:
        _log("[ERROR] No domain structures found!")
        return

    _log("STEP 2: Parsing BED file")
    domains = parse_bed(args.bed)
    if len(domains) > 0:
        bed_min = min(start for _, start, _ in domains)
        bed_max = max(end for _, _, end in domains)

    bed_lookup = {}
    for idx, (chrom, start, end) in enumerate(domains):
        bed_lookup[(chrom, start, end)] = idx

    _log("STEP 3: Computing MB region boundaries")
    mb_groups, effective_ranges = compute_effective_ranges(domains, MB=1_000_000)

    max_mb_idx = max(mb_groups.keys()) if mb_groups else args.start_mb
    if args.end_mb < 0:
        end_mb_use = max_mb_idx
    else:
        end_mb_use = args.end_mb
        if end_mb_use < max_mb_idx:
            _log(
                f"[WARN] Provided --end-mb ({end_mb_use}) is below BED coverage "
                f"(max MB {max_mb_idx}). MBs beyond {end_mb_use} will not be generated."
            )

    if args.start_mb > end_mb_use:
        _log(f"[WARN] start-mb ({args.start_mb}) > end-mb ({end_mb_use}); adjusting end-mb to start-mb.")
        end_mb_use = args.start_mb

    _log("STEP 4: Loading Hi-C matrix")
    hic = load_hic_matrix(args.hic)

    if args.coordinate_mapping:
        if not os.path.exists(args.coordinate_mapping):
            _log(f"[ERROR] Coordinate mapping file not found: {args.coordinate_mapping}")
            return
        allowed_bins = load_allowed_bins_from_mapping(args.coordinate_mapping)
    else:
        allowed_bins = infer_allowed_bins_from_hic(hic)

    _log("STEP 5: Generating MB structures")

    refine_params = {
        'lam_contact': args.refine_lam_contact,
        'lam_smooth': args.refine_lam_smooth,
        'lam_chain': args.refine_lam_chain,
        'max_iter': args.refine_max_iter,
        'revert_threshold_drop': args.refine_revert_drop
    }

    success_count = 0
    fail_count = 0
    skip_count = 0

    if args.mb_workers and args.mb_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        args_dict = {
            "domain_folder": args.domain_folder,
            "output_dir": args.output_dir,
            "fine_res": int(args.fine_res),
            "disable_adaptive_alpha": bool(args.disable_adaptive_alpha),
            "enable_refine": bool(args.enable_refine),
        }

        with ProcessPoolExecutor(
            max_workers=args.mb_workers,
            initializer=_mb_worker_init,
            initargs=(domain_coords, domains, mb_groups, effective_ranges, hic, bed_lookup, args_dict, refine_params, allowed_bins),
        ) as ex:
            futures = {ex.submit(_run_one_mb_worker, mb_idx): mb_idx for mb_idx in range(args.start_mb, end_mb_use + 1)}
            for fut in as_completed(futures):
                mb_idx = futures[fut]
                try:
                    ok = bool(fut.result())
                except Exception as e:
                    _log(f"\n[ERROR] MB_{mb_idx:03d} crashed: {e}")
                    ok = False

                if mb_idx not in mb_groups:
                    skip_count += 1
                elif ok:
                    success_count += 1
                else:
                    fail_count += 1
    else:
        for mb_idx in range(args.start_mb, end_mb_use + 1):
            if mb_idx not in mb_groups:
                _log(f"\nMB_{mb_idx:03d}: No domains assigned, skipping")
                skip_count += 1
                ok = False
            else:
                ok = generate_mb_structure(
                    mb_idx,
                    mb_groups[mb_idx],
                    domain_coords,
                    effective_ranges[mb_idx],
                    hic,
                    args.domain_folder,
                    bed_lookup,
                    allowed_bins=allowed_bins,
                    fine_res=args.fine_res,
                    output_dir=args.output_dir,
                    adaptive_alpha=not args.disable_adaptive_alpha,
                    post_refine=args.enable_refine,
                    refine_params=refine_params,
                )
                if ok:
                    success_count += 1
                else:
                    fail_count += 1

    _log("GENERATION COMPLETE")

    if fail_count > 0:
        _log(f"\n[WARNING] {fail_count} MB regions failed to generate.")
        _log("Bypassing hard stop. Passing the successful structures to the Global Assembly stage...")

if __name__ == '__main__':
    main()
