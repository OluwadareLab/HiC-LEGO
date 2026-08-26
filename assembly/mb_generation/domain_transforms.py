import numpy as np
from scipy.spatial.transform import Rotation as R

def apply_rigid_transform(coords, params):

    if len(coords) == 0:
        return coords.copy()

    rx, ry, rz, log_scale, tx, ty, tz = params

    rot = R.from_euler('xyz', [rx, ry, rz], degrees=False)
    X_rot = rot.apply(coords)

    scale = np.exp(log_scale)
    scale = np.clip(scale, 0.1, 10.0)
    X_scaled = X_rot * scale

    X_final = X_scaled + np.array([tx, ty, tz])

    return X_final

def check_coordinate_sanity(coords, label="Coordinates", max_magnitude=1e5):

    if len(coords) == 0:
        return True, 0.0

    max_val = np.max(np.abs(coords))
    has_nan = np.any(~np.isfinite(coords))

    if has_nan:
        print(f"  [ERROR] {label}: Contains NaN or Inf values")
        return False, max_val

    if max_val > max_magnitude:
        print(f"  [ERROR] {label}: Max value {max_val:.2e} exceeds threshold {max_magnitude:.2e}")
        return False, max_val

    return True, max_val

def get_domain_alpha(domain_index, domain_folder):

    import os
    import re

    domain_name = f"domain{domain_index + 1}"
    log_path = os.path.join(domain_folder, domain_name, f"{domain_name}_trained_log.txt")

    if not os.path.exists(log_path):
        return None

    try:
        with open(log_path, 'r') as f:
            content = f.read()
            match = re.search(r'Optimal conversion factor:\s*([\d.]+)', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"  [WARN] Failed to read alpha from {log_path}: {e}")

    return None

def compute_median_alpha(domain_indices, domain_folder, default=0.3):

    alphas = []
    for di in domain_indices:
        alpha = get_domain_alpha(di, domain_folder)
        if alpha is not None and alpha > 0:
            alphas.append(alpha)

    if len(alphas) == 0:
        print(f"  [WARN] No valid alphas found, using default {default}")
        return default, []

    median_alpha = float(np.median(alphas))
    return median_alpha, alphas

def build_initial_structure(domain_slots, eff_start, eff_end, fine_res=10000, allowed_bins=None):

    if allowed_bins is None:
        abs_bins = list(range(eff_start, eff_end, fine_res))
    else:
        abs_bins = [b for b in range(eff_start, eff_end, fine_res) if b in allowed_bins]
    n_bins = len(abs_bins)

    if n_bins == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=bool), []

    X_init = np.zeros((n_bins, 3), dtype=float)
    present_mask = np.zeros(n_bins, dtype=bool)

    abs_bin_to_idx = {b: i for i, b in enumerate(abs_bins)}

    for (di, start_bp, end_bp, coords) in domain_slots:
        dom_bins = list(range(start_bp, end_bp, fine_res))
        n_dom_bins = len(dom_bins)
        n_coords = len(coords)

        if n_coords != n_dom_bins:
            if n_coords == 0:
                continue
            indices = np.clip(
                np.round(np.linspace(0, n_coords - 1, n_dom_bins)).astype(int),
                0, n_coords - 1
            )
            coords_resampled = coords[indices]
        else:
            coords_resampled = coords

        for i, bin_id in enumerate(dom_bins):
            idx = abs_bin_to_idx.get(bin_id)
            if idx is not None:
                X_init[idx] = coords_resampled[i]
                present_mask[idx] = True

    return X_init, present_mask, abs_bins
