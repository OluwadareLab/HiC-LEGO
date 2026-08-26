import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

def if2pd(input_if, alpha=0.4, eps=1e-6, qclip=(5, 95)):
    W = np.asarray(input_if, dtype=float)
    W = (W + W.T) / 2.0

    D = np.power(W + eps, -alpha)

    finite = np.isfinite(D)
    if np.any(finite):
        q_low, q_high = np.nanpercentile(D[finite], qclip)
        if np.isfinite(q_low) and np.isfinite(q_high) and q_high > q_low:
            D = np.clip(D, q_low, q_high)

    np.fill_diagonal(D, 0.0)
    return D

def compute_sparse_if2pd(sparse_if, required_indices, alpha=0.4, eps=1e-6, qclip=(5,95)):
    import numpy as _np
    from scipy.sparse import issparse as _issparse
    if not _issparse(sparse_if):
        dense = _np.asarray(sparse_if)
        sub = dense[_np.ix_(required_indices, required_indices)]
    else:
        k = len(required_indices)
        sub = _np.zeros((k, k), dtype=float)
        idx_map = {g:i for i,g in enumerate(required_indices)}
        coo = sparse_if.tocoo()
        for r,c,v in zip(coo.row, coo.col, coo.data):
            ir = idx_map.get(r)
            ic = idx_map.get(c)
            if ir is not None and ic is not None:
                sub[ir, ic] += v
                if ir != ic:
                    sub[ic, ir] += v
    D = _np.power(sub + eps, -alpha)
    finite = _np.isfinite(D)
    if _np.any(finite):
        q_low, q_high = _np.nanpercentile(D[finite], qclip)
        if _np.isfinite(q_low) and _np.isfinite(q_high) and q_high > q_low:
            D = _np.clip(D, q_low, q_high)
    _np.fill_diagonal(D, 0.0)
    return D

def rotation_matrix(x, y, eps=1e-12):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx < eps or ny < eps:
        return np.eye(x.size)

    u = x / nx
    v = y - np.dot(u, y) * u
    nv = np.linalg.norm(v)
    if nv < eps:
        return np.eye(x.size)
    v /= nv

    cost = np.dot(x, y) / (nx * ny)
    cost = np.clip(cost, -1.0, 1.0)
    sint = np.sqrt(max(1.0 - cost**2, 0.0))

    I = np.eye(x.size)
    uv = np.column_stack((u, v))
    rot2 = np.array([[cost, -sint], [sint, cost]])
    return I - np.outer(u, u) - np.outer(v, v) + uv @ rot2 @ uv.T

def rotate(x, c, r):
    return ((x - c) @ r) + c

def get_start_point(all_points):
    return np.vstack([pts[0, :] for pts in all_points[1:]])

def get_end_point(all_points):
    return np.vstack([pts[-1, :] for pts in all_points[:-1]])

def evaluate_dist(start_point, end_point):
    return np.linalg.norm(start_point - end_point, axis=1)

def ave_dist(_, pd_mat):
    finite = np.isfinite(pd_mat)
    if np.any(finite):
        return float(np.nanmean(pd_mat[finite]))
    return 1.0

def get_dist_vec(all_points, id_list, pd_mat, khood=5):
    n = len(all_points)
    start_id = [ids[0] for ids in id_list[1:]]
    end_id   = [ids[-1] for ids in id_list[:-1]]

    dist_vec = []
    for i in range(n - 1):
        r = start_id[i] - 1
        c = end_id[i]   - 1
        d = pd_mat[r, c] if 0 <= r < pd_mat.shape[0] and 0 <= c < pd_mat.shape[1] else np.nan

        if not np.isfinite(d) or d <= 0:
            r0, c0 = max(0, r - khood), max(0, c - khood)
            r1, c1 = min(pd_mat.shape[0], r + khood + 1), min(pd_mat.shape[1], c + khood + 1)
            block = pd_mat[r0:r1, c0:c1]
            finite = np.isfinite(block) & (block > 0)
            if np.any(finite):
                d = float(np.nanmedian(block[finite]))
            else:
                d = ave_dist(None, pd_mat)

        dist_vec.append(d)

    return np.array(dist_vec, dtype=float)

def construct_obj_from_hic(interaction_matrix, resolution, chr_name):
    max_pos = int(
        max(np.max(interaction_matrix[:, 0]), np.max(interaction_matrix[:, 1])) / resolution
    ) + 1

    i_ind = (interaction_matrix[:, 0] / resolution).astype(int)
    j_ind = (interaction_matrix[:, 1] / resolution).astype(int)
    x = interaction_matrix[:, 2]

    input_if = coo_matrix((x, (i_ind, j_ind)), shape=(max_pos, max_pos))

    class Flamingo:
        def __init__(self, IF, n_frag, chr_name):
            self.IF = IF
            self.n_frag = n_frag
            self.chr_name = chr_name

        def __repr__(self):
            return (f"Flamingo Object:\n"
                    f"- IF: {self.IF.shape[0]}x{self.IF.shape[1]} sparse matrix\n"
                    f"- n_frag: {self.n_frag}\n"
                    f"- chr_name: '{self.chr_name}'")

    return Flamingo(IF=input_if, n_frag=max_pos, chr_name=chr_name)

def normalize_3col_if(interaction_matrix, resolution, n_iters=50, eps=1e-8):
    max_pos = int(
        max(np.max(interaction_matrix[:, 0]), np.max(interaction_matrix[:, 1])) / resolution
    ) + 1
    i_ind = (interaction_matrix[:, 0] / resolution).astype(int)
    j_ind = (interaction_matrix[:, 1] / resolution).astype(int)
    x = interaction_matrix[:, 2].astype(float)

    M = np.zeros((max_pos, max_pos), dtype=float)
    for i, j, v in zip(i_ind, j_ind, x):
        if 0 <= i < max_pos and 0 <= j < max_pos:
            M[i, j] += v
            if i != j:
                M[j, i] += v

    total_original = M.sum()

    M = M + eps

    adaptive_iters = min(n_iters, max(5, int(20 * 1000 / max_pos)))

    for iter_idx in range(adaptive_iters):
        row_sums = M.sum(axis=1)
        row_sums[row_sums <= 0] = 1.0
        M = (M.T / row_sums).T

        col_sums = M.sum(axis=0)
        col_sums[col_sums <= 0] = 1.0
        M = M / col_sums

        if not np.all(np.isfinite(M)):
            print(f"[WARN] Normalization unstable at iteration {iter_idx+1}/{adaptive_iters} - stopping early")
            break

        if iter_idx > 5:
            row_sums_check = M.sum(axis=1)
            row_sums_nonzero = row_sums_check[row_sums_check > eps]
            if len(row_sums_nonzero) > 0:
                deviation = np.abs(row_sums_nonzero - 1.0).max()
                if deviation < 0.01:
                    break

    if not np.all(np.isfinite(M)):
        print(f"[ERROR] Normalization produced non-finite values! Returning original matrix.")
        return interaction_matrix.copy()

    rows, cols = np.nonzero(M)
    vals = M[rows, cols]
    keep = rows <= cols
    rows = rows[keep]
    cols = cols[keep]
    vals = vals[keep]

    pos_i = (rows * resolution).astype(int)
    pos_j = (cols * resolution).astype(int)
    normalized = np.column_stack([pos_i, pos_j, vals])

    if not np.all(np.isfinite(normalized)):
        print(f"[ERROR] Normalized output contains non-finite values! Returning original.")
        return interaction_matrix.copy()

    return normalized

def construct_flamingo_prediction_from_pdb(pdb_file):
    fragment_ids, coordinates_list = [], []
    with open(pdb_file, "r") as file:
        for line in file:
            if line.startswith("ATOM"):
                fragment_id = int(line[6:11].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                fragment_ids.append(fragment_id)
                coordinates_list.append([x, y, z])

    coordinates = np.array(coordinates_list, dtype=float)
    input_n = len(fragment_ids)
    return {"id": fragment_ids, "coordinates": coordinates, "input_n": input_n}

def extract_coordinates_from_pdb(pdb_file):
    return construct_flamingo_prediction_from_pdb(pdb_file)

import json
import numpy as np
import pandas as pd

def load_pos_to_index_map(txt_path, assume_resolution=None):
    df = pd.read_csv(txt_path, sep=r"\s+", header=None, names=["pos", "idx"], engine="python")
    pos2idx = dict(zip(df["pos"].astype(int), df["idx"].astype(int)))

    if assume_resolution is not None:
        res = int(assume_resolution)
    else:
        uniq = sorted(df["pos"].unique())
        res = int(uniq[1] - uniq[0]) if len(uniq) >= 2 else 5000

    return pos2idx, res

def _bp_to_idx_floor(pos_bp, res):
    return int(pos_bp // res)

def _idx_to_bp(idx0, res):
    return int(idx0 * res)

def build_domain_bins_map_from_files(domains_bed_path,
                                     pos2idx_txt_path,
                                     chr_name="chr17",
                                     assume_resolution=None,
                                     available_domain_keys=None):
    pos2idx, res = load_pos_to_index_map(pos2idx_txt_path, assume_resolution=assume_resolution)

    bed = pd.read_csv(domains_bed_path, sep=r"\s+", header=None,
                      names=["chrom", "start", "end"], engine="python")
    bed = bed[bed["chrom"] == chr_name].copy()
    bed = bed.sort_values(["start", "end"]).reset_index(drop=True)

    if available_domain_keys is not None:
        def _knum(k):
            try:
                return int("".join(ch for ch in k if ch.isdigit()))
            except Exception:
                return 10**9
        keys_sorted = sorted(list(available_domain_keys), key=_knum)
        if len(keys_sorted) != len(bed):
            print(f"[WARN] Number of domains in BED ({len(bed)}) != domains available ({len(keys_sorted)}). "
                  f"Will zip up to min length.")
        n = min(len(bed), len(keys_sorted))
        key_list = keys_sorted[:n]
        bed = bed.iloc[:n].copy()
    else:
        key_list = [f"domain{i+1}" for i in range(len(bed))]

    mp = {}
    for (idx, row), key in zip(bed.iterrows(), key_list):
        s_bp = int(row["start"])
        e_bp = int(row["end"])

        start_idx0 = _bp_to_idx_floor(s_bp, res)
        end_idx0   = _bp_to_idx_floor(max(e_bp - 1, s_bp), res)

        if end_idx0 < start_idx0:
            end_idx0 = start_idx0

        idxs0 = np.arange(start_idx0, end_idx0 + 1, dtype=int)
        bins_1based = (idxs0 + 1).tolist()
        mp[key] = bins_1based

    return mp, res

def reconcile_bins_to_points(global_bins, n_points):
    if len(global_bins) == n_points:
        return list(global_bins)

    if n_points <= 1:
        return [global_bins[len(global_bins)//2]]

    sel = np.linspace(0, len(global_bins) - 1, num=n_points)
    sel = np.round(sel).astype(int)
    sel = np.clip(sel, 0, len(global_bins) - 1)
    return [global_bins[i] for i in sel]

def process_domain_structures(base_folder):
    domain_results = {}
    for domain_folder in sorted(os.listdir(base_folder)):
        domain_path = os.path.join(base_folder, domain_folder)
        if not os.path.isdir(domain_path):
            continue
        pdb_file_name = f"{domain_folder}_trained_structure.pdb"
        pdb_file_path = os.path.join(domain_path, pdb_file_name)
        if os.path.exists(pdb_file_path):
            result = extract_coordinates_from_pdb(pdb_file_path)
            domain_results[domain_folder] = result
        else:
            print(f"[WARN] PDB not found: {pdb_file_name} in {domain_folder}")
    return domain_results

def _read_pdb_simple_coords(pdb_file):
    fragment_ids, coordinates_list = [], []
    with open(pdb_file, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                fragment_id = int(line[6:11].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except Exception:
                continue
            fragment_ids.append(fragment_id)
            coordinates_list.append([x, y, z])
    coords = np.array(coordinates_list, dtype=float)
    coords = _jitter_duplicate_points(coords, eps=0.01)
    return {"id": fragment_ids, "coordinates": coords, "input_n": len(fragment_ids)}

def _jitter_duplicate_points(coords, eps=0.01):
    if coords.shape[0] <= 1:
        return coords
    coords = coords.copy()
    rng = np.random.default_rng(42)
    dup_run = 0
    for i in range(1, coords.shape[0]):
        if np.linalg.norm(coords[i] - coords[i - 1]) < 1e-12:
            dup_run += 1
            v = rng.normal(0, 1.0, size=3)
            n = np.linalg.norm(v)
            if n < 1e-12:
                v = np.array([1.0, 0.0, 0.0])
                n = 1.0
            coords[i] = coords[i] + (dup_run * eps * v / n)
        else:
            dup_run = 0
    return coords

def process_mb_structures(mb_folder):
    mb_results = {}
    if not os.path.isdir(mb_folder):
        print(f"[WARN] MB folder not found: {mb_folder}")
        return mb_results

    for name in sorted(os.listdir(mb_folder)):
        if not name.startswith("MB_") or not name.endswith(".pdb"):
            continue
        if name.endswith("_filled.pdb"):
            base = name[:-11]
        elif name.endswith("_structure.pdb"):
            base = name[:-14]
        else:
            continue

        key = base
        pdb_path = os.path.join(mb_folder, name)
        try:
            result = _read_pdb_simple_coords(pdb_path)
            mb_results[key] = result
        except Exception as e:
            print(f"[WARN] Failed reading PDB {name}: {e}")
            continue
    return mb_results

def build_mb_bins_map_from_meta(mb_folder, resolution=10_000):
    mp = {}
    if not os.path.isdir(mb_folder):
        print(f"[WARN] MB folder not found: {mb_folder}")
        return mp

    for name in sorted(os.listdir(mb_folder)):
        if not (name.startswith("MB_") and name.endswith("_meta.json")):
            continue
        base = name[:-10]
        key = base
        meta_path = os.path.join(mb_folder, name)
        try:
            meta = json.load(open(meta_path))
            s = int(meta.get('effective_start_bp', meta.get('start_bp', 0)))
            e = int(meta.get('effective_end_bp', meta.get('end_bp', s)))
        except Exception as e:
            print(f"[WARN] Failed reading meta {name}: {e}. Using default single bin mapping.")
            s = 0
            e = 0

        start_idx0 = _bp_to_idx_floor(s, resolution)
        end_idx0 = _bp_to_idx_floor(max(e - 1, s), resolution)
        if end_idx0 < start_idx0:
            end_idx0 = start_idx0

        idxs0 = np.arange(start_idx0, end_idx0 + 1, dtype=int)
        bins_1based = (idxs0 + 1).tolist()
        mp[key] = bins_1based

    return mp

def build_mb_bins_map_from_logs(mb_folder, resolution=10_000):
    mp = {}
    if not os.path.isdir(mb_folder):
        print(f"[WARN] MB folder not found: {mb_folder}")
        return mp

    for name in sorted(os.listdir(mb_folder)):
        if not (name.startswith("MB_") and name.endswith(".log")):
            continue
        key = name[:-4]
        log_path = os.path.join(mb_folder, name)
        try:
            meta = json.load(open(log_path))
            s = int(meta.get("effective_start_bp", meta.get("start_bp", 0)))
            e = int(meta.get("effective_end_bp", meta.get("end_bp", s)))
            if e <= s and "n_beads" in meta:
                try:
                    e = s + int(meta["n_beads"]) * int(resolution)
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] Failed reading log {name}: {e}. Skipping.")
            continue

        start_idx0 = _bp_to_idx_floor(s, resolution)
        end_idx0 = _bp_to_idx_floor(max(e - 1, s), resolution)
        if end_idx0 < start_idx0:
            end_idx0 = start_idx0

        idxs0 = np.arange(start_idx0, end_idx0 + 1, dtype=int)
        bins_1based = (idxs0 + 1).tolist()
        mp[key] = bins_1based

    return mp

def build_mb_anchor_fraction_map(mb_folder, mb_size_bp=1_000_000):
    amap = {}
    if not os.path.isdir(mb_folder):
        print(f"[WARN] MB folder not found: {mb_folder}")
        return amap

    for name in sorted(os.listdir(mb_folder)):
        if not (name.startswith("MB_") and name.endswith("_meta.json")):
            continue
        base = name[:-10]
        key = base
        meta_path = os.path.join(mb_folder, name)
        try:
            meta = json.load(open(meta_path))
            s = int(meta.get('effective_start_bp', meta.get('start_bp', 0)))
            e = int(meta.get('effective_end_bp', meta.get('end_bp', s)))
            mid = (s + e) // 2
        except Exception as e:
            print(f"[WARN] Failed reading meta {name}: {e}. Skipping anchor fraction.")
            continue

        if mid < 0:
            mid = 0
        anchor_zero_based = mid // mb_size_bp
        frac = (mid - anchor_zero_based * mb_size_bp) / float(mb_size_bp)
        anchor_idx1 = int(anchor_zero_based + 1)
        frac = float(np.clip(frac, 0.0, 0.999999))
        amap[key] = (anchor_idx1, frac)

    return amap

def build_mb_anchor_map_from_logs(mb_folder, backbone_length=None, mb_size_bp=1_000_000, resolution=10_000):
    amap = {}
    if not os.path.isdir(mb_folder):
        print(f"[WARN] MB folder not found: {mb_folder}")
        return amap

    for name in sorted(os.listdir(mb_folder)):
        if not (name.startswith("MB_") and name.endswith(".log")):
            continue
        key = name[:-4]
        log_path = os.path.join(mb_folder, name)
        try:
            meta = json.load(open(log_path))
            s = int(meta.get("effective_start_bp", meta.get("start_bp", 0)))
            e = int(meta.get("effective_end_bp", meta.get("end_bp", s)))
            if e <= s and "n_beads" in meta:
                try:
                    e = s + int(meta["n_beads"]) * int(resolution)
                except Exception:
                    pass
        except Exception as ex:
            print(f"[WARN] Failed reading log {name}: {ex}")
            continue

        if e < s:
            e = s
        mid = (s + e) // 2
        if mid < 0:
            mid = 0
        anchor_zero_based = mid // mb_size_bp
        anchor_idx1 = int(anchor_zero_based + 1)
        seg_start_bp = int(anchor_zero_based * mb_size_bp)
        frac = (mid - seg_start_bp) / float(mb_size_bp)
        frac = float(np.clip(frac, 0.0, 0.999999))

        if backbone_length is not None:
            try:
                anchor_idx1 = max(1, min(int(backbone_length), anchor_idx1))
            except Exception:
                anchor_idx1 = max(1, anchor_idx1)

        amap[key] = (anchor_idx1, frac)

    return amap

def build_mb_anchor_fraction_map_from_meta(mb_folder, mb_size_bp=1_000_000):
    amap = {}
    if not os.path.isdir(mb_folder):
        print(f"[WARN] MB folder not found: {mb_folder}")
        return amap
    for name in sorted(os.listdir(mb_folder)):
        if not (name.startswith("MB_") and name.endswith("_meta.json")):
            continue
        key = name[:-10]
        try:
            meta = json.load(open(os.path.join(mb_folder, name)))
            s = int(meta.get('start_bp', 0))
            e = int(meta.get('end_bp', s))
            if e < s:
                e = s
            seg = int(s // mb_size_bp) + 1
            mid = (s + e) // 2
            seg_start_bp = int((seg - 1) * mb_size_bp)
            frac = (mid - seg_start_bp) / float(mb_size_bp)
            frac = float(np.clip(frac, 0.0, 0.999999))
            amap[key] = (seg, frac)
        except Exception as ex:
            print(f"[WARN] Failed reading raw meta for {key}: {ex}")
            continue
    return amap

def load_backbone_coordinate_mapping(mapping_txt_path):
    if not os.path.isfile(mapping_txt_path):
        raise FileNotFoundError(f"Coordinate mapping file not found: {mapping_txt_path}")
    df = pd.read_csv(mapping_txt_path, sep=r"\s+", header=None, names=["bp", "idx0"], engine="python")
    rows = []
    for _, r in df.iterrows():
        try:
            bp = int(r["bp"])
            idx0 = int(r["idx0"])
            rows.append((bp, idx0 + 1))
        except Exception:
            continue
    rows.sort(key=lambda t: t[0])
    return rows

def build_backbone_intervals_from_mapping(mapping_txt_path):
    rows = load_backbone_coordinate_mapping(mapping_txt_path)
    intervals = []
    for i in range(len(rows)):
        s_bp, a1 = rows[i]
        e_bp = rows[i + 1][0] if i + 1 < len(rows) else None
        intervals.append({"start_bp": int(s_bp), "end_bp": (int(e_bp) if e_bp is not None else None), "anchor_idx1": int(a1)})
    return intervals

def build_mb_anchor_map_from_coordinate_mapping(mb_folder, mapping_txt_path, use_raw_first=True):
    amap = {}
    if not os.path.isdir(mb_folder):
        print(f"[WARN] MB folder not found: {mb_folder}")
        return amap

    intervals = build_backbone_intervals_from_mapping(mapping_txt_path)

    def _overlaps(a_s, a_e, b_s, b_e):
        if a_e is None:
            return (b_e is None) or (b_e > a_s)
        if b_e is None:
            return a_e > b_s
        return (a_s < b_e) and (b_s < a_e)

    meta_files = [n for n in sorted(os.listdir(mb_folder)) if n.startswith("MB_") and n.endswith("_meta.json")]
    for name in meta_files:
        key = name[:-10]
        meta_path = os.path.join(mb_folder, name)
        try:
            meta = json.load(open(meta_path))
        except Exception as e:
            print(f"[WARN] Failed reading meta {name}: {e}")
            continue

        raw_s = int(meta.get("start_bp", 0)) if "start_bp" in meta else None
        raw_e = int(meta.get("end_bp", raw_s if raw_s is not None else 0)) if "end_bp" in meta else None
        eff_s = int(meta.get("effective_start_bp", raw_s if raw_s is not None else 0))
        eff_e = int(meta.get("effective_end_bp", eff_s))

        if use_raw_first and (raw_s is not None and raw_e is not None and not (raw_s == 0 and raw_e == 0)):
            s_bp, e_bp = raw_s, raw_e
        else:
            s_bp, e_bp = eff_s, eff_e

        if e_bp < s_bp:
            e_bp = s_bp

        chosen = None
        for iv in intervals:
            S = iv["start_bp"]
            E = iv["end_bp"]
            if _overlaps(S, E, s_bp, e_bp):
                chosen = iv
                break
        if chosen is None:
            continue

        a1 = int(chosen["anchor_idx1"])
        amap[key] = (a1, 0.0)

    return amap

def build_mb_anchor_fraction_map_from_bins(mb_bins_map, resolution=10_000, mb_size_bp=1_000_000):
    amap = {}
    if not isinstance(mb_bins_map, dict) or not mb_bins_map:
        return amap

    bins_per_mb = int(round(mb_size_bp / resolution))
    if bins_per_mb <= 0:
        bins_per_mb = 100

    for key, bins in mb_bins_map.items():
        if not bins:
            continue
        bmin = int(min(bins))
        bmax = int(max(bins))
        seg = int((bmin - 1) // bins_per_mb) + 1
        mid_bin = int(round((bmin + bmax) / 2.0))
        mid_bp = int((mid_bin - 1) * resolution + resolution / 2.0)
        seg_start_bp = int((seg - 1) * mb_size_bp)
        frac = (mid_bp - seg_start_bp) / float(mb_size_bp)
        frac = float(np.clip(frac, 0.0, 0.999999))
        amap[key] = (seg, frac)

    return amap

def build_mb_anchor_fraction_map_from_meta(mb_folder, mb_size_bp=1_000_000):
    amap = {}
    if not os.path.isdir(mb_folder):
        print(f"[WARN] MB folder not found: {mb_folder}")
        return amap

    for name in sorted(os.listdir(mb_folder)):
        if not (name.startswith("MB_") and name.endswith("_meta.json")):
            continue
        base = name[:-10]
        key = base
        meta_path = os.path.join(mb_folder, name)
        try:
            meta = json.load(open(meta_path))
            eff_s = int(meta.get('effective_start_bp', 0))
            eff_e = int(meta.get('effective_end_bp', eff_s))
            left_boundary = eff_s
            mid = (eff_s + eff_e) // 2
        except Exception as e:
            print(f"[WARN] Failed reading meta {name}: {e}. Skipping.")
            continue

        if left_boundary < 0:
            left_boundary = 0
        seg = int(left_boundary // mb_size_bp) + 1

        seg_start_bp = (seg - 1) * mb_size_bp
        frac = (mid - seg_start_bp) / float(mb_size_bp)
        frac = float(np.clip(frac, 0.0, 0.999999))
        amap[key] = (seg, frac)

    return amap
