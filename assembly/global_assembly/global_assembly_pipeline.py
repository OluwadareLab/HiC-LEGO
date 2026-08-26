import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.interpolate import CubicSpline
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from itertools import islice
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

def _json_default_encoder(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'global_assembly_partial_parallel_work'))

try:
    from .utils2 import (
        construct_obj_from_hic,
        construct_flamingo_prediction_from_pdb,
        process_mb_structures,
        build_mb_bins_map_from_logs,
        build_mb_anchor_map_from_logs,
        if2pd,
        rotation_matrix,
        rotate,
    )
except Exception:
    try:
        from utils2 import (
            construct_obj_from_hic,
            construct_flamingo_prediction_from_pdb,
            process_mb_structures,
            build_mb_bins_map_from_logs,
            build_mb_anchor_map_from_logs,
            if2pd,
            rotation_matrix,
            rotate,
        )
    except ImportError:
        print("[WARN] Could not import from utils2, defining minimal versions")

class GlobalAssemblyPipeline:

    def __init__(self, mb_dir, backbone_path, hic_path, chr_name, resolution,
                 output_dir=None, verbose=True, num_threads=30, coordinate_mapping_path=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mb_dir = mb_dir
        self.backbone_path = backbone_path
        self.hic_path = hic_path
        self.chr_name = chr_name
        self.resolution = resolution
        self.output_dir = output_dir or os.path.join(current_dir, 'outputs')
        self.verbose = verbose
        self.quiet_log = True
        self.num_threads = num_threads
        self.coordinate_mapping_path = coordinate_mapping_path

        self.registry_lock = Lock()

        self.G0 = 0
        self.N = None
        self.mb_structures = {}
        self.mb_info = []
        self.backbone_coords = None
        self.hic_matrix = None
        self.hic_sparse = None
        self.valid_bin_starts = None
        self.bp_to_global_idx = {}

        self.owner = None
        self.local_index = None
        self.structure_id = None

        self.P_inter = []
        self.P_intra = {}

        self.stage_c_pruning_profile = 'auto'
        self.stage_c_inter_mode = 'auto'
        self.stage_c_inter_topk_adj = None
        self.stage_c_inter_topfrac_adj = None
        self.stage_c_inter_minkeep_adj = None
        self.stage_c_inter_topk_skip1 = None
        self.stage_c_inter_topfrac_skip1 = None
        self.stage_c_inter_minkeep_skip1 = None
        self.stage_c_intra_max_contacts = None
        self.stage_c_strata_bp = (2_000_000, 10_000_000)
        self.stage_c_strata_ratios = (0.45, 0.35, 0.20)
        self.stage_c_strata_min_keep = 40

        self.alpha_inter = None
        self.scale_inter = 1.0
        self.alpha_global = None

        self.dscc_eval_universe = 'contacts'
        self.dscc_all_pairs_zero_policy = 'epsilon'
        self.dscc_all_pairs_if_epsilon = 1e-6
        self.dscc_all_pairs_max_pairs = None

        self.X_initial = None
        self.X_rotated = None
        self.X_stitched = None
        self.X_refined = None

        self.diagnostics = {}

        self.enable_boundary_alignment = True
        self.boundary_alignment_mode = 'sequential'
        self.boundary_alignment_window = 5
        self.boundary_alignment_trigger_sigma = 6.0
        self.boundary_alignment_strength = 1.0

        self.enable_tail_gap_refinement = False
        self.tail_gap_refine_max_iter = 200
        self.tail_gap_refine_lr = 0.03
        self.tail_gap_refine_topk_per_bin = 24
        self.tail_gap_refine_lambda_contact = 1.0
        self.tail_gap_refine_lambda_smooth = 0.10
        self.tail_gap_refine_lambda_bond = 0.20
        self.tail_gap_refine_lambda_reg = 0.02
        self.tail_gap_refine_huber_delta = 1.0

        self.collapse_terminal_tail_gaps = True

        self.stage_i_rigid_refinement = True
        self.stage_i_rigid_lambda_inter = 1.0
        self.stage_i_rigid_max_sweeps = 8
        self.stage_i_rigid_mu_translation = 0.12
        self.stage_i_rigid_per_structure_maxiter = 20

        self._t_bins_by_structure = None
        self._t_start_end_by_structure = None
        self._bin_to_structure = None

        os.makedirs(self.output_dir, exist_ok=True)

    def _suggest_workers(self, n_tasks=None):

        n = int(self.num_threads) if self.num_threads else 1
        if n_tasks is None:
            return max(1, n)
        return max(1, min(n, int(n_tasks)))

    def _parallel_map(self, fn, iterable, n_tasks=None, chunksize=None):
        workers = self._suggest_workers(n_tasks=n_tasks)
        if workers <= 1:
            return list(map(fn, iterable))

        if chunksize is None:
            chunksize = 10

        with ThreadPoolExecutor(max_workers=workers) as executor:
            try:
                return list(executor.map(fn, iterable, chunksize=chunksize))
            except TypeError:
                return list(executor.map(fn, iterable))

    def _resolve_stage_c_pruning_config(self):
        profile = str(getattr(self, 'stage_c_pruning_profile', 'auto')).lower().replace('_', '-')
        inter_mode = str(getattr(self, 'stage_c_inter_mode', 'auto')).lower().replace('_', '-')

        cfg = {
            'profile': profile,
            'inter_mode': 'topk',
            'adj_top_k': 2000,
            'adj_top_frac': 0.05,
            'adj_min_keep': 100,
            'skip1_top_k': 1000,
            'skip1_top_frac': 0.03,
            'skip1_min_keep': 50,
            'intra_max_contacts': 1000,
            'strata_bp': (2_000_000, 10_000_000),
            'strata_ratios': (0.45, 0.35, 0.20),
            'strata_min_keep': 40,
        }

        if profile == 'practical-5kb' or (profile == 'auto' and int(self.resolution) <= 5000):
            cfg.update({
                'inter_mode': 'distance-stratified',
                'adj_top_k': 8000,
                'adj_top_frac': 0.15,
                'adj_min_keep': 300,
                'skip1_top_k': 3500,
                'skip1_top_frac': 0.08,
                'skip1_min_keep': 120,
                'intra_max_contacts': 5000,
                'strata_bp': (2_000_000, 10_000_000),
                'strata_ratios': (0.45, 0.35, 0.20),
                'strata_min_keep': 60,
            })

        if inter_mode in ('topk', 'distance-stratified'):
            cfg['inter_mode'] = inter_mode

        overrides = {
            'adj_top_k': getattr(self, 'stage_c_inter_topk_adj', None),
            'adj_top_frac': getattr(self, 'stage_c_inter_topfrac_adj', None),
            'adj_min_keep': getattr(self, 'stage_c_inter_minkeep_adj', None),
            'skip1_top_k': getattr(self, 'stage_c_inter_topk_skip1', None),
            'skip1_top_frac': getattr(self, 'stage_c_inter_topfrac_skip1', None),
            'skip1_min_keep': getattr(self, 'stage_c_inter_minkeep_skip1', None),
            'intra_max_contacts': getattr(self, 'stage_c_intra_max_contacts', None),
            'strata_min_keep': getattr(self, 'stage_c_strata_min_keep', None),
        }
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

        if getattr(self, 'stage_c_strata_bp', None) is not None:
            cfg['strata_bp'] = tuple(self.stage_c_strata_bp)
        if getattr(self, 'stage_c_strata_ratios', None) is not None:
            cfg['strata_ratios'] = tuple(self.stage_c_strata_ratios)

        cfg['adj_top_k'] = int(max(1, cfg['adj_top_k']))
        cfg['skip1_top_k'] = int(max(1, cfg['skip1_top_k']))
        cfg['adj_top_frac'] = float(np.clip(cfg['adj_top_frac'], 0.0, 1.0))
        cfg['skip1_top_frac'] = float(np.clip(cfg['skip1_top_frac'], 0.0, 1.0))
        cfg['adj_min_keep'] = int(max(1, cfg['adj_min_keep']))
        cfg['skip1_min_keep'] = int(max(1, cfg['skip1_min_keep']))
        cfg['intra_max_contacts'] = int(max(1, cfg['intra_max_contacts']))
        cfg['strata_min_keep'] = int(max(1, cfg['strata_min_keep']))

        bp_thresholds = [int(max(1, b)) for b in cfg['strata_bp']]
        bp_thresholds = sorted(bp_thresholds)
        cfg['strata_bp'] = tuple(bp_thresholds)
        cfg['strata_bins'] = tuple(max(1, int(round(b / float(self.resolution)))) for b in bp_thresholds)

        ratios = np.array(cfg['strata_ratios'], dtype=float)
        if ratios.size != 3 or not np.all(np.isfinite(ratios)) or float(np.sum(ratios)) <= 0:
            ratios = np.array([0.45, 0.35, 0.20], dtype=float)
        ratios = np.clip(ratios, 0.0, None)
        ratios = ratios / np.sum(ratios)
        cfg['strata_ratios'] = tuple(float(x) for x in ratios)

        return cfg

    def _prune_inter_contacts(self, contacts, pair_kind, cfg):
        if not contacts:
            return []

        if pair_kind == 'adjacent':
            top_k = cfg['adj_top_k']
            top_frac = cfg['adj_top_frac']
            min_keep = cfg['adj_min_keep']
        else:
            top_k = cfg['skip1_top_k']
            top_frac = cfg['skip1_top_frac']
            min_keep = cfg['skip1_min_keep']

        keep_n = min(int(top_k), max(int(min_keep), int(len(contacts) * float(top_frac))))
        keep_n = min(len(contacts), max(1, keep_n))

        if cfg['inter_mode'] == 'distance-stratified':
            return self._distance_stratified_select(contacts, keep_n, cfg)

        contacts_sorted = sorted(contacts, key=lambda x: x[2], reverse=True)
        return contacts_sorted[:keep_n]

    def _distance_stratified_select(self, contacts, keep_n, cfg):
        if keep_n >= len(contacts):
            return contacts

        b1, b2 = cfg['strata_bins']
        near, mid, far = [], [], []
        for u, v, if_uv in contacts:
            d = abs(int(v) - int(u))
            if d <= b1:
                near.append((u, v, if_uv))
            elif d <= b2:
                mid.append((u, v, if_uv))
            else:
                far.append((u, v, if_uv))

        strata = [near, mid, far]
        for bucket in strata:
            bucket.sort(key=lambda x: x[2], reverse=True)

        ratios = cfg['strata_ratios']
        quotas = [int(round(keep_n * r)) for r in ratios]
        while sum(quotas) < keep_n:
            quotas[0] += 1
        while sum(quotas) > keep_n:
            idx = int(np.argmax(quotas))
            quotas[idx] = max(0, quotas[idx] - 1)

        min_per = int(cfg['strata_min_keep'])
        selected = []

        for i, bucket in enumerate(strata):
            if not bucket:
                continue
            target = max(quotas[i], min_per)
            take_n = min(len(bucket), target)
            selected.extend(bucket[:take_n])

        if len(selected) > keep_n:
            selected.sort(key=lambda x: x[2], reverse=True)
            return selected[:keep_n]

        if len(selected) < keep_n:
            selected_set = set(selected)
            leftovers = [c for c in sorted(contacts, key=lambda x: x[2], reverse=True) if c not in selected_set]
            need = keep_n - len(selected)
            selected.extend(leftovers[:need])

        return selected

    def _build_structure_bin_cache(self):
        if self.structure_id is None or self.N is None:
            return

        n_structures = len(self.mb_info)
        t_bins = [[] for _ in range(n_structures)]
        t_start_end = [(None, None) for _ in range(n_structures)]

        for t in range(self.N):
            s = int(self.structure_id[t])
            if s >= 0:
                t_bins[s].append(t)
                st, en = t_start_end[s]
                if st is None:
                    t_start_end[s] = (t, t)
                else:
                    t_start_end[s] = (st, t)

        self._t_bins_by_structure = t_bins
        self._t_start_end_by_structure = t_start_end

    def _build_contact_adjacency(self, contacts):

        adj = [[] for _ in range(self.N)]
        eps = 1e-6
        for u, v, if_uv in contacts:
            if if_uv <= 0:
                continue
            w = if_uv ** 0.5
            wish = np.power(if_uv + eps, -self.alpha_global)
            if 0 <= u < self.N and 0 <= v < self.N:
                adj[u].append((v, wish, w))
                adj[v].append((u, wish, w))
        return adj

    def _compute_contact_density(self):

        contact_count = np.zeros(self.N, dtype=int)

        for u, v, _ in self.P_inter:
            if 0 <= u < self.N:
                contact_count[u] += 1
            if 0 <= v < self.N:
                contact_count[v] += 1

        for s, contacts in self.P_intra.items():
            for u, v, _ in contacts:
                if 0 <= u < self.N:
                    contact_count[u] += 1
                if 0 <= v < self.N:
                    contact_count[v] += 1

        if np.max(contact_count) > 0:
            contact_density = contact_count.astype(float) / np.max(contact_count)
        else:
            contact_density = np.zeros(self.N)

        return contact_density, contact_count

    def _has_hic_contact(self, u, v):
        if self.hic_sparse is None:
            return False
        return self.hic_sparse[u, v] > 0 or self.hic_sparse[v, u] > 0

    def log(self, msg, level="INFO"):

        if not self.verbose:
            return
        text = "" if msg is None else str(msg)
        if getattr(self, "quiet_log", True):
            m = text.strip()
            keep_prefixes = (
                "STAGE ",
                "PIPELINE COMPLETE",
                "PDB saved:",
                "CSV saved:",
                "Coordinate mapping file saved:",
                "Diagnostics saved:",
                "Output paths:",
            )
            if not any(m.startswith(p) for p in keep_prefixes):
                return
            text = m
        print(f"[{level}] {text}", flush=True)

    def run(self):
        self.log("="*80)
        self.log("GLOBAL CHROMOSOME ASSEMBLY PIPELINE")
        self.log("="*80)

        self.log("\n" + "="*80)
        self.log("STAGE A: Global Bin Registry Setup")
        self.log("="*80)
        self.stage_A_global_bin_registry()

        self.log("\n" + "="*80)
        self.log("STAGE B: Backbone Anchoring")
        self.log("="*80)
        self.stage_B_backbone_anchoring()

        self.log("\n" + "="*80)
        self.log("STAGE C: Inter-Contact Set Selection")
        self.log("="*80)
        self.stage_C_inter_contact_selection()

        self.log("\n" + "="*80)
        self.log("STAGE D: Alpha Optimization")
        self.log("="*80)
        self._quick_placement_for_alpha()
        self.stage_D_alpha_strategy()

        self.log("\n" + "="*80)
        self.log("STAGE E: Scaling")
        self.log("="*80)
        self.stage_E_scaling()

        self.log("\n" + "="*80)
        self.log("STAGE F: Initial Rigid Placement (Translation)")
        self.log("="*80)
        self.stage_F_initial_placement()

        self.log("\n" + "="*80)
        self.log("STAGE G: Rigid Rotation Assembly")
        self.log("="*80)
        self.stage_G_rigid_rotation()

        self.log("\n" + "="*80)
        self.log("STAGE H: Stitching into Global Array")
        self.log("="*80)
        self.stage_H_stitching()

        self.log("\n" + "="*80)
        self.log("STAGE I: Global Refinement")
        self.log("="*80)
        self.stage_I_global_refinement()

        self.log("\n" + "="*80)
        self.log("STAGE J: Final Diagnostics and Output")
        self.log("="*80)
        self.stage_J_diagnostics_and_save()

        self.log("\n" + "="*80)
        self.log("PIPELINE COMPLETE")
        self.log("="*80)

        return self.X_refined

    def stage_A_global_bin_registry(self):

        self.log("Loading Hi-C matrix (3-column format: pos_i, pos_j, IF)...")
        self.hic_matrix = np.loadtxt(self.hic_path)

        if self.hic_matrix.shape[1] != 3:
            raise ValueError(f"Hi-C matrix must be 3-column format (pos_i, pos_j, IF), "
                           f"but got {self.hic_matrix.shape[1]} columns")

        finite_mask = np.all(np.isfinite(self.hic_matrix), axis=1)
        if np.sum(~finite_mask) > 0:
            self.log(f"Filtering {np.sum(~finite_mask)} invalid Hi-C entries (NaN/Inf)", "WARN")
            self.hic_matrix = self.hic_matrix[finite_mask]

        if np.any(self.hic_matrix[:, 2] < 0):
            n_negative = np.sum(self.hic_matrix[:, 2] < 0)
            self.log(f"Warning: {n_negative} Hi-C entries have negative IF values", "WARN")

        pos_i_raw = self.hic_matrix[:, 0].astype(np.int64)
        pos_j_raw = self.hic_matrix[:, 1].astype(np.int64)

        valid_bins_source = 'hic_unique_positions'
        valid_bins = None
        if self.coordinate_mapping_path:
            if not os.path.exists(self.coordinate_mapping_path):
                raise FileNotFoundError(f"Coordinate mapping not found: {self.coordinate_mapping_path}")

            mapped = []
            with open(self.coordinate_mapping_path, 'r') as f:
                for line in f:
                    if not line.strip() or line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) < 1:
                        continue
                    try:
                        mapped.append(int(parts[0]))
                    except Exception:
                        continue
            if len(mapped) == 0:
                raise RuntimeError(
                    f"No valid bins found in coordinate mapping: {self.coordinate_mapping_path}"
                )
            valid_bins = np.array(sorted(set(mapped)), dtype=np.int64)
            valid_bins_source = 'coordinate_mapping'
        else:
            valid_bins = np.array(sorted(set(np.concatenate([pos_i_raw, pos_j_raw]).tolist())), dtype=np.int64)

        self.valid_bin_starts = valid_bins
        self.N = int(len(valid_bins))
        self.G0 = int(valid_bins[0])
        max_pos = int(valid_bins[-1])
        self.bp_to_global_idx = {int(bp): i for i, bp in enumerate(valid_bins.tolist())}

        self.log(f"Hi-C matrix loaded: {len(self.hic_matrix):,} contacts")
        self.log(f"Valid-bin source: {valid_bins_source}")
        self.log(f"Valid-bin genomic range: {self.G0:,} - {max_pos:,} bp")
        self.log(f"Total valid bins N = {self.N:,} (resolution = {self.resolution:,} bp)")
        self.log(f"IF range: [{np.min(self.hic_matrix[:, 2]):.2f}, {np.max(self.hic_matrix[:, 2]):.2f}]")

        self.log("Loading MB structures and logs...")
        self.mb_structures = process_mb_structures(self.mb_dir)
        self.log(f"Found {len(self.mb_structures)} MB structures")

        log_files = sorted(Path(self.mb_dir).glob("MB_*.log"))
        self.mb_info = []

        def parse_log_file(log_path):
            key = log_path.stem
            if key not in self.mb_structures:
                return None
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                info = {
                    'key': key,
                    'mb_index': log_data.get('mb_index', 0),
                    'A_s': log_data.get('effective_start_bp', 0),
                    'B_s': log_data.get('effective_end_bp', 0),
                    'n_beads': log_data.get('n_beads', len(self.mb_structures[key]['coordinates'])),
                    'log_path': str(log_path)
                }
                coords = self.mb_structures[key]['coordinates']
                info['n_pdb'] = len(coords)
                info['coords'] = coords
                return info
            except Exception as e:
                if self.verbose:
                    self.log(f"Failed to parse {log_path}: {e}", "WARN")
                return None

        results = self._parallel_map(parse_log_file, log_files, n_tasks=len(log_files), chunksize=5)
        self.mb_info = [info for info in results if info is not None]

        self.mb_info.sort(key=lambda x: x['mb_index'])
        self.log(f"Loaded {len(self.mb_info)} MB structures with logs")

        self.log("Converting MB regions to global bin indices...")

        def convert_mb_region(args):
            s, info = args
            A_s = info['A_s']
            B_s = info['B_s']

            n_pdb = info['n_pdb']

            bps = self.valid_bin_starts
            idx_exclusive = np.where((bps >= A_s) & (bps < B_s))[0]
            idx_inclusive = np.where((bps >= A_s) & (bps <= B_s))[0]
            idx_right_inclusive = np.where((bps > A_s) & (bps <= B_s))[0]

            candidates = [
                ('exclusive', idx_exclusive),
                ('inclusive', idx_inclusive),
                ('right-inclusive', idx_right_inclusive),
            ]
            chosen_mode, chosen_idx = min(
                candidates,
                key=lambda x: (abs(len(x[1]) - n_pdb), 0 if x[0] == 'exclusive' else 1)
            )

            if self.verbose and len(chosen_idx) != n_pdb:
                self.log(
                    f"MB {info['key']}: n_pdb={n_pdb}, matched_bins={len(chosen_idx)} "
                    f"(mode={chosen_mode}, A={A_s}, B={B_s})",
                    "WARN"
                )

            if len(chosen_idx) > 0:
                t_start = int(chosen_idx[0])
                t_end = int(chosen_idx[-1])
                span = int(t_end - t_start)
                mid_t = 0.5 * (t_start + t_end)
            else:
                t_start = -1
                t_end = -1
                span = 0
                mid_t = -1.0

            return {
                's': s,
                'key': info['key'],
                't_start': t_start,
                't_end': t_end,
                'n_expected': int(len(chosen_idx)),
                'n_pdb': n_pdb,
                'span': span,
                'mid_t': float(mid_t),
                't_bins': chosen_idx.tolist(),
            }

        mb_bin_mappings = self._parallel_map(convert_mb_region, list(enumerate(self.mb_info)), n_tasks=len(self.mb_info), chunksize=10)

        self.log("Building global bin registry...")
        self.owner = np.full(self.N, None, dtype=object)
        self.local_index = np.full(self.N, -1, dtype=int)
        self.structure_id = np.full(self.N, -1, dtype=int)

        overlaps = []

        for mapping in mb_bin_mappings:
            s = mapping['s']
            t_start = mapping['t_start']
            t_end = mapping['t_end']
            t_bins = mapping['t_bins']

            for k, t in enumerate(t_bins):
                if t < 0 or t >= self.N:
                    continue

                if self.owner[t] is not None:
                    existing_s = self.structure_id[t]
                    existing_span = mb_bin_mappings[existing_s]['span']
                    current_span = mapping['span']

                    if current_span < existing_span:
                        overlaps.append((t, existing_s, s, 'span'))
                        self.owner[t] = mapping['key']
                        self.local_index[t] = k
                        self.structure_id[t] = s
                    elif current_span > existing_span:
                        overlaps.append((t, s, existing_s, 'span'))
                        continue
                    else:
                        existing_mid = mb_bin_mappings[existing_s]['mid_t']
                        current_mid = mapping['mid_t']
                        if abs(t - current_mid) < abs(t - existing_mid):
                            overlaps.append((t, existing_s, s, 'midpoint'))
                            self.owner[t] = mapping['key']
                            self.local_index[t] = k
                            self.structure_id[t] = s
                        else:
                            overlaps.append((t, s, existing_s, 'midpoint'))
                            continue
                else:
                    self.owner[t] = mapping['key']
                    self.local_index[t] = k
                    self.structure_id[t] = s

        self._build_structure_bin_cache()

        gap_bins = np.where(self.owner == None)[0]
        self.log(f"Bin registry complete:")
        self.log(f"  Covered bins: {np.sum(self.owner != None):,} / {self.N:,}")
        self.log(f"  Gap bins: {len(gap_bins):,}")
        self.log(f"  Overlaps: {len(overlaps)}")

        self.diagnostics['stage_A'] = {
            'N': int(self.N),
            'G0': int(self.G0),
            'valid_bin_source': valid_bins_source,
            'n_structures': len(self.mb_info),
            'covered_bins': int(np.sum(self.owner != None)),
            'gap_bins': int(len(gap_bins)),
            'overlaps': len(overlaps),
            'overlap_details': overlaps[:100]
        }

        return mb_bin_mappings

    def stage_B_backbone_anchoring(self):

        self.log("Loading backbone structure...")
        backbone_data = construct_flamingo_prediction_from_pdb(self.backbone_path)
        self.backbone_coords = np.array(backbone_data['coordinates'])
        n_backbone = len(self.backbone_coords)
        self.log(f"Backbone loaded: {n_backbone} beads (1Mb resolution)")

        self.mb_anchors = {}

        def compute_anchor(args):
            s, info = args
            A_s = info['A_s']
            B_s = info['B_s']

            mid_bp = (A_s + B_s) / 2

            bp_from_start = mid_bp - self.G0
            b_float = bp_from_start / 1_000_000
            b_center = b_float - 0.5
            b = int(round(b_center))
            b = max(0, min(b, len(self.backbone_coords) - 1))

            span_mb = (B_s - A_s) / 1_000_000
            log_msg = None
            if span_mb > 1.5:
                log_msg = f"MB {info['key']} (s={s}): spans {span_mb:.2f} Mb " \
                         f"({A_s:,} - {B_s:,} bp), center at {mid_bp:,} bp -> backbone bead {b}"
            else:
                log_msg = f"MB {info['key']} (s={s}): midpoint={mid_bp:,} bp -> backbone bead {b}"

            return s, b, log_msg

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(compute_anchor, enumerate(self.mb_info)))
            for s, b, log_msg in results:
                self.mb_anchors[s] = b
                if self.verbose and log_msg:
                    self.log(log_msg)

        self.diagnostics['stage_B'] = {
            'n_backbone': int(n_backbone),
            'anchors': {s: int(b) for s, b in self.mb_anchors.items()},
            'anchor_details': {
                s: {
                    'mb_key': self.mb_info[s]['key'],
                    'start_bp': int(self.mb_info[s]['A_s']),
                    'end_bp': int(self.mb_info[s]['B_s']),
                    'mid_bp': int((self.mb_info[s]['A_s'] + self.mb_info[s]['B_s']) / 2),
                    'backbone_bead': int(b),
                    'span_mb': float((self.mb_info[s]['B_s'] - self.mb_info[s]['A_s']) / 1_000_000)
                }
                for s, b in self.mb_anchors.items()
            }
        }

    def stage_C_inter_contact_selection(self):

        self.log("Building inter-contact set P_inter...")

        pos_i = self.hic_matrix[:, 0]
        pos_j = self.hic_matrix[:, 1]
        if_vals = self.hic_matrix[:, 2]

        if self.valid_bin_starts is None or len(self.valid_bin_starts) == 0:
            raise RuntimeError("Valid-bin registry is not initialized. Stage A must run first.")

        valid_bins = self.valid_bin_starts

        def _map_bp_to_compact_idx(pos_bp):
            pos_bp = pos_bp.astype(np.int64)
            idx = np.searchsorted(valid_bins, pos_bp)
            in_bounds = idx < len(valid_bins)
            idx_safe = np.clip(idx, 0, len(valid_bins) - 1)
            exact = in_bounds & (valid_bins[idx_safe] == pos_bp)
            out = np.full(len(pos_bp), -1, dtype=int)
            out[exact] = idx[exact].astype(int)
            return out

        i_ind = _map_bp_to_compact_idx(pos_i)
        j_ind = _map_bp_to_compact_idx(pos_j)

        valid_mask = (i_ind >= 0) & (i_ind < self.N) & \
                     (j_ind >= 0) & (j_ind < self.N) & \
                     (if_vals > 0) & np.isfinite(if_vals)

        n_valid = np.sum(valid_mask)
        n_total = len(if_vals)
        if n_valid < n_total:
            self.log(f"Filtered Hi-C contacts: {n_total:,} -> {n_valid:,} valid "
                    f"({100*n_valid/n_total:.1f}%)")

        i_ind = i_ind[valid_mask]
        j_ind = j_ind[valid_mask]
        if_vals = if_vals[valid_mask]

        self.hic_sparse = coo_matrix((if_vals, (i_ind, j_ind)), shape=(self.N, self.N))

        self.hic_sparse = self.hic_sparse.tocsr()
        self.hic_sparse = self.hic_sparse.maximum(self.hic_sparse.T)

        self.log(f"Sparse Hi-C matrix: {self.hic_sparse.nnz:,} non-zero entries "
                f"({100*self.hic_sparse.nnz/(self.N*self.N):.4f}% density)")

        if self._t_start_end_by_structure is None or self._t_bins_by_structure is None:
            self._build_structure_bin_cache()

        stage_c_cfg = self._resolve_stage_c_pruning_config()
        self.log(
            "Stage C pruning config: "
            f"profile={stage_c_cfg['profile']}, mode={stage_c_cfg['inter_mode']}, "
            f"adj(top_k={stage_c_cfg['adj_top_k']}, top_frac={stage_c_cfg['adj_top_frac']:.3f}, min={stage_c_cfg['adj_min_keep']}), "
            f"skip1(top_k={stage_c_cfg['skip1_top_k']}, top_frac={stage_c_cfg['skip1_top_frac']:.3f}, min={stage_c_cfg['skip1_min_keep']}), "
            f"intra_max={stage_c_cfg['intra_max_contacts']}"
        )

        def collect_inter_contacts(s):
            s_next = s + 1
            contacts_all = []

            t_start_s, t_end_s = self._t_start_end_by_structure[s]
            t_start_next, t_end_next = self._t_start_end_by_structure[s_next]
            if t_start_s is None or t_start_next is None:
                return []

            contacts = []
            submatrix = self.hic_sparse[t_start_s:t_end_s+1, t_start_next:t_end_next+1].toarray()
            for i, t_u in enumerate(range(t_start_s, t_end_s + 1)):
                for j, t_v in enumerate(range(t_start_next, t_end_next + 1)):
                    if_u = submatrix[i, j]
                    if if_u > 0:
                        contacts.append((t_u, t_v, if_u))

            if len(contacts) > 0:
                contacts = self._prune_inter_contacts(contacts, pair_kind='adjacent', cfg=stage_c_cfg)
                contacts_all.extend(contacts)
                if self.verbose:
                    self.log(f"Structure pair ({s}, {s+1}): {len(contacts)} inter contacts")

            if s + 2 < len(self.mb_info):
                s_next2 = s + 2

                t_start_s2, t_end_s2 = self._t_start_end_by_structure[s]
                t_start_next2, t_end_next2 = self._t_start_end_by_structure[s_next2]

                if t_start_s2 is not None and t_start_next2 is not None:
                    contacts2 = []
                    submatrix2 = self.hic_sparse[t_start_s2:t_end_s2+1, t_start_next2:t_end_next2+1].toarray()
                    for i, t_u in enumerate(range(t_start_s2, t_end_s2 + 1)):
                        for j, t_v in enumerate(range(t_start_next2, t_end_next2 + 1)):
                            if_u = submatrix2[i, j]
                            if if_u > 0:
                                contacts2.append((t_u, t_v, if_u))

                    if len(contacts2) > 0:
                        contacts2 = self._prune_inter_contacts(contacts2, pair_kind='skip1', cfg=stage_c_cfg)
                        contacts_all.extend(contacts2)
                        if self.verbose:
                            self.log(f"Structure pair ({s}, {s+2}): {len(contacts2)} inter contacts")

            return contacts_all

        results = self._parallel_map(
            collect_inter_contacts,
            range(len(self.mb_info) - 1),
            n_tasks=max(0, len(self.mb_info) - 1),
            chunksize=1,
        )
        P_inter_all = []
        for contacts in results:
            P_inter_all.extend(contacts)

        self.P_inter = P_inter_all
        self.log(f"Total inter contacts: {len(self.P_inter):,}")

        def collect_intra_contacts(s):
            t_bins = self._t_bins_by_structure[s] if self._t_bins_by_structure is not None else [t for t in range(self.N) if self.structure_id[t] == s]
            if len(t_bins) < 2:
                return s, []

            contacts = []
            for i, t_u in enumerate(t_bins):
                for t_v in t_bins[i+1:]:
                    if_u = self.hic_sparse[t_u, t_v]
                    if if_u > 0:
                        contacts.append((t_u, t_v, if_u))

            if len(contacts) > stage_c_cfg['intra_max_contacts']:
                contacts.sort(key=lambda x: x[2], reverse=True)
                contacts = contacts[:stage_c_cfg['intra_max_contacts']]

            return s, contacts

        self.P_intra = {}
        results = self._parallel_map(
            collect_intra_contacts,
            range(len(self.mb_info)),
            n_tasks=len(self.mb_info),
            chunksize=1,
        )
        for s, contacts in results:
            if contacts:
                self.P_intra[s] = contacts

        self.diagnostics['stage_C'] = {
            'n_inter_contacts': len(self.P_inter),
            'n_intra_contacts': sum(len(v) for v in self.P_intra.values()),
            'n_structure_pairs': len(self.mb_info) - 1,
            'pruning_config': {
                'profile': stage_c_cfg['profile'],
                'inter_mode': stage_c_cfg['inter_mode'],
                'adj_top_k': int(stage_c_cfg['adj_top_k']),
                'adj_top_frac': float(stage_c_cfg['adj_top_frac']),
                'adj_min_keep': int(stage_c_cfg['adj_min_keep']),
                'skip1_top_k': int(stage_c_cfg['skip1_top_k']),
                'skip1_top_frac': float(stage_c_cfg['skip1_top_frac']),
                'skip1_min_keep': int(stage_c_cfg['skip1_min_keep']),
                'intra_max_contacts': int(stage_c_cfg['intra_max_contacts']),
                'strata_bp': [int(x) for x in stage_c_cfg['strata_bp']],
                'strata_bins': [int(x) for x in stage_c_cfg['strata_bins']],
                'strata_ratios': [float(x) for x in stage_c_cfg['strata_ratios']],
                'strata_min_keep': int(stage_c_cfg['strata_min_keep']),
            }
        }

    def _quick_placement_for_alpha(self):
        self.log("Quick placement for alpha search...")
        X_quick = np.zeros((self.N, 3))

        anchor_positions = {}
        structures_per_anchor = {}

        if self._t_bins_by_structure is None:
            self._build_structure_bin_cache()

        for s, info in enumerate(self.mb_info):
            coords = info['coords']
            c_s = np.mean(coords, axis=0)
            b = self.mb_anchors[s]
            B_s = self.backbone_coords[b]

            anchor_key = tuple(B_s)
            if anchor_key not in anchor_positions:
                anchor_positions[anchor_key] = []
                structures_per_anchor[anchor_key] = 0
            anchor_positions[anchor_key].append(s)
            structures_per_anchor[anchor_key] += 1

            coords_translated = coords + (B_s - c_s)

            t_bins = self._t_bins_by_structure[s] if self._t_bins_by_structure is not None else [t for t in range(self.N) if self.structure_id[t] == s]
            for t in t_bins:
                k = self.local_index[t]
                if k >= 0 and k < len(coords_translated):
                    X_quick[t] = coords_translated[k]

        overlapping_anchors = {k: v for k, v in structures_per_anchor.items() if v > 1}
        if overlapping_anchors:
            self.log(f"WARNING: {len(overlapping_anchors)} backbone anchors have multiple structures. "
                    f"This may cause poor alpha search results.", "WARN")

        gap_bins = np.where(self.owner == None)[0]
        for t in gap_bins:
            known_bins = np.where(self.owner != None)[0]
            if len(known_bins) > 0:
                dists = np.abs(known_bins - t)
                nearest = known_bins[np.argmin(dists)]
                X_quick[t] = X_quick[nearest]

        valid_coords = X_quick[self.owner != None]
        if len(valid_coords) > 0:
            coord_std = np.std(valid_coords, axis=0)
            coord_range = np.ptp(valid_coords, axis=0)
            self.log(f"Quick placement: {len(valid_coords):,} valid coordinates, "
                    f"std = [{coord_std[0]:.2f}, {coord_std[1]:.2f}, {coord_std[2]:.2f}], "
                    f"range = [{coord_range[0]:.2f}, {coord_range[1]:.2f}, {coord_range[2]:.2f}]")

            if np.max(coord_range) < 1.0:
                self.log(f"WARNING: Coordinates are very clustered (max range = {np.max(coord_range):.2f}). "
                        f"This will cause poor dSCC for all alphas.", "WARN")

        self.X_quick_placement = X_quick

    def stage_D_alpha_strategy(self):

        self.log("Grid search for α_inter (inter-only)...")

        if hasattr(self, 'X_quick_placement'):
            X_rough = self.X_quick_placement.copy()
            self.log("Using quick placement coordinates for alpha search")
        elif hasattr(self, 'backbone_coords_scaled'):
            X_rough = self._get_rough_coordinates_scaled()
            self.log("Using scaled backbone coordinates for alpha search")
        else:
            X_rough = self._get_rough_coordinates()
            self.log("Using rough backbone coordinates for alpha search")

        alpha_candidates = np.arange(0.1, 2.1, 0.1)

        self.log(f"Testing {len(alpha_candidates)} alpha values from 0.1 to 2.0 (step 0.1)...")

        P_inter_sample = self.P_inter
        self.log(f"Using all {len(P_inter_sample):,} inter contacts for alpha search")

        def evaluate_alpha(alpha):
            if len(P_inter_sample) > 0:
                test_u, test_v, test_if = P_inter_sample[0]
                test_wish_01 = np.power(test_if + 1e-6, -0.1)
                test_wish_10 = np.power(test_if + 1e-6, -1.0)
                if abs(test_wish_01 - test_wish_10) < 1e-10:
                    pass

            scale_alpha = self._estimate_global_scale_from_contacts(
                X_rough, alpha, P_inter_sample, weights_mode='sqrt_if'
            )
            dscc = self._compute_inter_dscc(X_rough, alpha, P_inter_sample, scale_s=scale_alpha)
            stress = self._compute_inter_stress(
                X_rough, alpha, P_inter_sample, huber_delta=1.0, scale_s=scale_alpha
            )
            score = -stress
            return alpha, dscc, score, scale_alpha, stress

        if len(P_inter_sample) > 0:
            test_u, test_v, test_if = P_inter_sample[0]
            test_wish_01 = np.power(test_if + 1e-6, -0.1)
            test_wish_10 = np.power(test_if + 1e-6, -1.0)
            test_wish_20 = np.power(test_if + 1e-6, -2.0)
            self.log(f"DEBUG: Test IF={test_if:.2f}, wish_dist(α=0.1)={test_wish_01:.6f}, "
                    f"wish_dist(α=1.0)={test_wish_10:.6f}, wish_dist(α=2.0)={test_wish_20:.6f}")

            if len(P_inter_sample) >= 10:
                if_vals = [if_uv for _, _, if_uv in P_inter_sample[:10]]
                wish_01_vals = [np.power(if_uv + 1e-6, -0.1) for if_uv in if_vals]
                wish_10_vals = [np.power(if_uv + 1e-6, -1.0) for if_uv in if_vals]
                wish_01_range = max(wish_01_vals) - min(wish_01_vals)
                wish_10_range = max(wish_10_vals) - min(wish_10_vals)
                self.log(f"DEBUG: Wish distance range for 10 contacts: α=0.1 → {wish_01_range:.6f}, α=1.0 → {wish_10_range:.6f}")

        best_alpha = 0.5
        best_dscc = -np.inf
        best_score = -np.inf
        best_scale = 1.0
        best_stress = np.inf
        all_results = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(evaluate_alpha, alpha_candidates))
            for alpha, dscc, score, scale_alpha, stress in results:
                all_results.append((alpha, dscc, score, scale_alpha, stress))
                if score > best_score:
                    best_score = score
                    best_dscc = dscc
                    best_alpha = alpha
                    best_scale = scale_alpha
                    best_stress = stress

        all_results.sort(key=lambda x: x[2], reverse=True)

        self.log(f"Top 10 alpha candidates:")
        for i, (alpha, dscc, score, scale_alpha, stress) in enumerate(all_results[:10]):
            self.log(
                f"  {i+1}. α = {alpha:.3f}, dSCC = {dscc:.4f}, "
                f"scale = {scale_alpha:.4f}, stress = {stress:.6g}"
            )

        if len(all_results) > 1:
            dscc_values = [dscc for _, dscc, _, _, _ in all_results]
            dscc_range = max(dscc_values) - min(dscc_values)
            dscc_std = np.std(dscc_values)

            if dscc_range < 1e-6:
                self.log(f"ERROR: All alphas have IDENTICAL dSCC ({best_dscc:.6f}). "
                        f"This indicates a critical bug!", "ERROR")

                if hasattr(self, '_dscc_debug_info'):
                    debug = self._dscc_debug_info
                    self.log(f"DEBUG INFO: pred_dist range={debug['pred_range']:.6f}, std={debug['pred_std']:.6f}, "
                            f"wish_dist range={debug['wish_range']:.6f}, std={debug['wish_std']:.6f}", "ERROR")
                    self.log(f"DEBUG: Sample predicted_dists: {[f'{x:.3f}' for x in debug['pred_sample']]}", "ERROR")
                    self.log(f"DEBUG: Sample wish_dists (α={debug['alpha']:.1f}): {[f'{x:.6f}' for x in debug['wish_sample']]}", "ERROR")
                    if 'spearman' in debug and 'pearson' in debug:
                        self.log(f"DEBUG: Spearman={debug['spearman']:.6f}, Pearson={debug['pearson']:.6f}", "ERROR")

                self.log(f"ROOT CAUSE: Spearman correlation is RANK-BASED!", "ERROR")
                self.log(f"  - Higher IF → Lower wish distance (for ANY alpha)", "ERROR")
                self.log(f"  - So rank order of wish distances is IDENTICAL across all alphas!", "ERROR")
                self.log(f"  - If predicted distances also have stable rank order, Spearman is identical!", "ERROR")
                self.log(f"SOLUTION: Use Pearson correlation or a value-based metric instead of Spearman", "ERROR")
            elif dscc_range < 0.01:
                self.log(f"WARNING: All alphas have very similar dSCC (range = {dscc_range:.6f}, std = {dscc_std:.6f}). "
                        f"This suggests coordinate estimates are too poor for reliable alpha selection.", "WARN")

        if best_dscc < 0.1:
            self.log(f"WARNING: Best alpha dSCC is very low ({best_dscc:.4f}). "
                    f"This indicates that coordinate estimates are poor, making alpha selection unreliable. "
                    f"All alphas likely have low dSCC due to poor coordinate alignment.", "WARN")

        self.alpha_inter = best_alpha
        self.scale_inter = float(best_scale)
        self.log(
            f"Best α_inter = {best_alpha:.3f} (dSCC = {best_dscc:.4f}, "
            f"scale_inter = {self.scale_inter:.4f}, stress = {best_stress:.6g})"
        )

        self.diagnostics['stage_D'] = {
            'alpha_inter': float(self.alpha_inter),
            'inter_dscc': float(best_dscc),
            'scale_inter': float(self.scale_inter),
            'inter_stress': float(best_stress)
        }

    def _get_rough_coordinates(self):
        X_rough = np.zeros((self.N, 3))

        if self._t_bins_by_structure is None:
            self._build_structure_bin_cache()

        for s, info in enumerate(self.mb_info):
            b = self.mb_anchors[s]
            anchor_pos = self.backbone_coords[b]

            t_bins = self._t_bins_by_structure[s] if self._t_bins_by_structure is not None else [t for t in range(self.N) if self.structure_id[t] == s]
            for t in t_bins:
                X_rough[t] = anchor_pos

        gap_bins = np.where(self.owner == None)[0]
        for t in gap_bins:
            known_bins = np.where(self.owner != None)[0]
            if len(known_bins) > 0:
                dists = np.abs(known_bins - t)
                nearest = known_bins[np.argmin(dists)]
                X_rough[t] = X_rough[nearest]

        return X_rough

    def _get_rough_coordinates_scaled(self):
        X_rough = np.zeros((self.N, 3))

        if self._t_bins_by_structure is None:
            self._build_structure_bin_cache()

        for s, info in enumerate(self.mb_info):
            b = self.mb_anchors[s]
            anchor_pos = self.backbone_coords_scaled[b]

            t_bins = self._t_bins_by_structure[s] if self._t_bins_by_structure is not None else [t for t in range(self.N) if self.structure_id[t] == s]
            for t in t_bins:
                X_rough[t] = anchor_pos

        gap_bins = np.where(self.owner == None)[0]
        for t in gap_bins:
            known_bins = np.where(self.owner != None)[0]
            if len(known_bins) > 0:
                dists = np.abs(known_bins - t)
                nearest = known_bins[np.argmin(dists)]
                X_rough[t] = X_rough[nearest]

        return X_rough

    def _compute_inter_dscc(self, X, alpha, P_inter, scale_s=1.0):

        if len(P_inter) == 0:
            return 0.0

        predicted_dists = []
        wish_dists = []

        if_vals_sample = []

        for u, v, if_uv in P_inter:
            if u >= len(X) or v >= len(X) or u < 0 or v < 0:
                continue

            pred_dist = np.linalg.norm(X[u] - X[v])

            eps = 1e-6
            if if_uv > 0:
                wish_dist = float(scale_s) * np.power(if_uv + eps, -alpha)
                if len(if_vals_sample) < 10:
                    if_vals_sample.append(if_uv)
            else:
                continue

            if np.isfinite(pred_dist) and np.isfinite(wish_dist) and pred_dist >= 0 and wish_dist > 0:
                predicted_dists.append(pred_dist)
                wish_dists.append(wish_dist)

        if len(predicted_dists) < 10:
            return 0.0

        wish_dists_array = np.array(wish_dists)
        wish_dist_range = np.max(wish_dists_array) - np.min(wish_dists_array)
        wish_dist_std = np.std(wish_dists_array)

        if wish_dist_range < 1e-10 or wish_dist_std < 1e-10:
            return 0.0

        pred_dist_range = np.max(predicted_dists) - np.min(predicted_dists)
        pred_dist_std = np.std(predicted_dists)
        wish_dist_range = np.max(wish_dists_array) - np.min(wish_dists_array)
        wish_dist_std = np.std(wish_dists_array)

        if pred_dist_range < 1e-10 or pred_dist_std < 1e-10:
            return 0.0

        try:
            corr_pearson, p_pearson = pearsonr(predicted_dists, wish_dists)

            corr_spearman, p_spearman = spearmanr(predicted_dists, wish_dists)

            if not hasattr(self, '_dscc_debug_info'):
                self._dscc_debug_info = {
                    'alpha': alpha,
                    'pred_range': pred_dist_range,
                    'pred_std': pred_dist_std,
                    'wish_range': wish_dist_range,
                    'wish_std': wish_dist_std,
                    'n_contacts': len(predicted_dists),
                    'pred_sample': predicted_dists[:5] if len(predicted_dists) >= 5 else predicted_dists,
                    'wish_sample': wish_dists[:5] if len(wish_dists) >= 5 else wish_dists,
                    'spearman': corr_spearman,
                    'pearson': corr_pearson
                }

            return corr_pearson if np.isfinite(corr_pearson) else 0.0
        except Exception as e:
            return 0.0

    def _build_distance_vectors_from_contacts(self, X, alpha, contacts, scale_s=1.0):

        if contacts is None or len(contacts) == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        predicted_dists = []
        wish_dists = []
        eps = 1e-6

        for u, v, if_uv in contacts:
            if if_uv <= 0:
                continue
            if u >= len(X) or v >= len(X) or u < 0 or v < 0:
                continue

            pred_dist = np.linalg.norm(X[u] - X[v])
            wish_dist = float(scale_s) * np.power(float(if_uv) + eps, -alpha)

            if np.isfinite(pred_dist) and np.isfinite(wish_dist) and pred_dist >= 0 and wish_dist > 0:
                predicted_dists.append(float(pred_dist))
                wish_dists.append(float(wish_dist))

        if len(predicted_dists) == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        return np.array(predicted_dists, dtype=float), np.array(wish_dists, dtype=float)

    def _compute_contact_set_spearman_dscc(self, X, alpha, contacts, scale_s=1.0):
        pred, wish = self._build_distance_vectors_from_contacts(X, alpha, contacts, scale_s=scale_s)
        if len(pred) < 10:
            return 0.0

        if np.std(pred) < 1e-12 or np.std(wish) < 1e-12:
            return 0.0

        try:
            corr, _ = spearmanr(pred, wish)
            return float(corr) if np.isfinite(corr) else 0.0
        except Exception:
            return 0.0

    def _compute_all_pairs_dscc(self, X, alpha, scale_s=1.0, block_size=512):

        if self.N is None or self.N < 2:
            return 0.0
        if self.hic_sparse is None:
            self.log("all-pairs dSCC requested but Hi-C sparse matrix is unavailable", "WARN")
            return 0.0

        n = int(self.N)
        n_pairs_total = (n * (n - 1)) // 2
        max_pairs = getattr(self, 'dscc_all_pairs_max_pairs', None)
        if max_pairs is not None and n_pairs_total > int(max_pairs):
            raise RuntimeError(
                f"all-pairs dSCC requires {n_pairs_total:,} pairs, exceeding configured "
                f"--dscc-all-pairs-max-pairs={int(max_pairs):,}. "
                "Increase the cap or switch to contacts-universe evaluation."
            )

        zero_policy = str(getattr(self, 'dscc_all_pairs_zero_policy', 'epsilon')).lower()
        if_eps = float(getattr(self, 'dscc_all_pairs_if_epsilon', 1e-6))
        if_eps = max(1e-12, if_eps)

        pred_chunks = []
        wish_chunks = []

        for i0 in range(0, n, block_size):
            i1 = min(n, i0 + block_size)
            Xi = X[i0:i1]

            for j0 in range(i0, n, block_size):
                j1 = min(n, j0 + block_size)
                Xj = X[j0:j1]

                d_block = cdist(Xi, Xj)
                if_block = self.hic_sparse[i0:i1, j0:j1].toarray().astype(float, copy=False)

                if zero_policy == 'skip-zeros':
                    valid_if = if_block > 0
                    if np.any(valid_if):
                        wish_block = np.zeros_like(if_block, dtype=float)
                        wish_block[valid_if] = float(scale_s) * np.power(if_block[valid_if], -alpha)
                    else:
                        wish_block = np.zeros_like(if_block, dtype=float)
                else:
                    wish_block = float(scale_s) * np.power(np.maximum(if_block, if_eps), -alpha)

                if i0 == j0:
                    tri_mask = np.triu(np.ones((i1 - i0, j1 - j0), dtype=bool), k=1)
                    if zero_policy == 'skip-zeros':
                        tri_mask &= (if_block > 0)
                    if np.any(tri_mask):
                        pred_chunks.append(d_block[tri_mask].ravel())
                        wish_chunks.append(wish_block[tri_mask].ravel())
                else:
                    if zero_policy == 'skip-zeros':
                        block_mask = (if_block > 0)
                        if np.any(block_mask):
                            pred_chunks.append(d_block[block_mask].ravel())
                            wish_chunks.append(wish_block[block_mask].ravel())
                    else:
                        pred_chunks.append(d_block.ravel())
                        wish_chunks.append(wish_block.ravel())

        if not pred_chunks:
            return 0.0

        pred = np.concatenate(pred_chunks)
        wish = np.concatenate(wish_chunks)
        if len(pred) < 10:
            return 0.0

        if np.std(pred) < 1e-12 or np.std(wish) < 1e-12:
            return 0.0

        corr, _ = spearmanr(pred, wish)
        return float(corr) if np.isfinite(corr) else 0.0

    def _compute_final_evaluation_dscc(self, X, alpha, scale_s=1.0):
        eval_universe = str(getattr(self, 'dscc_eval_universe', 'contacts')).lower().replace('_', '-')
        if eval_universe == 'all-pairs':
            dscc = self._compute_all_pairs_dscc(X, alpha, scale_s=scale_s)
            return float(dscc), {
                'metric': 'spearman',
                'universe': 'all-pairs',
                'zero_policy': str(getattr(self, 'dscc_all_pairs_zero_policy', 'epsilon')),
                'if_epsilon': float(getattr(self, 'dscc_all_pairs_if_epsilon', 1e-6)),
                'n_pairs_total': int((int(self.N) * (int(self.N) - 1)) // 2) if self.N is not None else 0,
            }

        p_all = list(self.P_inter)
        for _, contacts in self.P_intra.items():
            p_all.extend(contacts)
        dscc = self._compute_contact_set_spearman_dscc(X, alpha, p_all, scale_s=scale_s)
        return float(dscc), {
            'metric': 'spearman',
            'universe': 'contacts',
            'n_inter_contacts': int(len(self.P_inter)),
            'n_intra_contacts': int(sum(len(v) for v in self.P_intra.values())),
            'n_total_contacts': int(len(p_all)),
        }

    def stage_E_scaling(self):

        self.log("Applying contact-derived backbone scale from Stage D...")

        r = float(getattr(self, 'scale_inter', 1.0))
        if not np.isfinite(r) or r <= 0:
            self.log("Invalid Stage D scale_inter; falling back to unscaled backbone", "WARN")
            r = 1.0

        self.backbone_coords_scaled = float(r) * self.backbone_coords
        self.log(f"Backbone scale ratio (contact-derived): {r:.4f}")

        self.diagnostics['stage_E'] = {
            'scale_ratio': float(r),
            'n_ratios': 1,
            'source': 'stage_d_contact_fit_scale',
            'note': 'Cross-frame MB center-to-center ratio disabled; using contact-derived scale_inter'
        }

    def stage_F_initial_placement(self):

        self.log("Placing structures via translation...")

        X_placed = np.zeros((self.N, 3))

        if self._t_bins_by_structure is None:
            self._build_structure_bin_cache()

        def place_structure(args):
            s, info = args
            coords = info['coords']
            c_s = np.mean(coords, axis=0)

            b = self.mb_anchors[s]
            B_s = self.backbone_coords_scaled[b]

            coords_translated = coords + (B_s - c_s)

            placements = {}
            t_bins = self._t_bins_by_structure[s] if self._t_bins_by_structure is not None else [t for t in range(self.N) if self.structure_id[t] == s]
            for t in t_bins:
                k = self.local_index[t]
                if k >= 0 and k < len(coords_translated):
                    placements[t] = coords_translated[k]

            return placements

        results = self._parallel_map(place_structure, list(enumerate(self.mb_info)), n_tasks=len(self.mb_info), chunksize=1)
        for placements in results:
            for t, coord in placements.items():
                X_placed[t] = coord

        self.X_initial = X_placed

        n_mapped = 0
        for t in range(self.N):
            if self.owner[t] is not None:
                if np.any(np.isfinite(X_placed[t])) and np.linalg.norm(X_placed[t]) > 0:
                    n_mapped += 1
        self.log(f"Coordinate mapping: {n_mapped:,} bins mapped out of {np.sum(self.owner != None):,} owned bins")

        dscc = self._compute_inter_dscc(X_placed, self.alpha_inter, self.P_inter, scale_s=self.scale_inter)
        self.log(f"Initial placement complete. Inter dSCC: {dscc:.4f}")
        if dscc < 0:
            self.log(f"WARNING: Negative dSCC indicates alpha ({self.alpha_inter:.3f}) may be incorrect "
                    f"or coordinates need better alignment. This should improve with rotation.", "WARN")

        self.diagnostics['stage_F'] = {
            'inter_dscc': float(dscc),
            'n_mapped_bins': int(n_mapped)
        }

    def stage_G_rigid_rotation(self):

        import math
        import numpy as np
        from scipy.optimize import minimize
        from numba import njit

        self.log("Optimizing rigid transforms (rotation + small translation) with Numba JIT...")

        if self.X_initial is None:
            raise RuntimeError("Stage F must run before Stage G")
        if self.alpha_inter is None:
            raise RuntimeError("alpha_inter must be set before Stage G")

        max_sweeps = int(getattr(self, 'stage_g_max_sweeps', 10))
        convergence_threshold = float(getattr(self, 'stage_g_convergence_threshold', 1e-4))
        lambda_inter = float(getattr(self, 'stage_g_lambda_inter', 1.0))
        mu_translation = float(getattr(self, 'stage_g_mu_translation', 0.05))
        huber_delta = getattr(self, 'stage_g_huber_delta', None)
        max_translation_step = float(getattr(self, 'stage_g_max_translation_step', 2.0))
        max_rotation_step = float(getattr(self, 'stage_g_max_rotation_step', 0.6))
        per_structure_maxiter = int(getattr(self, 'stage_g_per_structure_maxiter', 60))
        scale_inter = float(getattr(self, 'scale_inter', 1.0))
        if not np.isfinite(scale_inter) or scale_inter <= 0:
            scale_inter = 1.0

        if self._t_bins_by_structure is None:
            self._build_structure_bin_cache()

        X_current = self.X_initial.copy()
        X_ref = self.X_initial.copy()
        contacts_by_structure = self._build_inter_contacts_by_structure(self.P_inter)

        if huber_delta is None:
            sample = self.P_inter[:min(len(self.P_inter), 5000)]
            if len(sample) >= 10:
                d = np.array([np.linalg.norm(X_current[u] - X_current[v]) for (u, v, _) in sample])
                med = float(np.median(d))
                huber_delta = max(1e-6, 0.1 * med)
            else:
                huber_delta = 1.0
        huber_delta = float(huber_delta)

        dscc0 = self._compute_inter_dscc(X_current, self.alpha_inter, self.P_inter, scale_s=scale_inter)
        stress0 = self._compute_inter_stress(
            X_current, self.alpha_inter, self.P_inter, huber_delta=huber_delta, scale_s=scale_inter
        )
        self.log(
            f"Stage G start: inter dSCC={dscc0:.4f}, inter stress={stress0:.6g} "
            f"(huber_delta={huber_delta:.4g}, scale_inter={scale_inter:.4f})"
        )

        sweep_history = []
        best_stress = float(stress0)
        best_X = X_current.copy()
        prev_sweep_stress = float(stress0)

        @njit(fastmath=False)
        def _numba_f_and_g(w, t, R, RT, Xs, c0, c_ref, mu_translation, huber_delta,
                           lam_w_arr, wish_arr, iu_arr, iv_arr,
                           fu_arr, fv_arr, x0u_arr, x0v_arr):

            f = 0.0
            gw0, gw1, gw2 = 0.0, 0.0, 0.0
            gt0, gt1, gt2 = 0.0, 0.0, 0.0

            n_bins = Xs.shape[0]
            c_now0, c_now1, c_now2 = 0.0, 0.0, 0.0
            for i in range(n_bins):
                c_now0 += Xs[i, 0]; c_now1 += Xs[i, 1]; c_now2 += Xs[i, 2]
            c_now0 /= n_bins; c_now1 /= n_bins; c_now2 /= n_bins

            dc0 = c_now0 - c_ref[0]
            dc1 = c_now1 - c_ref[1]
            dc2 = c_now2 - c_ref[2]

            f += mu_translation * (dc0*dc0 + dc1*dc1 + dc2*dc2)

            gt0 += 2.0 * mu_translation * dc0
            gt1 += 2.0 * mu_translation * dc1
            gt2 += 2.0 * mu_translation * dc2

            rtdc0 = RT[0,0]*dc0 + RT[0,1]*dc1 + RT[0,2]*dc2
            rtdc1 = RT[1,0]*dc0 + RT[1,1]*dc1 + RT[1,2]*dc2
            rtdc2 = RT[2,0]*dc0 + RT[2,1]*dc1 + RT[2,2]*dc2

            cw0 = c0[1]*rtdc2 - c0[2]*rtdc1
            cw1 = c0[2]*rtdc0 - c0[0]*rtdc2
            cw2 = c0[0]*rtdc1 - c0[1]*rtdc0

            gw0 += 2.0 * mu_translation * cw0
            gw1 += 2.0 * mu_translation * cw1
            gw2 += 2.0 * mu_translation * cw2

            for i in range(len(lam_w_arr)):
                lam_w = lam_w_arr[i]
                wish = wish_arr[i]
                iu = iu_arr[i]
                iv = iv_arr[i]

                if iu != -1:
                    xu0, xu1, xu2 = Xs[iu, 0], Xs[iu, 1], Xs[iu, 2]
                else:
                    xu0, xu1, xu2 = fu_arr[i, 0], fu_arr[i, 1], fu_arr[i, 2]

                if iv != -1:
                    xv0, xv1, xv2 = Xs[iv, 0], Xs[iv, 1], Xs[iv, 2]
                else:
                    xv0, xv1, xv2 = fv_arr[i, 0], fv_arr[i, 1], fv_arr[i, 2]

                dx0 = xu0 - xv0
                dx1 = xu1 - xv1
                dx2 = xu2 - xv2

                d = math.sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2)
                if d < 1e-9:
                    continue

                r = d - wish
                a = abs(r)

                if a <= huber_delta:
                    rho = 0.5 * r * r
                    drho = r
                else:
                    rho = huber_delta * (a - 0.5 * huber_delta)
                    drho = huber_delta * (1.0 if r >= 0.0 else -1.0)

                f += lam_w * rho

                gmag = (lam_w * drho) / d
                gx0 = gmag * dx0
                gx1 = gmag * dx1
                gx2 = gmag * dx2

                if iu != -1:
                    gt0 += gx0
                    gt1 += gx1
                    gt2 += gx2

                    rtgx0 = RT[0,0]*gx0 + RT[0,1]*gx1 + RT[0,2]*gx2
                    rtgx1 = RT[1,0]*gx0 + RT[1,1]*gx1 + RT[1,2]*gx2
                    rtgx2 = RT[2,0]*gx0 + RT[2,1]*gx1 + RT[2,2]*gx2

                    x0u0, x0u1, x0u2 = x0u_arr[i, 0], x0u_arr[i, 1], x0u_arr[i, 2]

                    gw0 += x0u1*rtgx2 - x0u2*rtgx1
                    gw1 += x0u2*rtgx0 - x0u0*rtgx2
                    gw2 += x0u0*rtgx1 - x0u1*rtgx0

                if iv != -1:
                    gt0 -= gx0
                    gt1 -= gx1
                    gt2 -= gx2

                    rtgx0 = RT[0,0]*gx0 + RT[0,1]*gx1 + RT[0,2]*gx2
                    rtgx1 = RT[1,0]*gx0 + RT[1,1]*gx1 + RT[1,2]*gx2
                    rtgx2 = RT[2,0]*gx0 + RT[2,1]*gx1 + RT[2,2]*gx2

                    x0v0, x0v1, x0v2 = x0v_arr[i, 0], x0v_arr[i, 1], x0v_arr[i, 2]

                    gw0 -= x0v1*rtgx2 - x0v2*rtgx1
                    gw1 -= x0v2*rtgx0 - x0v0*rtgx2
                    gw2 -= x0v0*rtgx1 - x0v1*rtgx0

            g = np.empty(6, dtype=np.float64)
            g[0], g[1], g[2] = gw0, gw1, gw2
            g[3], g[4], g[5] = gt0, gt1, gt2
            return f, g

        def optimize_structure(s):
            t_bins = self._t_bins_by_structure[s]
            if not t_bins:
                return None

            contacts_s = contacts_by_structure.get(s, [])
            if len(contacts_s) < 5:
                return {
                    's': int(s), 'success': True, 'n_contacts': int(len(contacts_s)),
                    'rot_norm': 0.0, 'trans_norm': 0.0, 'nit': 0, 'fun': None
                }

            X0 = X_current[t_bins].copy()
            X0_ref = X_ref[t_bins].copy()
            idx_map = {tb: i for i, tb in enumerate(t_bins)}
            c0 = np.mean(X0, axis=0)
            c_ref = np.mean(X0_ref, axis=0)

            n_c = len(contacts_s)
            lam_w_arr = np.zeros(n_c, dtype=np.float64)
            wish_arr = np.zeros(n_c, dtype=np.float64)
            iu_arr = np.full(n_c, -1, dtype=np.int64)
            iv_arr = np.full(n_c, -1, dtype=np.int64)
            fu_arr = np.zeros((n_c, 3), dtype=np.float64)
            fv_arr = np.zeros((n_c, 3), dtype=np.float64)
            x0u_arr = np.zeros((n_c, 3), dtype=np.float64)
            x0v_arr = np.zeros((n_c, 3), dtype=np.float64)

            count = 0
            for (u, v, if_uv) in contacts_s:
                if if_uv <= 0: continue
                iu = idx_map.get(u, -1)
                iv = idx_map.get(v, -1)
                if iu == -1 and iv == -1: continue

                lam_w_arr[count] = float(lambda_inter * (if_uv ** 0.5))
                wish_arr[count] = float(scale_inter * np.power(if_uv + 1e-6, -self.alpha_inter))
                iu_arr[count] = iu
                iv_arr[count] = iv

                if iu != -1:
                    x0u_arr[count] = X0[iu]
                else:
                    fu_arr[count] = X_current[u]

                if iv != -1:
                    x0v_arr[count] = X0[iv]
                else:
                    fv_arr[count] = X_current[v]

                count += 1

            if count == 0:
                return None

            lam_w_arr, wish_arr = lam_w_arr[:count], wish_arr[:count]
            iu_arr, iv_arr = iu_arr[:count], iv_arr[:count]
            fu_arr, fv_arr = fu_arr[:count], fv_arr[:count]
            x0u_arr, x0v_arr = x0u_arr[:count], x0v_arr[:count]

            def f_and_g(p):
                w = np.array(p[:3], dtype=float)
                t = np.array(p[3:], dtype=float)
                R = self._rodrigues(w)
                Xs = (X0 @ R.T) + t

                return _numba_f_and_g(
                    w, t, R, R.T, Xs, c0, c_ref, float(mu_translation), float(huber_delta),
                    lam_w_arr, wish_arr, iu_arr, iv_arr,
                    fu_arr, fv_arr, x0u_arr, x0v_arr
                )

            def fun(p):
                val, _ = f_and_g(p)
                return val

            def jac(p):
                _, g = f_and_g(p)
                return g

            x0_opt = np.zeros(6, dtype=float)
            bounds = [(-max_rotation_step, max_rotation_step)] * 3 + [(-max_translation_step, max_translation_step)] * 3

            try:
                res = minimize(fun, x0_opt, jac=jac, method='L-BFGS-B', bounds=bounds, options={'maxiter': per_structure_maxiter})
                p_opt = np.array(res.x, dtype=float)
                w_opt = np.clip(p_opt[:3], -max_rotation_step, max_rotation_step)
                t_opt = np.clip(p_opt[3:], -max_translation_step, max_translation_step)
                R_opt = self._rodrigues(w_opt)
                X_current[t_bins] = (X0 @ R_opt.T) + t_opt

                return {
                    's': int(s), 'success': bool(res.success), 'rot_norm': float(np.linalg.norm(w_opt)),
                    'trans_norm': float(np.linalg.norm(t_opt)), 'nit': int(getattr(res, 'nit', 0))
                }
            except Exception:
                return None

        for sweep in range(max_sweeps):
            per_struct = [optimize_structure(s) for s in range(len(self.mb_info))]
            per_struct = [d for d in per_struct if d is not None]

            dscc = self._compute_inter_dscc(X_current, self.alpha_inter, self.P_inter, scale_s=scale_inter)
            stress = self._compute_inter_stress(
                X_current, self.alpha_inter, self.P_inter, huber_delta=huber_delta, scale_s=scale_inter
            )

            mean_rot = float(np.mean([d['rot_norm'] for d in per_struct])) if per_struct else 0.0
            mean_tr = float(np.mean([d['trans_norm'] for d in per_struct])) if per_struct else 0.0
            n_moved = int(np.sum([(d['rot_norm'] > 1e-6) or (d['trans_norm'] > 1e-6) for d in per_struct]))

            if float(stress) < best_stress:
                best_stress = float(stress)
                best_X = X_current.copy()

            d_sweep = prev_sweep_stress - float(stress)
            prev_sweep_stress = float(stress)
            self.log(
                f"Sweep {sweep+1}/{max_sweeps}: stress={stress:.6g} (Δsweep={d_sweep:+.3g}), dSCC={dscc:.4f}, "
                f"moved={n_moved}/{len(self.mb_info)}, mean|w|={mean_rot:.3g}, mean|t|={mean_tr:.3g}"
            )

            if abs(d_sweep) / (abs(prev_sweep_stress) + 1e-12) < convergence_threshold:
                self.log(f"Converged after {sweep+1} sweeps (relative stress improvement < {convergence_threshold}).")
                break

        self.X_rotated = best_X
        final_dscc = self._compute_inter_dscc(self.X_rotated, self.alpha_inter, self.P_inter, scale_s=scale_inter)
        final_stress = self._compute_inter_stress(
            self.X_rotated, self.alpha_inter, self.P_inter, huber_delta=huber_delta, scale_s=scale_inter
        )
        self.log(f"Rotation+translation optimization complete. Final inter dSCC: {final_dscc:.4f}, stress={final_stress:.6g}")

        self.diagnostics['stage_G'] = {
            'objective': 'robust_weighted_stress',
            'lambda_inter': float(lambda_inter),
            'mu_translation': float(mu_translation),
            'huber_delta': float(huber_delta),
            'max_sweeps': int(max_sweeps),
            'convergence_threshold': float(convergence_threshold),
            'scale_inter': float(scale_inter),
            'final_inter_dscc': float(final_dscc),
            'final_inter_stress': float(final_stress),
            'n_sweeps': int(len(sweep_history)),
            'sweep_history_head': sweep_history[:50]
        }

    def _euler_to_rotation_matrix(self, angles):
        theta_x, theta_y, theta_z = angles

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta_x), -np.sin(theta_x)],
                       [0, np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                       [0, 1, 0],
                       [-np.sin(theta_y), 0, np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                       [np.sin(theta_z), np.cos(theta_z), 0],
                       [0, 0, 1]])

        R = Rz @ Ry @ Rx
        if np.linalg.det(R) < 0:
            R = -R
        return R

    def _skew(self, v):
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        return np.array([
            [0.0, -vz,  vy],
            [vz,  0.0, -vx],
            [-vy, vx,  0.0]
        ], dtype=float)

    def _rodrigues(self, w):
        theta = float(np.linalg.norm(w))
        if theta < 1e-12:
            return np.eye(3, dtype=float)
        k = w / theta
        K = self._skew(k)
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.eye(3, dtype=float) * c + (1.0 - c) * np.outer(k, k) + s * K
        return R

    def _huber(self, r, delta):
        a = abs(float(r))
        d = float(delta)
        if a <= d:
            return 0.5 * r * r, r
        return d * (a - 0.5 * d), d * np.sign(r)

    def _build_inter_contacts_by_structure(self, P_inter):
        out = defaultdict(list)
        if self.structure_id is None:
            return out
        for u, v, if_uv in P_inter:
            if not (0 <= u < self.N and 0 <= v < self.N):
                continue
            su = int(self.structure_id[u])
            sv = int(self.structure_id[v])
            if su >= 0:
                out[su].append((int(u), int(v), float(if_uv)))
            if sv >= 0 and sv != su:
                out[sv].append((int(u), int(v), float(if_uv)))
        return out

    def _compute_inter_stress(self, X, alpha, P_inter, huber_delta=1.0, scale_s=1.0):
        if len(P_inter) == 0:
            return 0.0
        tot = 0.0
        for u, v, if_uv in P_inter:
            if if_uv <= 0:
                continue
            d = float(np.linalg.norm(X[u] - X[v]))
            wish = float(scale_s) * float(np.power(if_uv + 1e-6, -alpha))
            r = d - wish
            rho, _ = self._huber(r, huber_delta)
            w = float(if_uv ** 0.5)
            tot += w * rho
        return float(tot)

    def _stage_g_gradient_check(self, structure_id=0, n_tests=3, eps=1e-6, seed=0):

        if self.X_initial is None:
            raise RuntimeError("Run through Stage F before gradient check")
        if self.alpha_inter is None:
            raise RuntimeError("alpha_inter must be set before gradient check")
        if self._t_bins_by_structure is None:
            self._build_structure_bin_cache()

        s = int(structure_id)
        if s < 0 or s >= len(self.mb_info):
            raise ValueError(f"Invalid structure_id={s}")

        lambda_inter = float(getattr(self, 'stage_g_lambda_inter', 1.0))
        mu_translation = float(getattr(self, 'stage_g_mu_translation', 0.05))
        huber_delta = getattr(self, 'stage_g_huber_delta', None)
        max_translation_step = float(getattr(self, 'stage_g_max_translation_step', 2.0))
        max_rotation_step = float(getattr(self, 'stage_g_max_rotation_step', 0.6))

        X_current = self.X_initial.copy()
        X_ref = self.X_initial.copy()
        contacts_by_structure = self._build_inter_contacts_by_structure(self.P_inter)
        contacts_s = contacts_by_structure.get(s, [])
        t_bins = self._t_bins_by_structure[s]
        if len(t_bins) == 0 or len(contacts_s) < 5:
            self.log(f"Stage G grad-check: structure {s} has insufficient bins/contacts ({len(t_bins)} bins, {len(contacts_s)} contacts)", "WARN")
            return

        if huber_delta is None:
            sample = self.P_inter[:min(len(self.P_inter), 5000)]
            if len(sample) >= 10:
                d = np.array([np.linalg.norm(X_current[u] - X_current[v]) for (u, v, _) in sample])
                med = float(np.median(d))
                huber_delta = max(1e-6, 0.1 * med)
            else:
                huber_delta = 1.0
        huber_delta = float(huber_delta)

        X0 = X_current[t_bins].copy()
        X0_ref = X_ref[t_bins].copy()
        idx_map = {tb: i for i, tb in enumerate(t_bins)}
        c0 = np.mean(X0, axis=0)
        c_ref = np.mean(X0_ref, axis=0)

        def f_and_g(p):
            w = np.array(p[:3], dtype=float)
            t = np.array(p[3:], dtype=float)
            R = self._rodrigues(w)
            Xs = (X0 @ R.T) + t

            f = 0.0
            g_w = np.zeros(3, dtype=float)
            g_t = np.zeros(3, dtype=float)

            c_now = np.mean(Xs, axis=0)
            dc = (c_now - c_ref)
            f += mu_translation * float(np.dot(dc, dc))
            g_t += 2.0 * mu_translation * dc
            g_w += 2.0 * mu_translation * (-(R @ self._skew(c0)).T @ dc)

            eps_d = 1e-9
            for (u, v, if_uv) in contacts_s:
                if if_uv <= 0:
                    continue
                w_uv = float(if_uv ** 0.5)
                wish = float(np.power(if_uv + 1e-6, -self.alpha_inter))

                iu = idx_map.get(u, None)
                iv = idx_map.get(v, None)
                if iu is None and iv is None:
                    continue

                xu = Xs[iu] if iu is not None else X_current[u]
                xv = Xs[iv] if iv is not None else X_current[v]

                dvec = (xu - xv)
                d = float(np.linalg.norm(dvec))
                if d < eps_d:
                    continue
                r = d - wish
                rho, drho = self._huber(r, huber_delta)
                f += lambda_inter * w_uv * rho

                grad_x = (lambda_inter * w_uv * drho) * (dvec / d)
                if iu is not None:
                    x0u = X0[iu]
                    g_t += grad_x
                    g_w += (-(R @ self._skew(x0u)).T @ grad_x)
                if iv is not None:
                    x0v = X0[iv]
                    g_t -= grad_x
                    g_w += (-(R @ self._skew(x0v)).T @ (-grad_x))

            g = np.concatenate([g_w, g_t])
            return f, g

        rng = np.random.default_rng(int(seed))
        self.log(
            f"Stage G grad-check: structure={s}, tests={int(n_tests)}, eps={eps:g}, huber_delta={huber_delta:.4g}"
        )

        for k in range(int(n_tests)):
            p = np.zeros(6, dtype=float)
            p[:3] = rng.uniform(-max_rotation_step, max_rotation_step, size=3)
            p[3:] = rng.uniform(-max_translation_step, max_translation_step, size=3)

            f0, g0 = f_and_g(p)
            g_fd = np.zeros_like(g0)
            for i in range(6):
                dp = np.zeros(6, dtype=float)
                dp[i] = float(eps)
                fp, _ = f_and_g(p + dp)
                fm, _ = f_and_g(p - dp)
                g_fd[i] = (fp - fm) / (2.0 * float(eps))

            num = float(np.linalg.norm(g0 - g_fd))
            den = float(np.linalg.norm(g_fd) + 1e-12)
            rel = num / den
            self.log(
                f"  test {k+1}: f={f0:.6g}, ||g||={np.linalg.norm(g0):.3g}, ||g_fd||={np.linalg.norm(g_fd):.3g}, rel_err={rel:.3g}"
            )

    def _collapse_terminal_tail_gaps(self, X_global):
        if self.owner is None:
            return
        owned = np.where(self.owner != None)[0]
        if len(owned) == 0:
            return
        last_owned = int(np.max(owned))
        if last_owned >= self.N - 1:
            return
        if self.owner[last_owned + 1] is not None:
            return

        first_gap = last_owned + 1
        tail_end = first_gap
        while tail_end + 1 < self.N and self.owner[tail_end + 1] is None:
            tail_end += 1

        anchor = X_global[last_owned].copy()
        for t in range(first_gap, tail_end + 1):
            X_global[t] = anchor

        self.diagnostics.setdefault('stage_H', {})
        self.diagnostics['stage_H']['tail_collapsed'] = {
            'first_gap': int(first_gap),
            'tail_end': int(tail_end),
            'n_bins': int(tail_end - first_gap + 1),
            'anchor_bp': int(self.valid_bin_starts[last_owned]) if self.valid_bin_starts is not None else -1,
        }
        self.log(
            f"Collapsed terminal tail gap bins {first_gap}-{tail_end} "
            f"to last owned bead (no extrapolation)"
        )

    def stage_H_stitching(self):

        self.log("Stitching structures into global array...")

        X_global = np.zeros((self.N, 3))

        for t in range(self.N):
            if self.owner[t] is not None:
                X_global[t] = self.X_rotated[t]

        gap_bins = np.where(self.owner == None)[0]
        self.log(f"Filling {len(gap_bins)} gap bins with cubic splines...")

        nearest_left = np.full(self.N, -1, dtype=int)
        last = -1
        for t in range(self.N):
            if self.owner[t] is not None:
                last = t
            nearest_left[t] = last

        nearest_right = np.full(self.N, -1, dtype=int)
        last = -1
        for t in range(self.N - 1, -1, -1):
            if self.owner[t] is not None:
                last = t
            nearest_right[t] = last

        gap_regions = []
        if len(gap_bins) > 0:
            current_region = [gap_bins[0]]
            for i in range(1, len(gap_bins)):
                if gap_bins[i] == gap_bins[i-1] + 1:
                    current_region.append(gap_bins[i])
                else:
                    if len(current_region) > 0:
                        gap_regions.append(current_region)
                    current_region = [gap_bins[i]]
            if len(current_region) > 0:
                gap_regions.append(current_region)

        for gap_region in gap_regions:
            t_start = gap_region[0]
            t_end = gap_region[-1]
            L = int(nearest_left[t_start])
            R = int(nearest_right[t_end])

            if L >= 0 and R >= 0 and R > L:
                boundary_t = [L]
                boundary_coords = [X_global[L]]

                for i in range(max(0, L-2), L):
                    if self.owner[i] is not None:
                        boundary_t.append(i)
                        boundary_coords.append(X_global[i])

                for i in range(R+1, min(self.N, R+3)):
                    if self.owner[i] is not None:
                        boundary_t.append(i)
                        boundary_coords.append(X_global[i])

                boundary_t.append(R)
                boundary_coords.append(X_global[R])

                unique_indices = sorted(set(boundary_t))
                unique_coords = [X_global[i] for i in unique_indices]

                if len(unique_indices) >= 2:
                    for dim in range(3):
                        coords_dim = [c[dim] for c in unique_coords]
                        spline = CubicSpline(unique_indices, coords_dim, bc_type='natural')
                        for t in gap_region:
                            X_global[t, dim] = spline(t)
                else:
                    for t in gap_region:
                        alpha_t = (t - L) / (R - L)
                        X_global[t] = X_global[L] + alpha_t * (X_global[R] - X_global[L])
            elif L >= 0:

                if L >= 1 and self.owner[L-1] is not None:
                    step_vec = X_global[L] - X_global[L-1]
                else:
                    step_vec = np.array([1.0, 0.0, 0.0])

                step_size = float(np.linalg.norm(step_vec))
                if not np.isfinite(step_size) or step_size < 1e-6:
                    step_size = 3.0
                    direction = np.array([1.0, 0.0, 0.0])
                else:
                    direction = step_vec / step_size

                for idx, t in enumerate(gap_region):
                    X_global[t] = X_global[L] + (idx + 1) * step_size * direction
            elif R >= 0:
                if R < self.N - 1:
                    direction = X_global[R+1] - X_global[R]
                else:
                    direction = np.array([1.0, 0.0, 0.0])

                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-10:
                    direction = direction / dir_norm
                else:
                    direction = np.array([1.0, 0.0, 0.0])

                step_size = 3.0
                if R < self.N - 1:
                    step_size = np.linalg.norm(X_global[R+1] - X_global[R])

                for idx, t in enumerate(reversed(gap_region)):
                    offset = (idx + 1) * step_size * direction
                    X_global[t] = X_global[R] - offset
            else:
                for t in gap_region:
                    bp_t = int(self.valid_bin_starts[t]) if self.valid_bin_starts is not None else int(t * self.resolution + self.G0)
                    b = int(round(bp_t / 1_000_000))
                    b = max(0, min(b, len(self.backbone_coords_scaled) - 1))
                    X_global[t] = self.backbone_coords_scaled[b]

        self.X_stitched = X_global

        if getattr(self, 'enable_boundary_alignment', False) and getattr(self, 'boundary_alignment_mode', 'none') != 'none':
            self.log("Applying boundary-aware rigid translations to reduce large jumps...")
            X_global = self._boundary_align_structures(
                X_global,
                window=int(getattr(self, 'boundary_alignment_window', 5)),
                trigger_sigma=float(getattr(self, 'boundary_alignment_trigger_sigma', 6.0)),
                strength=float(getattr(self, 'boundary_alignment_strength', 1.0)),
                mode=str(getattr(self, 'boundary_alignment_mode', 'sequential')),
            )
            self.X_stitched = X_global

        tail_refine_report = {
            'applied': False,
            'reason': 'no_tail_gap_detected'
        }

        if len(gap_bins) > 0:
            last_owned = int(np.max(np.where(self.owner != None)[0]))
            if last_owned < self.N - 1 and self.owner[last_owned + 1] is None:
                first_gap = last_owned + 1

                tail_end = first_gap
                while tail_end + 1 < self.N and self.owner[tail_end + 1] is None:
                    tail_end += 1

                if last_owned >= 1 and self.owner[last_owned - 1] is not None:
                    step_vec = X_global[last_owned] - X_global[last_owned - 1]
                    step_size = float(np.linalg.norm(step_vec))
                    if not np.isfinite(step_size) or step_size < 1e-6:
                        step_size = 3.0
                        direction = np.array([1.0, 0.0, 0.0])
                    else:
                        direction = step_vec / step_size

                    old_first = X_global[first_gap].copy()
                    new_first = X_global[last_owned] + step_size * direction
                    delta = new_first - old_first

                    X_global[first_gap:tail_end + 1] = X_global[first_gap:tail_end + 1] + delta
                    self.X_stitched = X_global

                if (
                    getattr(self, 'enable_tail_gap_refinement', True)
                    and not getattr(self, 'collapse_terminal_tail_gaps', False)
                ):
                    tail_refine_report = self._refine_terminal_tail_gap(
                        X_global,
                        first_gap=first_gap,
                        tail_end=tail_end,
                    )
                    self.X_stitched = X_global
                else:
                    tail_refine_report = {
                        'applied': False,
                        'reason': 'disabled_by_flag',
                        'first_gap': int(first_gap),
                        'tail_end': int(tail_end),
                        'n_tail_bins': int(tail_end - first_gap + 1),
                    }

        if getattr(self, 'collapse_terminal_tail_gaps', False):
            self._collapse_terminal_tail_gaps(X_global)
            self.X_stitched = X_global

        assert len(X_global) == self.N, f"Length mismatch: {len(X_global)} != {self.N}"
        assert not np.any(np.isnan(X_global)), "NaN detected in stitched coordinates"

        n_valid = np.sum([np.all(np.isfinite(X_global[t])) and np.linalg.norm(X_global[t]) > 0
                         for t in range(self.N)])
        self.log(f"Stitching complete. Array length: {len(X_global):,}, valid coordinates: {n_valid:,}")

        try:
            self.diagnostics.setdefault('stage_H', {})
            self.diagnostics['stage_H']['consecutive_jump_stats'] = self._compute_consecutive_jump_stats(X_global)
        except Exception:
            pass

        if n_valid < self.N:
            self.log(f"Warning: {self.N - n_valid} bins have invalid coordinates", "WARN")

        self.diagnostics['stage_H'] = {
            'n_gaps_filled': int(len(gap_bins)),
            'array_length': int(len(X_global)),
            'n_valid_coords': int(n_valid),
            'tail_gap_refinement': tail_refine_report,
        }

        jump_stats = self._compute_consecutive_jump_stats(X_global)
        self.diagnostics['stage_H']['consecutive_jump_stats'] = jump_stats

    def _refine_terminal_tail_gap(self, X_global, first_gap, tail_end):

        try:
            first_gap = int(first_gap)
            tail_end = int(tail_end)
            if tail_end < first_gap or first_gap <= 0:
                return {
                    'applied': False,
                    'reason': 'invalid_tail_region',
                    'first_gap': int(first_gap),
                    'tail_end': int(tail_end),
                }

            if self.hic_sparse is None:
                return {
                    'applied': False,
                    'reason': 'hic_sparse_unavailable',
                    'first_gap': int(first_gap),
                    'tail_end': int(tail_end),
                }

            n_tail = int(tail_end - first_gap + 1)
            tail_bins = np.arange(first_gap, tail_end + 1, dtype=int)
            tail_map = {int(t): i for i, t in enumerate(tail_bins.tolist())}

            alpha = self.alpha_global if self.alpha_global is not None else self.alpha_inter
            if alpha is None:
                alpha = 0.5
            alpha = float(alpha)

            scale_s = float(getattr(self, 'scale_inter', 1.0))
            if (not np.isfinite(scale_s)) or scale_s <= 0:
                scale_s = 1.0

            csr = self.hic_sparse.tocsr() if not hasattr(self.hic_sparse, 'indptr') else self.hic_sparse
            indptr = csr.indptr
            indices = csr.indices
            data = csr.data

            topk = int(max(1, getattr(self, 'tail_gap_refine_topk_per_bin', 24)))
            eps_if = 1e-6

            uu_u, uu_v, uu_wish, uu_w = [], [], [], []
            uk_u, uk_k, uk_wish, uk_w = [], [], [], []

            seen_uu = set()
            for u in tail_bins:
                row_start = int(indptr[u])
                row_end = int(indptr[u + 1])
                if row_end <= row_start:
                    continue

                neigh = indices[row_start:row_end]
                vals = data[row_start:row_end]
                if len(vals) == 0:
                    continue

                valid = np.isfinite(vals) & (vals > 0)
                if not np.any(valid):
                    continue
                neigh = neigh[valid]
                vals = vals[valid]

                if len(vals) > topk:
                    keep_idx = np.argpartition(vals, -topk)[-topk:]
                    neigh = neigh[keep_idx]
                    vals = vals[keep_idx]

                u_local = tail_map[int(u)]
                for v, if_uv in zip(neigh.tolist(), vals.tolist()):
                    if v == u:
                        continue
                    if_uv = float(if_uv)
                    wish = float(scale_s * np.power(if_uv + eps_if, -alpha))
                    wt = float(np.sqrt(max(if_uv, eps_if)))

                    if v in tail_map:
                        v_local = tail_map[int(v)]
                        a = min(u_local, v_local)
                        b = max(u_local, v_local)
                        key = (a, b)
                        if a != b and key not in seen_uu:
                            seen_uu.add(key)
                            uu_u.append(a)
                            uu_v.append(b)
                            uu_wish.append(wish)
                            uu_w.append(wt)
                    else:
                        uk_u.append(u_local)
                        uk_k.append(int(v))
                        uk_wish.append(wish)
                        uk_w.append(wt)

            if len(uu_u) + len(uk_u) < 8:
                return {
                    'applied': False,
                    'reason': 'insufficient_tail_contacts',
                    'first_gap': int(first_gap),
                    'tail_end': int(tail_end),
                    'n_tail_bins': int(n_tail),
                    'n_uu_contacts': int(len(uu_u)),
                    'n_uk_contacts': int(len(uk_u)),
                }

            device = self.device
            x0 = torch.tensor(X_global[first_gap:tail_end + 1], dtype=torch.float32, device=device)
            x = x0.clone().detach().requires_grad_(True)

            known_idx_unique = sorted(set(uk_k))
            known_map = {k: i for i, k in enumerate(known_idx_unique)}
            kcoords = torch.tensor(X_global[known_idx_unique], dtype=torch.float32, device=device)
            uk_k_local = [known_map[k] for k in uk_k]

            if len(uu_u) > 0:
                t_uu_u = torch.tensor(uu_u, dtype=torch.long, device=device)
                t_uu_v = torch.tensor(uu_v, dtype=torch.long, device=device)
                t_uu_wish = torch.tensor(uu_wish, dtype=torch.float32, device=device)
                t_uu_w = torch.tensor(uu_w, dtype=torch.float32, device=device)
            else:
                t_uu_u = t_uu_v = t_uu_wish = t_uu_w = None

            if len(uk_u) > 0:
                t_uk_u = torch.tensor(uk_u, dtype=torch.long, device=device)
                t_uk_k = torch.tensor(uk_k_local, dtype=torch.long, device=device)
                t_uk_wish = torch.tensor(uk_wish, dtype=torch.float32, device=device)
                t_uk_w = torch.tensor(uk_w, dtype=torch.float32, device=device)
            else:
                t_uk_u = t_uk_k = t_uk_wish = t_uk_w = None

            lam_contact = float(getattr(self, 'tail_gap_refine_lambda_contact', 1.0))
            lam_smooth = float(getattr(self, 'tail_gap_refine_lambda_smooth', 0.10))
            lam_bond = float(getattr(self, 'tail_gap_refine_lambda_bond', 0.20))
            lam_reg = float(getattr(self, 'tail_gap_refine_lambda_reg', 0.02))
            huber_delta = float(getattr(self, 'tail_gap_refine_huber_delta', 1.0))
            max_iter = int(max(1, getattr(self, 'tail_gap_refine_max_iter', 200)))
            lr = float(getattr(self, 'tail_gap_refine_lr', 0.03))

            anchor = torch.tensor(X_global[first_gap - 1], dtype=torch.float32, device=device)
            step_ref = float(np.linalg.norm(X_global[first_gap] - X_global[first_gap - 1]))
            if (not np.isfinite(step_ref)) or step_ref < 1e-6:
                step_ref = 1.0

            opt = torch.optim.Adam([x], lr=lr)
            eps = 1e-6
            last_loss = None

            def huber_vec(r):
                abs_r = torch.abs(r)
                quad = 0.5 * (r ** 2)
                lin = huber_delta * (abs_r - 0.5 * huber_delta)
                return torch.where(abs_r <= huber_delta, quad, lin)

            for _ in range(max_iter):
                opt.zero_grad()
                loss = torch.tensor(0.0, dtype=torch.float32, device=device)

                if t_uu_u is not None:
                    du = torch.norm(x[t_uu_u] - x[t_uu_v], dim=1) + eps
                    ru = torch.log(du) - torch.log(t_uu_wish + eps)
                    loss = loss + lam_contact * torch.mean(t_uu_w * huber_vec(ru))

                if t_uk_u is not None:
                    dk = torch.norm(x[t_uk_u] - kcoords[t_uk_k], dim=1) + eps
                    rk = torch.log(dk) - torch.log(t_uk_wish + eps)
                    loss = loss + lam_contact * torch.mean(t_uk_w * huber_vec(rk))

                if n_tail >= 3:
                    sec = x[2:] - 2.0 * x[1:-1] + x[:-2]
                    loss = loss + lam_smooth * torch.mean(torch.sum(sec * sec, dim=1))
                    sec0 = x[1] - 2.0 * x[0] + anchor
                    loss = loss + lam_smooth * torch.sum(sec0 * sec0)

                chain = torch.cat([anchor[None, :], x], dim=0)
                step = torch.norm(chain[1:] - chain[:-1], dim=1)
                loss = loss + lam_bond * torch.mean((step - step_ref) ** 2)

                loss = loss + lam_reg * torch.mean((x - x0) ** 2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_([x], max_norm=100.0)
                opt.step()
                last_loss = float(loss.detach().item())

            X_global[first_gap:tail_end + 1] = x.detach().cpu().numpy()
            return {
                'applied': True,
                'reason': 'ok',
                'first_gap': int(first_gap),
                'tail_end': int(tail_end),
                'n_tail_bins': int(n_tail),
                'n_uu_contacts': int(len(uu_u)),
                'n_uk_contacts': int(len(uk_u)),
                'n_iter': int(max_iter),
                'final_loss': float(last_loss) if last_loss is not None else None,
                'alpha_used': float(alpha),
                'scale_used': float(scale_s),
            }
        except Exception as e:
            return {
                'applied': False,
                'reason': f'exception: {e}',
                'first_gap': int(first_gap),
                'tail_end': int(tail_end),
            }

    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    def stage_I_rigid_refinement(self):
        self.log("Rigid per-MB Stage I refinement (preserves domain geometry)...")

        P_combined = list(self.P_inter)
        for contacts in self.P_intra.values():
            P_combined.extend(contacts)

        alpha_candidates = np.concatenate([
            np.arange(0.1, 0.5, 0.05),
            np.arange(0.5, 1.2, 0.1),
        ])
        best_alpha = float(self.alpha_inter)
        best_dscc = -np.inf
        for alpha in alpha_candidates:
            dscc = self._compute_inter_dscc(self.X_stitched, alpha, P_combined)
            if dscc > best_dscc:
                best_dscc = dscc
                best_alpha = float(alpha)
        self.alpha_global = best_alpha
        self.log(f"α_global = {best_alpha:.3f} (dSCC = {best_dscc:.4f})")

        orig = {
            'X_initial': self.X_initial,
            'X_rotated': getattr(self, 'X_rotated', None),
            'P_inter': self.P_inter,
            'alpha_inter': self.alpha_inter,
            'lambda_inter': getattr(self, 'stage_g_lambda_inter', 0.5),
            'max_sweeps': getattr(self, 'stage_g_max_sweeps', 5),
            'mu_translation': getattr(self, 'stage_g_mu_translation', 0.05),
            'per_structure_maxiter': getattr(self, 'stage_g_per_structure_maxiter', 10),
        }

        try:
            self.X_initial = self.X_stitched.copy()
            self.P_inter = P_combined
            self.alpha_inter = self.alpha_global
            self.stage_g_lambda_inter = float(getattr(self, 'stage_i_rigid_lambda_inter', 1.0))
            self.stage_g_max_sweeps = int(getattr(self, 'stage_i_rigid_max_sweeps', 8))
            self.stage_g_mu_translation = float(getattr(self, 'stage_i_rigid_mu_translation', 0.12))
            self.stage_g_per_structure_maxiter = int(
                getattr(self, 'stage_i_rigid_per_structure_maxiter', 20)
            )
            self.stage_G_rigid_rotation()
            self.X_refined = self.X_rotated.copy()
        finally:
            self.X_initial = orig['X_initial']
            self.X_rotated = orig['X_rotated']
            self.P_inter = orig['P_inter']
            self.alpha_inter = orig['alpha_inter']
            self.stage_g_lambda_inter = orig['lambda_inter']
            self.stage_g_max_sweeps = orig['max_sweeps']
            self.stage_g_mu_translation = orig['mu_translation']
            self.stage_g_per_structure_maxiter = orig['per_structure_maxiter']

        scale_s = float(getattr(self, 'scale_inter', 1.0))
        final_dscc, eval_meta = self._compute_final_evaluation_dscc(
            self.X_refined, self.alpha_global, scale_s=scale_s
        )
        self.diagnostics['stage_I'] = {
            'mode': 'rigid',
            'alpha_global': float(self.alpha_global),
            'final_dscc': float(final_dscc),
            'scale_s_final': float(scale_s),
            'final_dscc_contract': eval_meta,
            'refinement_history': [],
        }
        self.log(f"Rigid refinement complete. Final dSCC: {final_dscc:.4f}")

    def stage_I_global_refinement(self):
        if getattr(self, 'stage_i_rigid_refinement', True):
            return self.stage_I_rigid_refinement()

        self.log("Global refinement (flexible per-bead)...")

        self.log("Grid search for α_global...")
        alpha_candidates = np.concatenate([
            np.arange(0.1, 0.5, 0.05),
            np.arange(0.5, 2.0, 0.1),
            np.arange(2.0, 3.1, 0.2)
        ])
        best_alpha = self.alpha_inter
        best_dscc = -np.inf

        P_combined = list(self.P_inter)
        for s, contacts in self.P_intra.items():
            P_combined.extend(contacts)

        self.log(f"Using ALL contacts for optimization: {len(self.P_inter):,} inter + {sum(len(v) for v in self.P_intra.values()):,} intra = {len(P_combined):,} total")

        def evaluate_alpha_global(alpha):
            return alpha, self._compute_inter_dscc(self.X_stitched, alpha, P_combined)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(evaluate_alpha_global, alpha_candidates))
            for alpha, dscc in results:
                if dscc > best_dscc:
                    best_dscc = dscc
                    best_alpha = alpha

        self.alpha_global = best_alpha
        self.log(f"Best α_global = {best_alpha:.3f} (dSCC = {best_dscc:.4f})")

        X_refined = self.X_stitched.copy()
        X0 = self.X_stitched.copy()

        lambda_inter = 1.0
        lambda_intra = 0.2
        mu = 0.8
        nu_base = 0.4

        scale_s = 1.0
        scale_update_every = 5

        use_log_distance_loss = True
        log_eps = 1e-6

        use_huber = True
        huber_delta = 2.0

        lambda_preserve = 0.25
        preserve_offsets = (1, 5, 10)

        r_min = 0.8
        lambda_hardcore = 0.35

        contact_density, contact_count = self._compute_contact_density()
        smoothness_weights = 1.0 - 0.9 * contact_density
        smoothness_weights = np.clip(smoothness_weights, 0.4, 1.0)

        structure_boundaries = []
        consecutive_dists = []
        for t in range(self.N - 1):
            if self.structure_id[t] != self.structure_id[t+1] and \
            self.structure_id[t] >= 0 and self.structure_id[t+1] >= 0:
                structure_boundaries.append(t)
                if not self._has_hic_contact(t, t+1):
                    dist = np.linalg.norm(X_refined[t+1] - X_refined[t])
                    consecutive_dists.append(dist)

        if len(consecutive_dists) > 0:
            non_boundary_dists = []
            for t in range(self.N - 1):
                if t not in structure_boundaries:
                    dist = np.linalg.norm(X_refined[t+1] - X_refined[t])
                    if dist > 0:
                        non_boundary_dists.append(dist)
            if len(non_boundary_dists) > 0:
                d_max = 2.5 * np.median(non_boundary_dists)
            else:
                d_max = 50.0
        else:
            d_max = 50.0

        lambda_repel = 0.15
        repel_window = 15
        repel_sigma = 8.0

        self.log(f"Tier 1 Smoothness: Adaptive weights range [{np.min(smoothness_weights):.3f}, {np.max(smoothness_weights):.3f}]")
        self.log(f"Tier 1 Boundary: {len(structure_boundaries)} boundaries, d_max = {d_max:.3f}")

        learning_rate = 0.005
        min_learning_rate = 0.001

        max_iter = 250
        best_dscc = self._compute_inter_dscc(X_refined, self.alpha_global, P_combined)
        patience = 5
        no_improve_count = 0

        preserve_pairs = self._build_mb_preserve_pairs(offsets=preserve_offsets)
        refinement_history = []

        for iter in range(max_iter):
            if iter % scale_update_every == 0:
                scale_s = self._estimate_global_scale_from_contacts(
                    X_refined, self.alpha_global, P_combined, weights_mode='sqrt_if'
                )

            X_refined = self._refinement_step_tier1(
                X_refined, X0, P_combined,
                lambda_inter, lambda_intra, mu, nu_base,
                smoothness_weights, structure_boundaries, d_max,
                lambda_repel, repel_window, repel_sigma,
                learning_rate,
                scale_s=scale_s,
                use_log_distance_loss=use_log_distance_loss,
                log_eps=log_eps,
                use_huber=use_huber,
                huber_delta=huber_delta,
                preserve_pairs=preserve_pairs,
                lambda_preserve=lambda_preserve,
                apply_repulsion_on_contacts=True,
                r_min=r_min,
                lambda_hardcore=lambda_hardcore
            )

            if (iter + 1) % 10 == 0:
                dscc = self._compute_inter_dscc(X_refined, self.alpha_global, P_combined)
                stress_like = self._compute_contact_stress_like(
                    X_refined, self.alpha_global, P_combined,
                    scale_s=scale_s,
                    use_log_distance_loss=use_log_distance_loss,
                    log_eps=log_eps,
                    use_huber=use_huber,
                    huber_delta=huber_delta,
                    weights_mode='sqrt_if'
                )
                self.log(
                    f"Refinement iter {iter+1}: dSCC = {dscc:.4f} (lr={learning_rate:.4f}, s={scale_s:.3f}, stress={stress_like:.3e})"
                )
                refinement_history.append({
                    'iter': int(iter + 1),
                    'dscc': float(dscc),
                    'learning_rate': float(learning_rate),
                    'scale_s': float(scale_s),
                    'stress_like': float(stress_like)
                })

                if dscc > best_dscc:
                    best_dscc = dscc
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    learning_rate = max(min_learning_rate, learning_rate * 0.9)

                if no_improve_count >= patience and iter > 20:
                    self.log(f"Early stopping: dSCC not improving for {patience} iterations")
                    break

        X_refined = self._smooth_structure_boundaries(X_refined)

        self.X_refined = X_refined

        final_dscc, eval_meta = self._compute_final_evaluation_dscc(
            X_refined, self.alpha_global, scale_s=scale_s
        )
        self.log(
            f"Refinement complete. Final dSCC [{eval_meta['metric']}, {eval_meta['universe']}]: "
            f"{final_dscc:.4f}"
        )

        self.diagnostics['stage_I'] = {
            'mode': 'flexible',
            'alpha_global': float(self.alpha_global),
            'final_dscc': float(final_dscc),
            'scale_s_final': float(scale_s),
            'final_dscc_contract': eval_meta,
            'refinement_history': refinement_history[:200]
        }

    def _refinement_step_tier1(self, X, X0, P_combined, lambda_inter, lambda_intra, mu, nu_base,
                                smoothness_weights, structure_boundaries, d_max,
                                lambda_repel, repel_window, repel_sigma, learning_rate=0.005,
                                scale_s=1.0,
                                use_log_distance_loss=True,
                                log_eps=1e-6,
                                use_huber=True,
                                huber_delta=2.0,
                                preserve_pairs=None,
                                lambda_preserve=0.0,
                                apply_repulsion_on_contacts=True,
                                r_min=0.8,
                                lambda_hardcore=0.35,
                                batch_size=8192):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not hasattr(self, '_fast_gpu_cache_initialized'):
            inter_adj = self._build_contact_adjacency(self.P_inter)
            intra_adj = self._build_contact_adjacency([c for contacts in self.P_intra.values() for c in contacts])
            boundary_set = set(structure_boundaries)

            i_u, i_v, i_wsh, i_w_lam = [], [], [], []
            ia_u, ia_v, ia_wsh, ia_w_lam = [], [], [], []
            p_u, p_v, p_base_lam = [], [], []
            r_u, r_v, r_lam = [], [], []
            b_u, b_v = [], []

            for t in range(self.N):
                for other, wsh, w in inter_adj[t]:
                    i_u.append(t); i_v.append(other); i_wsh.append(wsh); i_w_lam.append(w * lambda_inter)

                if self.owner[t] is not None:
                    for other, wsh, w in intra_adj[t]:
                        ia_u.append(t); ia_v.append(other); ia_wsh.append(wsh); ia_w_lam.append(w * lambda_intra)

                if preserve_pairs is not None and lambda_preserve > 0.0:
                    try:
                        for other in preserve_pairs[t]:
                            p_u.append(t); p_v.append(other)
                            p_base_lam.append(np.linalg.norm(X0[t] - X0[other]))
                    except (IndexError, TypeError): pass

                if t in boundary_set and t+1 < self.N and not self._has_hic_contact(t, t+1):
                    b_u.append(t); b_v.append(t+1)

                for j in range(max(0, t - repel_window), min(self.N, t + repel_window + 1)):
                    if j == t: continue
                    contact = self._has_hic_contact(t, j)
                    lam_eff = 0.0 if (contact and not apply_repulsion_on_contacts) else lambda_repel * (0.35 if contact else 1.0)
                    if lam_eff > 0 or lambda_hardcore > 0:
                        r_u.append(t); r_v.append(j); r_lam.append(lam_eff)

            self._g_i_u = torch.tensor(i_u, dtype=torch.long, device=device) if i_u else None
            self._g_i_v = torch.tensor(i_v, dtype=torch.long, device=device) if i_v else None
            self._g_i_wsh = torch.tensor(i_wsh, dtype=torch.float32, device=device) if i_u else None
            self._g_i_w_lam = torch.tensor(i_w_lam, dtype=torch.float32, device=device) if i_u else None

            self._g_ia_u = torch.tensor(ia_u, dtype=torch.long, device=device) if ia_u else None
            self._g_ia_v = torch.tensor(ia_v, dtype=torch.long, device=device) if ia_v else None
            self._g_ia_wsh = torch.tensor(ia_wsh, dtype=torch.float32, device=device) if ia_u else None
            self._g_ia_w_lam = torch.tensor(ia_w_lam, dtype=torch.float32, device=device) if ia_u else None

            self._g_p_u = torch.tensor(p_u, dtype=torch.long, device=device) if p_u else None
            self._g_p_v = torch.tensor(p_v, dtype=torch.long, device=device) if p_v else None
            self._g_p_base = torch.tensor(p_base_lam, dtype=torch.float32, device=device) if p_u else None

            self._g_r_u = torch.tensor(r_u, dtype=torch.long, device=device) if r_u else None
            self._g_r_v = torch.tensor(r_v, dtype=torch.long, device=device) if r_v else None
            self._g_r_lam = torch.tensor(r_lam, dtype=torch.float32, device=device) if r_u else None

            self._g_b_u = torch.tensor(b_u, dtype=torch.long, device=device) if b_u else None
            self._g_b_v = torch.tensor(b_v, dtype=torch.long, device=device) if b_u else None

            self._g_nu_t = (nu_base * torch.tensor(smoothness_weights, dtype=torch.float32, device=device)).unsqueeze(1)
            self._g_mu2 = mu * 2.0

            self._fast_gpu_cache_initialized = True

        t_X = torch.tensor(X, dtype=torch.float32, device=device)
        t_X0 = torch.tensor(X0, dtype=torch.float32, device=device)
        t_grad = torch.zeros_like(t_X)

        def apply_contact_force(u, v, wsh, w_lam):
            diff = t_X[u] - t_X[v]
            d = torch.norm(diff, dim=1).unsqueeze(1) + 1e-15
            target = float(scale_s) * wsh.unsqueeze(1)
            if use_log_distance_loss:
                r = torch.log(d + log_eps) - torch.log(target + log_eps)
                dr = torch.where(torch.abs(r) <= huber_delta, r, huber_delta * torch.sign(r)) if use_huber else r
                force = w_lam.unsqueeze(1) * dr * (diff / (d + log_eps))
            else:
                r = d - target
                dr = torch.where(torch.abs(r) <= huber_delta, r, huber_delta * torch.sign(r)) if use_huber else r
                force = w_lam.unsqueeze(1) * 2.0 * dr * (diff / d)
            t_grad.index_add_(0, u, force)

        if self._g_i_u is not None: apply_contact_force(self._g_i_u, self._g_i_v, self._g_i_wsh, self._g_i_w_lam)
        if self._g_ia_u is not None: apply_contact_force(self._g_ia_u, self._g_ia_v, self._g_ia_wsh, self._g_ia_w_lam)

        if self._g_p_u is not None:
            diff = t_X[self._g_p_u] - t_X[self._g_p_v]
            d = torch.norm(diff, dim=1).unsqueeze(1) + 1e-15
            base = self._g_p_base.unsqueeze(1)
            if use_log_distance_loss:
                r = torch.log(d + log_eps) - torch.log(base + log_eps)
                dr = torch.where(torch.abs(r) <= huber_delta, r, huber_delta * torch.sign(r)) if use_huber else r
                force = lambda_preserve * dr * (diff / (d + log_eps))
            else:
                r = d - base
                dr = torch.where(torch.abs(r) <= huber_delta, r, huber_delta * torch.sign(r)) if use_huber else r
                force = lambda_preserve * 2.0 * dr * (diff / d)
            t_grad.index_add_(0, self._g_p_u, force)

        if self._g_r_u is not None:
            diff = t_X[self._g_r_u] - t_X[self._g_r_v]
            d = torch.norm(diff, dim=1).unsqueeze(1) + 1e-15
            direction = diff / d

            m_r = (d < (3 * repel_sigma)).squeeze(1)
            if m_r.any():
                u_r, d_r, dir_r, lam_r = self._g_r_u[m_r], d[m_r], direction[m_r], self._g_r_lam[m_r].unsqueeze(1)
                force_r = lam_r * torch.exp(- (d_r**2) / (2 * repel_sigma**2)) * dir_r
                t_grad.index_add_(0, u_r, force_r)

            if lambda_hardcore > 0:
                m_h = (d < r_min).squeeze(1)
                if m_h.any():
                    u_h, d_h, dir_h = self._g_r_u[m_h], d[m_h], direction[m_h]
                    force_h = lambda_hardcore * 2.0 * (r_min - d_h) * dir_h
                    t_grad.index_add_(0, u_h, force_h)

        if self._g_b_u is not None:
            diff = t_X[self._g_b_v] - t_X[self._g_b_u]
            d = torch.norm(diff, dim=1).unsqueeze(1) + 1e-15
            m_b = (d > d_max).squeeze(1)
            if m_b.any():
                u_b, d_b, diff_b = self._g_b_u[m_b], d[m_b], diff[m_b]
                force_b = -4.0 * (d_b - float(d_max)) * (diff_b / d_b)
                t_grad.index_add_(0, u_b, force_b)

        t_grad += self._g_mu2 * (t_X - t_X0)
        smooth_grad = 2.0 * (2.0 * t_X[1:-1] - t_X[:-2] - t_X[2:])
        t_grad[1:-1] += self._g_nu_t[1:-1] * smooth_grad

        g_norms = torch.norm(t_grad, dim=1, keepdim=True)
        clip_mask = (g_norms > 100.0).expand_as(t_grad)
        t_grad = torch.where(clip_mask, (t_grad / g_norms) * 100.0, t_grad)

        t_X.sub_(t_grad, alpha=learning_rate)
        return t_X.cpu().numpy()

    def _refinement_step(self, X, X0, P_combined, lambda_inter, lambda_intra, mu, nu, learning_rate=0.005):
        return self._refinement_step_tier1(
            X, X0, P_combined, lambda_inter, lambda_intra, mu, nu,
            np.ones(self.N), [], 0.0, 0.0, 0, 0.0, learning_rate
        )

    def _estimate_global_scale_from_contacts(self, X, alpha, contacts, weights_mode='sqrt_if'):
        if contacts is None or len(contacts) == 0:
            return 1.0
        eps = 1e-6
        num = 0.0
        den = 0.0
        for u, v, if_uv in contacts:
            if u < 0 or v < 0 or u >= len(X) or v >= len(X):
                continue
            d_pred = float(np.linalg.norm(X[u] - X[v]))
            if not np.isfinite(d_pred):
                continue
            if if_uv <= 0 or not np.isfinite(if_uv):
                continue
            d_wish = float(np.power(if_uv + eps, -float(alpha)))
            if not np.isfinite(d_wish) or d_wish <= 0:
                continue
            if weights_mode == 'sqrt_if':
                w = float(np.sqrt(if_uv))
            else:
                w = 1.0
            num += w * d_pred * d_wish
            den += w * d_wish * d_wish
        if den <= 1e-12:
            return 1.0
        s = num / den
        return float(np.clip(s, 1e-3, 1e6))

    def _compute_contact_stress_like(self, X, alpha, contacts, scale_s=1.0,
                                    use_log_distance_loss=True, log_eps=1e-6,
                                    use_huber=True, huber_delta=2.0,
                                    weights_mode='sqrt_if'):
        if contacts is None or len(contacts) == 0:
            return 0.0
        eps = 1e-6
        tot = 0.0
        for u, v, if_uv in contacts:
            if u < 0 or v < 0 or u >= len(X) or v >= len(X):
                continue
            if if_uv <= 0 or not np.isfinite(if_uv):
                continue
            d_pred = float(np.linalg.norm(X[u] - X[v]))
            d_wish = float(np.power(if_uv + eps, -float(alpha)))
            target = float(scale_s) * d_wish
            if weights_mode == 'sqrt_if':
                w = float(np.sqrt(if_uv))
            else:
                w = 1.0
            if use_log_distance_loss:
                r = np.log(d_pred + log_eps) - np.log(target + log_eps)
            else:
                r = d_pred - target
            if use_huber:
                a = abs(r)
                d = float(huber_delta)
                if a <= d:
                    rho = 0.5 * r * r
                else:
                    rho = d * (a - 0.5 * d)
            else:
                rho = 0.5 * r * r
            tot += w * rho
        return float(tot)

    def _build_mb_preserve_pairs(self, offsets=(1, 5, 10)):
        pairs = [[] for _ in range(self.N)]
        if self.structure_id is None:
            return pairs
        offs = [int(o) for o in offsets if int(o) > 0]
        for t in range(self.N):
            s = int(self.structure_id[t])
            if s < 0:
                continue
            for o in offs:
                j = t + o
                if j < self.N and int(self.structure_id[j]) == s:
                    pairs[t].append(j)
                j2 = t - o
                if j2 >= 0 and int(self.structure_id[j2]) == s:
                    pairs[t].append(j2)
        return pairs

    def _smooth_structure_boundaries(self, X):
        X_smooth = X.copy()

        boundaries = []
        for t in range(self.N - 1):
            if self.structure_id[t] != self.structure_id[t+1] and \
               self.structure_id[t] >= 0 and self.structure_id[t+1] >= 0:
                boundaries.append(t)

        if len(boundaries) == 0:
            return X_smooth

        K = 5
        for boundary_t in boundaries:
            t_start = max(0, boundary_t - K)
            t_end = min(self.N - 1, boundary_t + K + 1)

            if not self._has_hic_contact(boundary_t, boundary_t + 1):
                boundary_bins = list(range(t_start, t_end + 1))
                if len(boundary_bins) >= 4:
                    for dim in range(3):
                        coords = [X_smooth[t, dim] for t in boundary_bins]
                        try:
                            spline = CubicSpline(boundary_bins, coords, bc_type='natural')
                            for t in boundary_bins:
                                X_smooth[t, dim] = spline(t)
                        except:
                            pass

        return X_smooth

    def stage_J_diagnostics_and_save(self):

        self.log("Computing final diagnostics...")

        dscc_final, eval_meta = self._compute_final_evaluation_dscc(
            self.X_refined, self.alpha_global, scale_s=float(getattr(self, 'scale_inter', 1.0))
        )

        dscc_inter = self._compute_contact_set_spearman_dscc(
            self.X_refined, self.alpha_global, self.P_inter, scale_s=float(getattr(self, 'scale_inter', 1.0))
        )

        intra_corr = self._compute_intra_preservation()

        n_nans = np.sum(np.isnan(self.X_refined))
        is_monotonic = self._check_monotonic_ordering()

        boundary_stats = self._check_boundary_continuity()

        self.diagnostics['stage_J'] = {
            'dscc_final': float(dscc_final),
            'dscc_inter': float(dscc_inter),
            'dscc_final_contract': eval_meta,
            'intra_preservation': float(intra_corr),
            'n_nans': int(n_nans),
            'is_monotonic': bool(is_monotonic),
            'boundary_stats': boundary_stats
        }

        self.log(f"Final Diagnostics:")
        self.log(f"  Final dSCC [{eval_meta['metric']}, {eval_meta['universe']}]: {dscc_final:.4f}")
        self.log(f"  Inter dSCC (diagnostic, Spearman): {dscc_inter:.4f}")
        self.log(f"  Intra preservation: {intra_corr:.4f}")
        self.log(f"  NaN count: {n_nans}")
        self.log(f"  Monotonic: {is_monotonic}")

        self._save_outputs()

        diagnostics_path = os.path.join(self.output_dir, 'diagnostics.json')
        final_alpha = self.alpha_global if self.alpha_global is not None else self.alpha_inter
        with open(diagnostics_path, 'w') as f:
            json.dump({"alpha_global": float(final_alpha)}, f, indent=2)
        self.log(f"Diagnostics saved: {diagnostics_path}")

    def _compute_consecutive_jump_stats(self, X):
        d = np.linalg.norm(X[1:] - X[:-1], axis=1)
        if len(d) == 0:
            return {
                'n': 0,
                'min': None,
                'median': None,
                'mean': None,
                'p95': None,
                'p99': None,
                'max': None
            }
        return {
            'n': int(len(d)),
            'min': float(np.min(d)),
            'median': float(np.median(d)),
            'mean': float(np.mean(d)),
            'p95': float(np.quantile(d, 0.95)),
            'p99': float(np.quantile(d, 0.99)),
            'max': float(np.max(d))
        }

    def _boundary_align_structures(self, X_global, window=5, trigger_sigma=6.0, strength=1.0, mode='sequential'):
        if mode not in ('sequential', 'none'):
            self.log(f"Unknown boundary_alignment_mode={mode}, skipping", "WARN")
            return X_global
        if mode == 'none':
            return X_global
        if self._t_bins_by_structure is None:
            self._build_structure_bin_cache()

        X = X_global.copy()

        d = np.linalg.norm(X[1:] - X[:-1], axis=1)
        med = float(np.median(d)) if len(d) else 0.0
        mad = float(np.median(np.abs(d - med))) if len(d) else 0.0
        robust_sigma = 1.4826 * mad
        if robust_sigma <= 1e-12:
            robust_sigma = max(1e-6, med * 0.1)
        trigger = med + trigger_sigma * robust_sigma

        n_structures = len(self.mb_info)
        applied = []

        for s in range(n_structures - 1):
            left_bins = self._t_bins_by_structure[s]
            right_bins = self._t_bins_by_structure[s + 1]
            if not left_bins or not right_bins:
                continue

            left_bins_sorted = sorted(left_bins)
            right_bins_sorted = sorted(right_bins)

            Lw = left_bins_sorted[-max(1, window):]
            Rw = right_bins_sorted[:max(1, window)]

            cL = np.mean(X[Lw], axis=0)
            cR = np.mean(X[Rw], axis=0)
            gap = float(np.linalg.norm(cL - cR))

            if gap <= trigger:
                continue

            delta = (cL - cR) * float(strength)
            X[right_bins_sorted] = X[right_bins_sorted] + delta
            applied.append({
                'boundary': int(s),
                'gap_before': gap,
                'delta_norm': float(np.linalg.norm(delta)),
                'n_bins_shifted': int(len(right_bins_sorted))
            })

        self.diagnostics.setdefault('stage_H', {})
        self.diagnostics['stage_H']['boundary_alignment'] = {
            'mode': mode,
            'window': int(window),
            'trigger_sigma': float(trigger_sigma),
            'strength': float(strength),
            'robust_median_step': float(med),
            'robust_sigma_step': float(robust_sigma),
            'trigger_distance': float(trigger),
            'n_boundaries_shifted': int(len(applied)),
            'shift_details_head': applied[:50]
        }

        return X

    def _compute_intra_preservation(self):
        dists_final = []
        dists_initial = []

        for s, contacts in self.P_intra.items():
            for u, v, _ in contacts:
                if u < len(self.X_refined) and v < len(self.X_refined):
                    d_final = np.linalg.norm(self.X_refined[u] - self.X_refined[v])
                    d_initial = np.linalg.norm(self.X_stitched[u] - self.X_stitched[v])
                    if np.isfinite(d_final) and np.isfinite(d_initial):
                        dists_final.append(d_final)
                        dists_initial.append(d_initial)

        if len(dists_final) < 10:
            return 0.0

        try:
            corr, _ = spearmanr(dists_final, dists_initial)
            return corr if np.isfinite(corr) else 0.0
        except:
            return 0.0

    def _check_monotonic_ordering(self):
        for t in range(1, self.N):
            dist = np.linalg.norm(self.X_refined[t] - self.X_refined[t-1])
            if dist > 1000:
                return False
        return True

    def _check_boundary_continuity(self):
        boundary_dists = []

        for s in range(len(self.mb_info) - 1):
            s_next = s + 1

            t_bins_s = [t for t in range(self.N) if self.structure_id[t] == s]
            t_bins_next = [t for t in range(self.N) if self.structure_id[t] == s_next]

            if len(t_bins_s) > 0 and len(t_bins_next) > 0:
                t_last = max(t_bins_s)
                t_first = min(t_bins_next)
                dist = np.linalg.norm(self.X_refined[t_last] - self.X_refined[t_first])
                boundary_dists.append(dist)

        if len(boundary_dists) == 0:
            return {}

        return {
            'mean': float(np.mean(boundary_dists)),
            'median': float(np.median(boundary_dists)),
            'std': float(np.std(boundary_dists)),
            'min': float(np.min(boundary_dists)),
            'max': float(np.max(boundary_dists))
        }

    def _save_outputs(self):
        self.log("Saving outputs...")

        pdb_path = os.path.join(self.output_dir, f'{self.chr_name}_assembled_global.pdb')
        self._write_pdb(self.X_refined, pdb_path)
        self.log(f"PDB saved: {pdb_path}")

        csv_path = os.path.join(self.output_dir, f'{self.chr_name}_assembled_global.csv')
        bin_ids = np.arange(self.N)
        if self.valid_bin_starts is not None and len(self.valid_bin_starts) == self.N:
            start_bp = self.valid_bin_starts.astype(int)
        else:
            start_bp = (self.G0 + bin_ids * self.resolution).astype(int)
        end_bp = start_bp + int(self.resolution)

        df = pd.DataFrame({
            'chr': [self.chr_name] * self.N,
            'bin_id': bin_ids,
            'start_bp': start_bp.astype(int),
            'end_bp': end_bp.astype(int),
            'x': self.X_refined[:, 0],
            'y': self.X_refined[:, 1],
            'z': self.X_refined[:, 2]
        })
        df.to_csv(csv_path, index=False)
        self.log(f"CSV saved: {csv_path}")

        mapping_path = os.path.join(self.output_dir, f'{self.chr_name}_coordinate_mapping.csv')
        self._save_coordinate_mapping(mapping_path)
        self.log(f"Coordinate mapping file saved: {mapping_path}")

    def _save_coordinate_mapping(self, mapping_path):
        bin_ids = np.arange(self.N)
        if self.valid_bin_starts is not None and len(self.valid_bin_starts) == self.N:
            start_bp = self.valid_bin_starts.astype(int)
        else:
            start_bp = (self.G0 + bin_ids * self.resolution).astype(int)
        end_bp = start_bp + int(self.resolution)

        owner_list = []
        structure_id_list = []
        local_index_list = []
        mb_key_list = []
        is_gap_list = []
        backbone_bead_list = []

        for t in range(self.N):
            if self.owner[t] is not None:
                owner_list.append(self.owner[t])
                s = self.structure_id[t]
                structure_id_list.append(s)
                local_index_list.append(self.local_index[t])

                if s >= 0 and s < len(self.mb_info):
                    mb_key_list.append(self.mb_info[s]['key'])
                else:
                    mb_key_list.append('')

                is_gap_list.append(False)

                if s in self.mb_anchors:
                    backbone_bead_list.append(self.mb_anchors[s])
                else:
                    backbone_bead_list.append(-1)
            else:
                owner_list.append('')
                structure_id_list.append(-1)
                local_index_list.append(-1)
                mb_key_list.append('')
                is_gap_list.append(True)
                backbone_bead_list.append(-1)

        df_mapping = pd.DataFrame({
            'chr': [self.chr_name] * self.N,
            'bin_id': bin_ids,
            'start_bp': start_bp.astype(int),
            'end_bp': end_bp.astype(int),
            'x': self.X_refined[:, 0],
            'y': self.X_refined[:, 1],
            'z': self.X_refined[:, 2],
            'owner_mb': owner_list,
            'structure_id': structure_id_list,
            'local_index': local_index_list,
            'mb_key': mb_key_list,
            'is_gap': is_gap_list,
            'backbone_bead': backbone_bead_list
        })

        df_mapping.to_csv(mapping_path, index=False)

    def _ensure_minimum_separation(self, coords, min_sep=0.001):
        coords_fixed = coords.copy()

        for i in range(len(coords_fixed) - 1):
            diff = coords_fixed[i+1] - coords_fixed[i]
            dist = np.linalg.norm(diff)

            if dist < min_sep:
                offset = np.array([0.0, 0.0, 0.0])

                if i > 0:
                    prev_diff = coords_fixed[i] - coords_fixed[i-1]
                    prev_dist = np.linalg.norm(prev_diff)
                    if prev_dist > min_sep:
                        offset = (min_sep / prev_dist) * prev_diff
                    else:
                        offset = np.array([min_sep, 0.0, 0.0])
                else:
                    offset = np.array([min_sep, 0.0, 0.0])

                coords_fixed[i+1] = coords_fixed[i] + offset

        return coords_fixed

    def _prepare_pdb_export_coords(self, coords):
        xyz = np.array(coords, dtype=float, copy=True)
        if xyz.size == 0:
            return xyz, 1.0, np.zeros(3, dtype=float)

        xyz[~np.isfinite(xyz)] = 0.0

        min_allowed = -999.999
        max_allowed = 9999.999
        margin = 0.01
        min_t = min_allowed + margin
        max_t = max_allowed - margin
        allowed_span = max_t - min_t

        mins = np.min(xyz, axis=0)
        maxs = np.max(xyz, axis=0)
        spans = maxs - mins
        span_max = float(np.max(spans))

        scale = 1.0
        if span_max > allowed_span and span_max > 0:
            scale = float(allowed_span / span_max)
            xyz *= scale
            mins = np.min(xyz, axis=0)
            maxs = np.max(xyz, axis=0)

        target_mid = 0.5 * (min_t + max_t)
        current_mid = 0.5 * (mins + maxs)
        shift = target_mid - current_mid
        xyz += shift

        for a in range(3):
            amin = float(np.min(xyz[:, a]))
            amax = float(np.max(xyz[:, a]))
            if amin < min_t:
                xyz[:, a] += (min_t - amin)
            if amax > max_t:
                xyz[:, a] += (max_t - amax)

        return xyz, scale, shift

    def _write_pdb(self, coords, out_path):
        coords_fixed = self._ensure_minimum_separation(coords, min_sep=0.001)
        coords_pdb, pdb_scale, pdb_shift = self._prepare_pdb_export_coords(coords_fixed)

        residues_per_chain = 9999
        chain_symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        max_chains = len(chain_symbols)
        n_atoms = len(coords_pdb)

        n_chains_needed = int(np.ceil(max(1, n_atoms) / float(residues_per_chain)))
        if n_chains_needed > max_chains:
            self.log(
                f"PDB export requires {n_chains_needed} chains but only {max_chains} symbols are available; "
                f"chain IDs will wrap.",
                "WARN"
            )
        if abs(float(pdb_scale) - 1.0) > 1e-9:
            self.log(f"PDB export applied uniform scale {pdb_scale:.6f} to fit fixed-width fields.", "WARN")

        self.diagnostics.setdefault('stage_J', {})
        self.diagnostics['stage_J']['pdb_export_transform'] = {
            'scale': float(pdb_scale),
            'shift': [float(pdb_shift[0]), float(pdb_shift[1]), float(pdb_shift[2])],
            'residues_per_chain': int(residues_per_chain),
            'n_chains_used': int(n_chains_needed),
        }

        with open(out_path, 'w') as f:
            for i, (x, y, z) in enumerate(coords_pdb, start=1):
                chain_idx = (i - 1) // residues_per_chain
                chain_id = chain_symbols[chain_idx % max_chains]
                res_seq = ((i - 1) % residues_per_chain) + 1
                serial = ((i - 1) % 99999) + 1
                f.write(
                    f"ATOM  {serial:5d}  CA  MET {chain_id}{res_seq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )

                if (res_seq == residues_per_chain) and (i < n_atoms):
                    f.write("TER\n")

            f.write("TER\n")
            f.write("END\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Global Chromosome Assembly Pipeline")
    parser.add_argument('--mb-dir', type=str, required=True,
                       help='Directory containing MB structures and logs')
    parser.add_argument('--backbone', type=str, required=True,
                       help='Path to 1Mb backbone PDB')
    parser.add_argument('--hic', type=str, required=True,
                       help='Path to Hi-C matrix (3-column)')
    parser.add_argument('--coordinate-mapping', type=str, default=None,
                       help='Optional coordinate mapping file (first column = valid bin start bp). If set, global registry is restricted to these bins.')
    parser.add_argument('--chr', type=str, default='chr14',
                       help='Chromosome name')
    parser.add_argument('--resolution', type=int, default=10000,
                       help='Resolution in bp')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode')
    parser.add_argument('--num-threads', type=int, default=30,
                       help='Number of threads for parallel processing (default: 30)')

    parser.add_argument('--stage-c-pruning-profile', type=str, default='auto',
                       choices=['auto', 'legacy', 'practical-5kb'],
                       help='Stage C pruning profile (default: auto; auto uses practical settings at <=5kb)')
    parser.add_argument('--stage-c-inter-mode', type=str, default='auto',
                       choices=['auto', 'topk', 'distance-stratified'],
                       help='Inter-contact pruning mode (default: auto)')
    parser.add_argument('--stage-c-inter-topk-adj', type=int, default=None,
                       help='Override top-K cap for adjacent structure pairs (s,s+1)')
    parser.add_argument('--stage-c-inter-topfrac-adj', type=float, default=None,
                       help='Override top-fraction keep rate for adjacent structure pairs')
    parser.add_argument('--stage-c-inter-minkeep-adj', type=int, default=None,
                       help='Override minimum contacts to keep for adjacent structure pairs')
    parser.add_argument('--stage-c-inter-topk-skip1', type=int, default=None,
                       help='Override top-K cap for skip-1 structure pairs (s,s+2)')
    parser.add_argument('--stage-c-inter-topfrac-skip1', type=float, default=None,
                       help='Override top-fraction keep rate for skip-1 structure pairs')
    parser.add_argument('--stage-c-inter-minkeep-skip1', type=int, default=None,
                       help='Override minimum contacts to keep for skip-1 structure pairs')
    parser.add_argument('--stage-c-intra-max-contacts', type=int, default=None,
                       help='Override max intra contacts retained per structure')
    parser.add_argument('--stage-c-strata-bp', type=str, default=None,
                       help='Distance-stratified cutoffs in bp as "near,mid" (default: 2000000,10000000)')
    parser.add_argument('--stage-c-strata-ratios', type=str, default=None,
                       help='Distance-stratified keep ratios as "near,mid,far" (default: 0.45,0.35,0.20)')
    parser.add_argument('--stage-c-strata-min-keep', type=int, default=None,
                       help='Minimum contacts to keep per non-empty distance stratum (default: 40)')

    parser.add_argument('--stage-g-max-sweeps', type=int, default=10,
                       help='Max Gauss–Seidel sweeps in Stage G (default: 10)')
    parser.add_argument('--stage-g-convergence-threshold', type=float, default=1e-4,
                       help='Relative stress-improvement threshold for Stage G convergence (default: 1e-4)')
    parser.add_argument('--stage-g-lambda-inter', type=float, default=1.0,
                       help='Weight on inter-contact stress in Stage G (default: 1.0)')
    parser.add_argument('--stage-g-mu-translation', type=float, default=0.05,
                       help='Translation anchoring strength in Stage G (default: 0.05)')
    parser.add_argument('--stage-g-huber-delta', type=float, default=None,
                       help='Huber delta for Stage G robust stress (default: auto from distances)')
    parser.add_argument('--stage-g-max-translation-step', type=float, default=2.0,
                       help='Max translation magnitude per structure update in Stage G (default: 2.0)')
    parser.add_argument('--stage-g-max-rotation-step', type=float, default=0.6,
                       help='Max axis-angle rotation magnitude (radians) per structure update in Stage G (default: 0.6)')
    parser.add_argument('--stage-g-per-structure-maxiter', type=int, default=60,
                       help='Max L-BFGS-B iterations per structure update in Stage G (default: 60)')

    parser.add_argument('--stage-g-grad-check', action='store_true',
                       help='Run a finite-difference gradient check for Stage G and exit')
    parser.add_argument('--stage-g-grad-check-structure', type=int, default=0,
                       help='Structure id to gradient-check (default: 0)')
    parser.add_argument('--stage-g-grad-check-tests', type=int, default=3,
                       help='Number of random parameter tests for gradient check (default: 3)')
    parser.add_argument('--stage-g-grad-check-eps', type=float, default=1e-6,
                       help='Finite difference epsilon for gradient check (default: 1e-6)')
    parser.add_argument('--stage-g-grad-check-seed', type=int, default=0,
                       help='RNG seed for gradient check (default: 0)')

    parser.add_argument('--legacy-flexible-stage-i', action='store_true',
                       help='Use legacy flexible per-bead Stage I (default: rigid I1-style refinement)')
    parser.add_argument('--legacy-tail-extrapolation', action='store_true',
                       help='Keep extrapolated telomere gap chain (default: collapse tail gaps to last owned bead)')

    parser.add_argument('--disable-boundary-alignment', action='store_true',
                       help='Disable post-stitch rigid translation alignment at MB boundaries')
    parser.add_argument('--boundary-alignment-window', type=int, default=5,
                       help='Number of beads used on each side of a boundary (default: 5)')
    parser.add_argument('--boundary-alignment-trigger-sigma', type=float, default=6.0,
                       help='Outlier threshold in robust-sigma for triggering a boundary shift (default: 6.0)')
    parser.add_argument('--boundary-alignment-strength', type=float, default=1.0,
                       help='Fraction of boundary centroid gap to remove when triggered (default: 1.0)')

    parser.add_argument('--disable-tail-gap-refinement', action='store_true',
                       help='Disable Stage H terminal-tail contact-aware local refinement')
    parser.add_argument('--tail-gap-refine-max-iter', type=int, default=200,
                       help='Max optimization iterations for Stage H terminal-tail refinement (default: 200)')
    parser.add_argument('--tail-gap-refine-lr', type=float, default=0.03,
                       help='Learning rate for Stage H terminal-tail refinement (default: 0.03)')
    parser.add_argument('--tail-gap-refine-topk-per-bin', type=int, default=24,
                       help='Top-K Hi-C contacts retained per tail bin for local refinement (default: 24)')
    parser.add_argument('--tail-gap-refine-lambda-contact', type=float, default=1.0,
                       help='Contact-fit loss weight for terminal-tail refinement (default: 1.0)')
    parser.add_argument('--tail-gap-refine-lambda-smooth', type=float, default=0.10,
                       help='Smoothness loss weight for terminal-tail refinement (default: 0.10)')
    parser.add_argument('--tail-gap-refine-lambda-bond', type=float, default=0.20,
                       help='Bond-length regularization weight for terminal-tail refinement (default: 0.20)')
    parser.add_argument('--tail-gap-refine-lambda-reg', type=float, default=0.02,
                       help='Initialization-anchoring weight for terminal-tail refinement (default: 0.02)')
    parser.add_argument('--tail-gap-refine-huber-delta', type=float, default=1.0,
                       help='Huber delta for terminal-tail contact residuals (default: 1.0)')

    parser.add_argument('--dscc-eval-universe', type=str, default='contacts',
                       choices=['contacts', 'all-pairs'],
                       help='Final dSCC evaluation universe (default: contacts). Use all-pairs to evaluate over all i<j bin pairs.')
    parser.add_argument('--dscc-all-pairs-zero-policy', type=str, default='epsilon',
                       choices=['epsilon', 'skip-zeros'],
                       help='Zero-IF handling for all-pairs dSCC (default: epsilon).')
    parser.add_argument('--dscc-all-pairs-if-epsilon', type=float, default=1e-6,
                       help='IF epsilon floor used when --dscc-all-pairs-zero-policy=epsilon (default: 1e-6).')
    parser.add_argument('--dscc-all-pairs-max-pairs', type=int, default=None,
                       help='Optional safety cap on number of all-pairs evaluations; None disables cap.')

    args = parser.parse_args()

    pipeline = GlobalAssemblyPipeline(
        mb_dir=args.mb_dir,
        backbone_path=args.backbone,
        hic_path=args.hic,
        chr_name=args.chr,
        resolution=args.resolution,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        num_threads=args.num_threads,
        coordinate_mapping_path=args.coordinate_mapping,
    )

    pipeline.enable_boundary_alignment = (not args.disable_boundary_alignment)
    pipeline.boundary_alignment_window = int(args.boundary_alignment_window)
    pipeline.boundary_alignment_trigger_sigma = float(args.boundary_alignment_trigger_sigma)
    pipeline.boundary_alignment_strength = float(args.boundary_alignment_strength)
    pipeline.collapse_terminal_tail_gaps = not args.legacy_tail_extrapolation
    pipeline.stage_i_rigid_refinement = not args.legacy_flexible_stage_i
    if pipeline.collapse_terminal_tail_gaps:
        pipeline.enable_tail_gap_refinement = False
    else:
        pipeline.enable_tail_gap_refinement = (not args.disable_tail_gap_refinement)
    pipeline.tail_gap_refine_max_iter = int(args.tail_gap_refine_max_iter)
    pipeline.tail_gap_refine_lr = float(args.tail_gap_refine_lr)
    pipeline.tail_gap_refine_topk_per_bin = int(args.tail_gap_refine_topk_per_bin)
    pipeline.tail_gap_refine_lambda_contact = float(args.tail_gap_refine_lambda_contact)
    pipeline.tail_gap_refine_lambda_smooth = float(args.tail_gap_refine_lambda_smooth)
    pipeline.tail_gap_refine_lambda_bond = float(args.tail_gap_refine_lambda_bond)
    pipeline.tail_gap_refine_lambda_reg = float(args.tail_gap_refine_lambda_reg)
    pipeline.tail_gap_refine_huber_delta = float(args.tail_gap_refine_huber_delta)
    pipeline.dscc_eval_universe = str(args.dscc_eval_universe)
    pipeline.dscc_all_pairs_zero_policy = str(args.dscc_all_pairs_zero_policy)
    pipeline.dscc_all_pairs_if_epsilon = float(args.dscc_all_pairs_if_epsilon)
    pipeline.dscc_all_pairs_max_pairs = (
        int(args.dscc_all_pairs_max_pairs) if args.dscc_all_pairs_max_pairs is not None else None
    )

    pipeline.stage_c_pruning_profile = str(args.stage_c_pruning_profile)
    pipeline.stage_c_inter_mode = str(args.stage_c_inter_mode)
    pipeline.stage_c_inter_topk_adj = args.stage_c_inter_topk_adj
    pipeline.stage_c_inter_topfrac_adj = args.stage_c_inter_topfrac_adj
    pipeline.stage_c_inter_minkeep_adj = args.stage_c_inter_minkeep_adj
    pipeline.stage_c_inter_topk_skip1 = args.stage_c_inter_topk_skip1
    pipeline.stage_c_inter_topfrac_skip1 = args.stage_c_inter_topfrac_skip1
    pipeline.stage_c_inter_minkeep_skip1 = args.stage_c_inter_minkeep_skip1
    pipeline.stage_c_intra_max_contacts = args.stage_c_intra_max_contacts
    if args.stage_c_strata_min_keep is not None:
        pipeline.stage_c_strata_min_keep = int(args.stage_c_strata_min_keep)

    def _parse_csv_numbers(text, cast=float):
        vals = [v.strip() for v in str(text).split(',') if v.strip()]
        return [cast(v) for v in vals]

    if args.stage_c_strata_bp is not None:
        try:
            strata_bp = _parse_csv_numbers(args.stage_c_strata_bp, cast=int)
            if len(strata_bp) >= 2:
                pipeline.stage_c_strata_bp = tuple(strata_bp[:2])
        except Exception:
            pipeline.log(f"Invalid --stage-c-strata-bp='{args.stage_c_strata_bp}', using defaults", "WARN")

    if args.stage_c_strata_ratios is not None:
        try:
            strata_ratios = _parse_csv_numbers(args.stage_c_strata_ratios, cast=float)
            if len(strata_ratios) >= 3:
                pipeline.stage_c_strata_ratios = tuple(strata_ratios[:3])
        except Exception:
            pipeline.log(f"Invalid --stage-c-strata-ratios='{args.stage_c_strata_ratios}', using defaults", "WARN")

    pipeline.stage_g_max_sweeps = int(args.stage_g_max_sweeps)
    pipeline.stage_g_convergence_threshold = float(args.stage_g_convergence_threshold)
    pipeline.stage_g_lambda_inter = float(args.stage_g_lambda_inter)
    pipeline.stage_g_mu_translation = float(args.stage_g_mu_translation)
    pipeline.stage_g_huber_delta = args.stage_g_huber_delta
    pipeline.stage_g_max_translation_step = float(args.stage_g_max_translation_step)
    pipeline.stage_g_max_rotation_step = float(args.stage_g_max_rotation_step)
    pipeline.stage_g_per_structure_maxiter = int(args.stage_g_per_structure_maxiter)

    if args.stage_g_grad_check:
        pipeline.log("Running Stage G gradient check mode...")
        pipeline.stage_A_global_bin_registry()
        pipeline.stage_B_backbone_anchoring()
        pipeline.stage_C_inter_contact_selection()
        pipeline._quick_placement_for_alpha()
        pipeline.stage_D_alpha_strategy()
        pipeline.stage_E_scaling()
        pipeline.stage_F_initial_placement()
        pipeline._stage_g_gradient_check(
            structure_id=int(args.stage_g_grad_check_structure),
            n_tests=int(args.stage_g_grad_check_tests),
            eps=float(args.stage_g_grad_check_eps),
            seed=int(args.stage_g_grad_check_seed),
        )
        return None

    result = pipeline.run()

    return result

if __name__ == '__main__':
    main()
