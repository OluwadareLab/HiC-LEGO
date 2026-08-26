# HiC-LEGO

**HiC-LEGO: Biologically Guided High-Resolution 3D Genome Reconstruction Preserves Chromatin Organization at Kilobase Resolution**

HiC-LEGO reconstructs 3D chromosome structures from Hi-C data by building domain-level models, generating MB region (LEGO) structures, and assembling them into a global chromosome model. Implemented in **Python 3.10** with **PyTorch** (GPU preferred; CPU supported).

---

## System requirements

| Item | Details |
|------|---------|
| Python | 3.10 |
| Environment | Conda |
| Key packages | PyTorch, PyTorch Geometric, NumPy, SciPy, pandas, NetworkX, scikit-learn, Numba |

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/OluwadareLab/HiC-LEGO.git
cd HiC-LEGO
```

### 2. Environment setup

**GPU (default)** — uses CUDA PyTorch when a GPU is available:

```bash
conda env create -f environment.yml
conda activate hiclego
```

**CPU only** 

```bash
conda env create -f environment-cpu.yml
conda activate hiclego-cpu
```


### 3. Input requirements

Example input files can be found in our zenodo repository. 

Place inputs under `inputs/` (or pass specific paths). An example chromosome folder looks like:

```text
inputs/<experiment>_chr22/
├── domains/              # one or more domain-list files
├── chr22_5kb.txt         # fine-resolution Hi-C contact list
└── chr22_1mb.txt         # 1 Mb Hi-C contact list
```

Required inputs:

- **Domain lists** — directory of domain list files  
- **Fine-resolution Hi-C** — contact list at the target resolution (e.g. 5 kb) in 3 column Hi-C interaction format
- **1 Mb Hi-C** — used for the backbone structure in 3 column Hi-C interaction format  

Step 0 of `run_all.py` prepares optimal domains, the 1 Mb backbone PDB, and the coordinate mapping into `src/preprocessing/<experiment>/`.

### 4. Run the pipeline

Main entry point: **`run_all.py`**.

**For example input folder:**

```bash
python run_all.py \
  --experiment gm12878 \
  --chr chr22 \
  --res 5kb \
  --suffix h2 \
  --input-dir inputs/gm12878_chr22
```

**Using explicit paths:**

```bash
python run_all.py \
  --experiment gm12878 \
  --chr chr22 \
  --res 5kb \
  --suffix h2 \
  --domains-dir /path/to/domains \
  --hic-matrix /path/to/chr22_5kb.txt \
  --hic-1mb /path/to/chr22_1mb.txt
```

Useful options:

| Flag | Meaning |
|------|---------|
| `--experiment` | Cell line / experiment name (required) |
| `--chr` | Chromosome (default: `chr22`) (required) |
| `--res` | Resolution label or bp (`5kb`, `10kb`, `5000`, …) (required) |
| `--suffix` | Run tag (default: `h2`) (Required) |


### 5. Where outputs live

For experiment `gm12878` and run id `chr22_5kb_h2`:

| Stage | Location |
|-------|----------|
| Complete log | `logs/gm12878/chr22_5kb_h2_full_pipeline.log` |
| Prepared inputs | `src/preprocessing/gm12878/chr22_5kb_h2/` |
| Domain structures | `src/intra_domains/outputs/gm12878/chr22_5kb_h2/` |
| Micro-blocks | `assembly/mb_generation/outputs/gm12878/chr22_5kb_h2_fixed_optimaldomains/` |
| **Final global structure** | `assembly/global_assembly/outputs/gm12878/output_chr22_5kb_h2_global/` |

Final assembly includes PDB/CSV coordinates and diagnostics under the global output folder.

---


