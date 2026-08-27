# HiC-LEGO

**HiC-LEGO: Biologically Guided High-Resolution 3D Genome Reconstruction Preserves Chromatin Organization at Kilobase Resolution**

HiC-LEGO reconstructs 3D chromosome structures from Hi-C data by building domain-level models, generating MB region (LEGO) structures, and assembling them into a global chromosome model. Implemented in **Python 3.10** with **PyTorch** (GPU preferred; CPU supported).

---

## System requirements

| Item | Details |
|------|---------|
| Python | 3.10 |
| Environment | Conda |
| Key packages | PyTorch, PyTorch Geometric, NumPy, SciPy, pandas, NetworkX, scikit-learn, Numba, hic-straw |

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

Example input files can be found in our zenodo repository. Paths may live under `inputs/` or anywhere; pass them explicitly on the CLI.

Every run needs to pass **domains** (which is a **folder** containing any number of domain list files in .txt format; the pipeline builds an optimal domain list from them). For input Hi-C matrices, choose **one** of the following:

**A. `.hic` file** — the pipeline extracts KR-normalized fine-resolution and 1 Mb contact lists for specified `--chr` / `--res` via [hic-straw](https://pypi.org/project/hic-straw/) (that resolution and KR must already exist in the `.hic`):

```text
domains/          # a folder with one/multiple domain-list files
file.hic          # .hic file
```

**B. Prebuilt contact lists** — fine-resolution (e.g. 5 kb) and 1 Mb matrices in 3-column format (`bin1_start`, `bin2_start`, `IF`):

```text
domains/          # domain-list files
chr22_5kb.txt     # fine-resolution contacts
chr22_1mb.txt     # 1 Mb backbone contacts
```

### 4. Run the pipeline

Main entry point: **`run_all.py`**.

**From a `.hic` file** (domains still required):

```bash
python run_all.py \
  --experiment gm12878 \
  --chr chr22 \
  --res 5kb \
  --suffix h2 \
  --domains-dir /path/to/domains \
  --hic-file /path/to/file.hic
```

**From already extracted contact matrices** 

```bash
python run_all.py \
  --experiment gm12878 \
  --chr chr22 \
  --res 5kb \
  --suffix h2 \
  --domains-dir /path/to/domains \
  --hic-matrix /path/to/chr_5kb.txt \
  --hic-1mb /path/to/chr_1mb.txt
```


| Flag | Meaning |
|------|---------|
| `--experiment` | Cell line / experiment name (required) |
| `--chr` | Chromosome (default: `chr22`) (required) |
| `--res` | Resolution label or bp (`5kb`, `10kb`, `5000`, …) (required) |
| `--suffix` | Run tag (default: `h2`) (Required) |
| `--hic-file` | `.hic` path; extracts KR fine + 1 Mb contact lists |


### 5. Where outputs live

For experiment `gm12878` and run id `chr22_5kb_h2`:

| Stage | Location |
|-------|----------|
| Complete log | `logs/gm12878/chr22_5kb_h2_full_pipeline.log` |
| Prepared inputs | `src/preprocessing/gm12878/chr22_5kb_h2/` |
| Domain structures | `src/intra_domains/outputs/gm12878/chr22_5kb_h2/` |
| MB LEGO Blocks | `assembly/mb_generation/outputs/gm12878/chr22_5kb_h2_fixed_optimaldomains/` |
| **Final global structure** | `assembly/global_assembly/outputs/gm12878/output_chr22_5kb_h2_global/` |

Final assembly includes PDB/CSV coordinates and diagnostics under the global output folder.

---


