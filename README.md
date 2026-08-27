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

Example input files are available on our [Zenodo repository](https://doi.org/10.5281/zenodo.22085228). Paths may live under `inputs/` or anywhere; pass them explicitly on the CLI.

Every run needs **domains** (a folder of domain list `.txt` files; the pipeline builds an optimal domain list from them). For Hi-C, choose **one** of the following:

| Option | What you provide | Walkthrough |
|--------|------------------|-------------|
| **A. `.hic` file** | `domains/` + a Juicebox `.hic` | [Example 1](#example-1-from-a-hic-file-gm12878-chr22--5-kb) |
| **B. Prebuilt contact lists** | `domains/` + fine + 1 Mb matrices | [Example 2](#example-2-from-prebuilt-contact-lists-gm12878-chr22--5-kb) |

**A. `.hic` file** — the pipeline extracts KR-normalized fine-resolution and 1 Mb contact lists for `--chr` / `--res` via [hic-straw](https://pypi.org/project/hic-straw/) (that resolution and KR must already exist in the `.hic`):

```text
domains/          # a folder with one/multiple domain-list files
file.hic          # .hic file
```

**B. Prebuilt contact lists** — fine-resolution (e.g. 5 kb) and 1 Mb matrices in 3-column format (`bin1_start`, `bin2_start`, `IF`):

```text
domains/          # a folder with one/multiple domain-list files
chr22_5kb.txt     # fine-resolution 3 column contact matrix
chr22_1mb.txt     # 1 Mb backbone 3 column contact matrix
```

### 4. Example runs

Main entry point: **`run_all.py`**.

**A. From a `.hic` file** (extracts fine + 1 Mb contact lists; domains still required):

```bash
python run_all.py \
  --experiment <cell_line> \
  --chr <chr> \
  --res <resolution> \
  --suffix <run_tag> \
  --domains-dir /path/to/domains \
  --hic-file /path/to/file.hic
```

**B. From prebuilt contact lists:**

```bash
python run_all.py \
  --experiment <cell_line> \
  --chr <chr> \
  --res <resolution> \
  --suffix <run_tag> \
  --domains-dir /path/to/domains \
  --hic-matrix /path/to/<chr>_<res>.txt \
  --hic-1mb /path/to/<chr>_1mb.txt
```

Step-by-step examples for GM12878 chr22 @ 5 kb follow below.

#### Example 1: From a `.hic` file (GM12878 chr22 at 5 kb)

**1. Download the Hi-C file**

Access the [GSE63525 GEO entry](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525) for Hi-C data from Rao et al. (2014). In our work we used `GSE63525_GM12878_insitu_primary+replicate_combined.hic` (~51 GB).

```bash
cd inputs/gm12878_with_hic

wget -c ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_insitu_primary+replicate_combined.hic
```

**2. Domains folder**

Example domain lists for GM12878 chr22 are under `inputs/gm12878_with_hic/domains/`. Expected layout:

```text
inputs/gm12878_with_hic/
├── domains/
│   ├── chr22_arrowhead.txt
│   ├── chr22_TopDom_domains_15L.txt
│   ├── chr22_spectraltad_domains_75L.txt
│   └── chr22_gm12878_hickey_domains.txt
└── GSE63525_GM12878_insitu_primary+replicate_combined.hic
```

**3. Run the pipeline**

```bash
conda activate hiclego
cd HiC-LEGO

python run_all.py \
  --experiment gm12878 \
  --chr chr22 \
  --res 5kb \
  --suffix h2 \
  --domains-dir inputs/gm12878_with_hic/domains \
  --hic-file inputs/gm12878_with_hic/GSE63525_GM12878_insitu_primary+replicate_combined.hic
```

#### Example 2: From prebuilt contact lists (GM12878 chr22 @ 5 kb)

**1. Download example inputs**

Download `example inputs_gm12878_chr22.zip` from our [Zenodo repository](https://doi.org/10.5281/zenodo.22085228) (~67 MB). It contains the domain lists, fine-resolution chr22 matrix, and 1 Mb chr22 matrix used in our example runs.

```bash
mkdir -p inputs
cd inputs

wget -c "https://zenodo.org/records/22085228/files/example%20inputs_gm12878_chr22.zip?download=1" \
  -O example_inputs_gm12878_chr22.zip
```

**2. Unzip**

```bash
unzip example_inputs_gm12878_chr22.zip -d gm12878_chr22
```

Expected layout after unzipping:

```text
inputs/gm12878_chr22/
├── domains/
│   ├── chr22_arrowhead.txt
│   ├── chr22_TopDom_domains_15L.txt
│   ├── chr22_spectraltad_domains_75L.txt
│   └── chr22_gm12878_hickey_domains.txt
├── chr22_5kb.txt
└── chr22_1mb.txt
```

**3. Run the pipeline**

```bash
conda activate hiclego
cd HiC-LEGO

python run_all.py \
  --experiment gm12878 \
  --chr chr22 \
  --res 5kb \
  --suffix h2 \
  --domains-dir inputs/gm12878_chr22/domains \
  --hic-matrix inputs/gm12878_chr22/chr22_5kb.txt \
  --hic-1mb inputs/gm12878_chr22/chr22_1mb.txt
```

#### CLI options

| Flag | Meaning |
|------|---------|
| `--experiment` | Cell line / experiment name (required) |
| `--chr` | Chromosome (default: `chr22`) |
| `--res` | Resolution label or bp (`5kb`, `10kb`, `5000`, …) |
| `--suffix` | Run tag (default: `h2`) |
| `--hic-file` | `.hic` path; extracts KR fine + 1 Mb contact lists |
| `--skip-hic-extract` | With `--hic-file`: reuse existing extracted matrices if present |

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
