# Breast Cancer Subtypes · GA Feature Selection (GSE45827)

End-to-end pipeline to select genes with a Genetic Algorithm and classify breast cancer subtypes on **GSE45827** (Affymetrix **GPL570**). Includes a static web UI (GitHub Pages) to browse the matrix, pick GA parameters, and view results.

## Quick start

```bash
# Python 3.10+
pip install -r requirements.txt

# Configure
# edit config.yaml as needed (GA params, paths, splits)

# Run full pipeline
python main.py
```

Key outputs:

* `data/interim/X_raw.tsv.gz` – probes × GSMs from series matrix
* `data/processed/X.tsv.gz` – genes × GSMs (collapsed, z-scored)
* `data/processed/X_cohort.tsv.gz`, `data/processed/y.tsv` – cohort and labels
* `data/processed/gsm_order.tsv`, `data/processed/metadata.tsv`
* `data/processed/ga_mask_k300.tsv`, `data/processed/ga_genes_k300.txt`
* `results/baseline/*`, `results/model_selection/*`
* `docs/data/*.json` – web site data

> Raw GEO files are **not** stored in the repo. The workflow downloads them.

## Data

* Series: **GSE45827**
* Platform: **GPL570** (HG-U133 Plus 2.0)
* Files used:
  `GSE45827_series_matrix.txt.gz`, `GSE45827_family.soft.gz`, `GPL570.annot.gz`

## Pipeline steps

1. **step01_reading_data**
   Read series matrix. Parse GSM metadata from SOFT. Save `X_raw.tsv.gz`, `metadata.tsv`, `gsm_order.tsv`.
2. **step02_preprocessing**
   Map probe→gene via GPL570 SOFT/annot. Collapse probes by median. Log2 check. Per-gene z-score. Save `X.tsv.gz`.
3. **step03_cohort**
   Build labels (Basal, HER2, LumA, LumB). Filter cohort. Align `X` and `y`.
4. **step04_split**
   Stratified train/val/test splits. Save `split_*.csv`.
5. **step05_baseline**
   Baselines: `logreg_en`, `linsvc`, `rf`. Evaluate on splits. Save metrics.
6. **step06_GA_feature_selection**
   GA over binary gene masks. Fitness = cross-val **f1_macro** of the estimator. Save `ga_mask_k300.tsv`, `ga_genes_k300.txt`, GA history.
7. **step07_model_selection**
   Train the chosen model on GA genes. CV for hyper-params. Report train/val/test and test-after-refit metrics. Write comparison TSV.
8. **step08_bio_checks**
   PAM50 overlap, simple marker checks, optional pathway enrichment (`gseapy`).
9. **step09_unsupervised**
   PCA/UMAP and clustering. Plots + silhouette where applicable.
10. **step10_differential**
    Simple DE per subtype vs rest. Volcano/heatmap and marker tables.

## GA fitness

* Objective: maximize **f1_macro** (K-fold CV as set in `config.yaml`).
* Chromosome: binary vector over genes.
* Operators: tournament selection, one/two-point crossover, bit-flip mutation.
* Constraints: population size, generations, crossover and mutation probabilities come from config or workflow inputs.

## Web site (GitHub Pages)

Static UI under `docs/`:

* Matrix viewer (top-variance subset) as heatmap or table.
* Controls show chosen model and GA parameters.
* Results panel reads `docs/data/results.json`.

Build site data locally:

```bash
python tools/prepare_web.py
# open docs/index.html
```

Deploy with GitHub Actions:

* Workflow: `.github/workflows/run_ga.yml`
* Settings → Pages → **Source: GitHub Actions**
* Actions → **run-ga** → set inputs → Run
  The job downloads GEO, runs steps 01–07, writes `docs/data/*.json`, and publishes Pages.

## Config

```yaml
seed: 42
gse: GSE45827
platform: GPL570
classes: [Basal, HER2, LumA, LumB]
split: {train: 0.6, val: 0.2, test: 0.2}
ga:
  estimator: logreg_en
  population_size: 80
  generations: 40
  crossover_prob: 0.8
  mutation_prob: 0.05
  cv_folds: 5
  scoring: f1_macro
paths:
  raw: data/raw
  interim: data/interim
  processed: data/processed
  results: results
```

## Repository layout

```
src/
  step01_reading_data.py
  step02_preprocessing.py
  step03_cohort.py
  step04_split.py
  step05_baseline.py
  step06_GA_feature_selection.py
  step07_model_selection.py
  step08_bio_checks.py
  step09_unsupervised.py
  step10_differential.py
  utils.py
tools/
  prepare_web.py
docs/
  index.html
  app.js
  data/
    matrix.json
    labels.json
    results.json
.github/workflows/
  run_ga.yml
config.yaml
requirements.txt
README.md
```

Download family	Format
SOFT formatted family file(s): https://ftp.ncbi.nlm.nih.gov/geo/series/GSE45nnn/GSE45827/soft/
GPL570.annot.gz: https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/soft/
