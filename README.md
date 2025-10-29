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

## Notes

* Keep `data/raw/`, `data/interim/`, `results/` out of git. Add exceptions only for small files.
* For very large raw files use Actions download or Releases, not LFS.
* All metrics are reproducible given the `seed` and fixed splits.


## What it does

* Loads GSE45827, parses GSM metadata, builds a labeled cohort.
* Maps probes→genes (GPL570), collapses probes, z-scores genes.
* Trains baselines, runs GA feature selection, then model selection.
* Exports compact JSON for a web matrix/heatmap and metrics.


## Pipeline (steps 01→10)

| Step | Module                           | Purpose                                       | Key outputs                                     |
| ---- | -------------------------------- | --------------------------------------------- | ----------------------------------------------- |
| 01   | `step01_reading_data.py`         | Read series matrix, parse SOFT metadata       | `X_raw.tsv.gz`, `metadata.tsv`, `gsm_order.tsv` |
| 02   | `step02_preprocessing.py`        | Probe→gene map, collapse, log2 check, z-score | `X.tsv.gz`                                      |
| 03   | `step03_cohort.py`               | Build labels and cohort filters               | `X_cohort.tsv.gz`, `y.tsv`                      |
| 04   | `step04_split.py`                | Stratified train/val/test                     | `split_train.csv` etc.                          |
| 05   | `step05_baseline.py`             | Baselines (`logreg_en`, `linsvc`, `rf`)       | `results/baseline/*`                            |
| 06   | `step06_GA_feature_selection.py` | GA over gene masks, CV scoring                | `ga_mask_k300.tsv`, `ga_genes_k300.txt`, GA log |
| 07   | `step07_model_selection.py`      | Train chosen model on GA genes + CV           | `results/model_selection/*`, compare TSV        |
| 08   | `step08_bio_checks.py`           | PAM50 overlap, simple enrichment              | JSON summary                                    |
| 09   | `step09_unsupervised.py`         | PCA/UMAP + clustering                         | plots, scores                                   |
| 10   | `step10_differential.py`         | Simple DE and visuals                         | tables, volcano/heatmap                         |

## GA fitness

* Chromosome: binary mask over genes.
* Fitness: **macro-F1** from K-fold CV of the chosen estimator.
* Operators: tournament selection, crossover, bit-flip mutation.
* Params: population, generations, crossover prob, mutation prob.

## Web UI (Pages)

Static app under `docs/`:

* Shows matrix (top-variance subset) as table or heatmap.
* Displays selected model + GA params.
* Loads metrics from `docs/data/results.json`.

## Quick start (local)

```bash
# Python 3.10+
pip install -r requirements.txt
python main.py          # runs steps in order
python tools/prepare_web.py  # writes docs/data/*.json
# open docs/index.html in a browser
```

## Run on GitHub Actions

* Workflow: `.github/workflows/run_ga.yml`
* Settings → Pages → Source = **GitHub Actions**
* Actions → **run-ga** → set inputs:

  * `model`: `logreg_en` | `linsvc` | `rf`
  * `population_size`, `generations`, `crossover_prob`, `mutation_prob`
* The job downloads GEO, runs steps 01–07, writes `docs/data/results.json`, and publishes Pages.


## Notes

* `data/raw/`, `data/interim/`, `results/` are git-ignored. Large GEO files are fetched by the workflow.
* Reproducibility via `seed` and fixed splits in `config.yaml`.
