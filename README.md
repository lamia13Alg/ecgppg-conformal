# Patient-specific conformal calibration for arrhythmia monitoring

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Code and results for the paper

> *Patient-specific conformal calibration for arrhythmia monitoring: contamination, reviewer
> dependence and distribution shift.* Submitted to *Expert Systems with Applications*.

The repository contains the full pipeline (feature extraction from ECG and PPG, population models,
conformal p-values with a sliding adjudicated buffer, adaptive conformal inference baseline,
reviewer models, episode-level alarms) and the per-record results of every run. Every number, table
and figure of the paper is regenerated from the files in `results/` by the scripts below.

**No patient data are included.** The two databases are public but must be obtained from their
providers (see *Data*). `results/` only holds per-record summary counts (alarms, reviews, true and
false positives), split assignments and aggregate statistics.

## Contents

```
ecgppg/                 Python package
  signal.py, features.py    ECG/PPG preprocessing, beat-level features
  data.py, pipeline.py      record loading and caching (VitalDB, Vilnius)
  vitaldb_labels.py         beat and rhythm labels from the VitalDB Arrhythmia annotations
  scores.py                 population models and non-conformity scores
  conformal.py, alarm.py    conformal p-values, sliding buffer, CUSUM
  revision.py, revision2.py monitor with reviewer models (errors, delays, pending policies), ACI
  deep.py                   1-D CNN baseline (PyTorch, optional)
  experiments.py, metrics.py splits, bootstrap, beat/episode metrics
  synthetic.py              synthetic streams (simulation study, smoke test)
scripts/                command-line entry points (see the pipeline table)
notebooks/              Kaggle notebook used for the final revision run
results/
  main/                 results_vdb.json, simulation.json, split_vitaldb.json (patient IDs of the split)
  results_rev/          multi-seed and main-split revision analyses (run_revision.py)
  results_rev2/         second-revision analyses (run_revision2*.py)
  paper_tables/         LaTeX macros, tables and figures input by the manuscript
```

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/<USER>/ecgppg-conformal.git
cd ecgppg-conformal
pip install -r requirements.txt
python scripts/smoke_test.py        # ~30 s, synthetic data; checks that everything runs
```

Tested with numpy 2.x, scipy 1.1x, scikit-learn 1.x, pandas 2–3, matplotlib 3.x, vitaldb 1.4+,
h5py 3.x and (for the CNN baseline only) torch 2.x. The analyses were run on Kaggle CPU notebooks.

## Data

| Role | Database | Access |
|---|---|---|
| Development, selection, internal test | VitalDB Arrhythmia Database v1.0.0, https://doi.org/10.13026/axd6-wm13 (labels) + VitalDB waveforms `SNUADC/ECG_II`, `SNUADC/PLETH` (streamed by the `vitaldb` package) | Open (PhysioNet) |
| External test (wearable, multi-day) | Vilnius long-term ECG + wrist PPG with annotated AF, https://doi.org/10.5281/zenodo.23187681 | Open, non-commercial licence |

Do not commit the raw files or the `cache/` folder (they are excluded by `.gitignore`).

## Reproducing the paper

All commands are run from the repository root. Steps 1–2 need Internet access; the analyses
(3–7) only read the feature cache. Long steps are resumable (`--hours` limits one session, e.g. a
12-h Kaggle run; re-running continues where it stopped).

| Step | Command | Output | Paper |
|---|---|---|---|
| 1 | `wget -r -N -c -np -nH --cut-dirs=3 -P physionet https://physionet.org/files/vitaldb-arrhythmia/1.0.0/`<br>`python scripts/prepare_data.py vitaldb --ann physionet/Annotation_Files --cache cache`<br>`python scripts/relabel_vitaldb.py --ann physionet/Annotation_Files --cache cache` | `cache/vitaldb/*.npz` | Data, inclusion |
| 2 | `python scripts/prepare_vilnius_stream.py --cache cache --raw /tmp/raw_vilnius --hours 10` | `cache/vilnius/*.npz` | Data |
| 3 | `python scripts/run_all.py --cache cache --out results/main` | `results_vdb.json`, `split_vitaldb.json`, `macros_vdb.tex`, `tab_vdb*.tex` (copy the `.tex` files to `results/paper_tables/`) | Development, selection, main test |
| 4 | `python scripts/simulation_study.py --out results/main --reps 30` | `simulation.json` | Simulation study |
| 5 | `python scripts/run_revision.py --cache cache --out results/results_rev --ann physionet/Annotation_Files` | `results_rev/*.json` | Multi-split stability, bootstrap, CNN/ACI baselines, reviewer scenarios |
| 6 | `python scripts/run_revision2.py --cache cache --out results/results_rev2`<br>`python scripts/run_revision2b.py --cache cache --out results/results_rev2`<br>`python scripts/run_revision2c.py --cache cache --out results/results_rev2` | `results_rev2/*` | α sweep, ACI comparison, reviewer models, causal filtering, episode budgets |
| 7 | see below | `results/paper_tables/` | all tables, figures and in-text numbers |

Step 7 (tables, macros and figures; seconds, no data needed):

```bash
OUT=results/paper_tables
python scripts/make_figures.py --res results/main/results_vdb.json --out $OUT
python scripts/make_simulation_figure.py results/main/simulation.json $OUT
python scripts/make_revision_tables.py results/results_rev $OUT
python scripts/make_revision2_tables.py results/results_rev2 results/results_rev $OUT
```

Running step 7 on the shipped `results/` reproduces the files in `results/paper_tables/` byte for
byte. Main settings: α = 1 %, buffer M = 300, gap G = 20, start-up review q = 5 %, main split seed 42,
ten additional splits (seeds 0–9).

### On Kaggle

`notebooks/kaggle_final_revision.ipynb` is the notebook used for the final run (Accelerator: None,
Internet: On). It embeds a copy of this code, rebuilds the cache if no cache dataset is attached and
runs steps 6b–6c; a run may need two 12-h sessions.

Other scripts: `smoke_test.py` (end-to-end check on synthetic data), `inspect_vilnius.py`
(prints the field names of the Vilnius MAT files), `run_cinc2015.py` (exploratory analysis on the
PhysioNet/CinC 2015 challenge set, not reported in the paper).

## Citation

If you use this code, please cite the paper and the archived software (see `CITATION.cff`).

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences; the Vilnius database is
restricted to non-commercial use.
