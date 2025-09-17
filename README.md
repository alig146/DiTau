# B2TF

A Boosted Di-tau Training Framework

### Overview
This repository contains two main ML pipelines for the boosted di-tau analysis:
- analysis_mva: Main analysis BDTs (Run 2 and Run 3 workflows) for the boosted di-tau selection.
- stxs_mva: A 3-class BDT classifier (Background, VBF_H, ggF_H) used for the STXS categorization.

### Project Structure
```
DiTau/
├── analysis_mva/                         # Main boosted di-tau MVA workflows
│   ├── analysis_mva_run2.ipynb          # Run 2 data/MC processing and BDT training
│   ├── analysis_mva_run3.ipynb          # Run 3 data/MC processing and BDT training
│   └── ...                              # Supporting notebooks and assets
├── stxs_mva/                            # STXS 3-class classifier workflow
│   ├── bdt_training.py                  # Main script to train Background/VBF/ggF classifier
│   ├── data_processing.py               # Utilities to build training DataFrame from pickles
│   └── ...
├── utils/                                # Shared utilities
│   ├── mva_utils.py                     # Physics variables, I/O, cuts, and helpers
│   └── ...
├── data/                                 # Staging area for prebuilt pickles or inputs
│   └── (generated externally; see note below)
├── environment.yml                       # Conda environment specification
├── requirements.txt                      # Python dependencies (pip)
└── README.md                             # This file
```

### Data inputs (important)
- The ROOT inputs and derived pickles used by the notebooks and training scripts are produced by the BOOM pipeline. See BOOM here: [BOOM](https://gitlab.cern.ch/ATauLeptonAnalysiS/boom).
- The contents of `data/` may change when BOOM is updated. Always verify that your local inputs are consistent with the current BOOM configuration before training.
- Typical inputs for stxs_mva training:
  - `data/raw_mc_run2.pkl` and `data/raw_data_run2.pkl` (Run 2)
  - These are consumed by `stxs_mva/data_processing.py` and `stxs_mva/bdt_training.py`.

### analysis_mva (main boosted di-tau BDTs)
- Primary notebooks:
  - `analysis_mva_run2.ipynb`: end-to-end Run 2 workflow (load inputs, apply cuts, build variables, train/evaluate BDTs).
  - `analysis_mva_run3.ipynb`: same for Run 3.
- These notebooks use helpers from `utils/mva_utils.py`:
  - Data/MC reading, trigger application, selections (`apply_cuts`), variable construction (`Var`, `Data_Var`), year-combining helpers.

### stxs_mva (3-class STXS classifier)
- Main training script: `stxs_mva/bdt_training.py`
  - Loads training data from prebuilt pickles using `stxs_mva/data_processing.py` (Run 2 by default).
  - Trains an XGBoost 3-class classifier with classes: `Background`, `VBF_H`, `ggF_H`.
  - Saves basic evaluation plots (confusion matrix, ROC curves, score distributions) and SHAP feature importances.
- Data assembly helper: `stxs_mva/data_processing.py`
  - Loads uncut MC/data pickles, applies standard cuts via `utils/mva_utils.py`, combines years, builds variables, and assembles a labeled DataFrame.
  - Signals: `ggH` (label 2), `VBFH` (label 1)
  - Backgrounds (explicit): `Ztt_inc`, `VV`, `Top`, `W`, `Zll_inc`, `ttV` (label 0); data is also treated as background (label 0).

### Environment
- Create environment with conda:
  - `conda env create -f environment.yml`
  - `conda activate <env>`
- Or install with pip: `pip install -r requirements.txt`

### Quick start
1) Generate or update inputs using BOOM. Confirm pickles in `data/` are current (see note above).
2) Run main boosted analysis notebooks:
   - `analysis_mva/analysis_mva_run2.ipynb`
   - `analysis_mva/analysis_mva_run3.ipynb`
3) Train STXS 3-class classifier:
   - `python stxs_mva/bdt_training.py`

### Notes
- The utilities in `utils/mva_utils.py` centralize data reading, weighting, trigger masks, selections, and physics variable building for both pipelines.
- If you change triggers, years, or DSID groups, update `mva_utils.py` and regenerate inputs from BOOM accordingly.

