# Brain-to-Text '25 Kaggle Competition
## Project Overview

This repository contains code and documentation for our team's solution to the [Brain-to-Text '25 Kaggle competition](https://www.kaggle.com/competitions/brain-to-text-25/overview). The challenge is to **decode intracortical neural activity during attempted speech into corresponding text**, pushing forward the field of neural signal decoding and brain-computer interfaces.

- **Team Name:** Neural Navigators
- **Team Members:** 5
- **Competition Dates:** September–December 2025

### Team & Communication

- **Team Members:** George Yu, Carl Shashuk, Rohini G, Nils Matteson, Daniel Yang
- **Communication:** [Slack Channel](https://data-science-hubgroup.slack.com/archives/C09EUS28ZEJ)

### Repository Structure
```
brain-to-text-25/
├── data/                # Local datasets (do not commit due to large file sizes)
├── notebooks/           # Exploration and modeling notebooks
├── src/                 # Scripts, models, and utilities
├── docs/                # Project docs and notes
├── requirements.txt     # Python dependencies
└── README.md            # Project overview
```
### Getting Started

1. **Join the competition** on Kaggle and download the dataset.
2. **Clone this repository** and set up the environment:
    ```
    git clone https://github.com/yjoechuen/brain-to-text.git
    cd brain-to-text-25
    pip install -r requirements.txt
    ```
3. Place the competition dataset into the `data/` directory. *(Data should NOT be pushed to the repo)*

### Tasks and Progress

- [x] Competition registration & GitHub setup
- [ ] Read through [dataset paper](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132)
- [ ] Initial exploratory data analysis (EDA)
- [ ] Baseline model/notebook implementation
- [ ] Data pipeline and environment reproducibility

### Contributing Guidelines

- Work on feature branches and create pull requests for code review.
- Document code, experiments, and findings via Markdown or Jupyter notebooks.
- Use concise, meaningful commit messages.
- **Never push competition data**—keep all data in your local `data/` directory and listed in `.gitignore`.

### References

- [Brain-to-Text '25 Kaggle competition](https://www.kaggle.com/competitions/brain-to-text-25/overview)
- [An Accurate and Rapidly Calibrating Speech Neuroprosthesis](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132)
- [Baseline Algorithm from Brain-to-Text '24](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text)

---

## Exploratory Data Analysis (EDA)

This section summarizes key findings from an initial exploration of the **Brain-to-Text ’25** dataset.

### Folder Structure
```
t15_copyTask_neuralData/
└── hdf5_data_final/
    ├── t15.2023.08.11      # Exception: only contains train
        └── data_train.hdf5         
    ├── t15.2023.08.13      # Contains test, train, and val
        ├── data_test.hdf5
        ├── data_train.hdf5
        └── data_val.hdf5
    ├── t15.2023.08.18
    ...
    └── t15.2025.04.13
```
The top-level folder directory is `t15_copyTask_neuralData`, which is around 11GB and takes about 4-6 hours to download. Inside this directory is a single folder, `hdf5_data_final`, which consists of 45 sessions spanning 20 months. Each session is labeled in the form `t15.YYYY.MM.DD` (e.g., `t15.2023.08.13`). All sessions contain 3 `.hdf5` files (`data_train.hdf5, data_test.hdf5, data_val.hdf5`), with the exception of `t15.2023.08.13`, which only contains a single `data_train.hdf5`.

### File Structure

### Dataset Structure
- **Inputs:** 512 neural features per 20 ms bin  
  - 256 electrodes × 2 features each:
    - **Threshold Crossings (TC):** counts of voltage threshold events (proxy for spikes).
    - **Spike-Band Power (SBP):** continuous signal energy in the high-frequency “spike band.”
- **Outputs (training/validation only):**
  - Sentence transcription
  - Phoneme sequence (aligned to neural features)
- **Splits:** Separate train, validation, and test files in HDF5 format.

### Trial Statistics
- **Trial lengths:** Sentences vary in duration, with most trials spanning a few hundred to a few thousand 20 ms bins.
- **Sentence lengths:** Range from single words to complex multi-word sentences.
- **Phoneme counts:** Distribution shows imbalance (some phonemes appear far more often than others).

### Feature Distributions
- **Threshold Crossings (TC):**
  - Sparse, non-negative integer counts.
  - Many bins contain 0 events; higher counts are less frequent.
- **Spike-Band Power (SBP):**
  - Continuous values per bin.
  - Smoothly varying over time; generally correlated with TC counts.

### Temporal Patterns
- Heatmaps of neural features (512 × time) show structured activity during attempted speech.
- Certain electrodes exhibit stronger, more consistent responses.
- TC and SBP features are correlated but provide complementary information.

### Observations
- Data quality is generally consistent across sessions, though some electrodes show lower variance (potentially inactive).
- Clear alignment between neural activity and sentence/phoneme onset.
- Strong motivation for per-electrode normalization and possible electrode selection.

---
