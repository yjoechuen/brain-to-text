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

## 🧠 Neural Features: Threshold Crossings and Spike-Band Power

Each trial’s neural data is represented as a sequence of **512 features** per 20 ms time bin.  
These features come from **256 microelectrodes** implanted in the speech motor cortex,  
arranged into four high-density arrays (64 electrodes each).

### ✨ What the features mean
- **Threshold Crossings (TC)**  
  - Each electrode records raw voltage signals from nearby neurons.  
  - A **threshold** is set at *–4.5 × the root mean square (RMS) noise level*.  
  - Every time the signal dips below that threshold, it is counted as a **spike-like event**.  
  - The **TC feature** for a bin is simply the **count of events** in that 20 ms window.  
  - Think of it as “how many strong blips did this electrode detect.”

- **Spike-Band Power (SBP)**  
  - Neural “spikes” also produce broadband energy in the **high-frequency band** (~250–5000 Hz).  
  - SBP measures the **energy (RMS power)** of the electrode’s signal in this band.  
  - Unlike TC (discrete counts), SBP is a **continuous value**, capturing the overall “buzz” of neural activity.  
  - Think of it as “how strong was the chatter in this frequency band.”

Together, TC and SBP give a fuller picture of neural activity:  
TC counts sharp, obvious spikes, while SBP measures the more subtle, continuous background activity. 

### 🎯 Why Collect Threshold Crossings (TC) and Spike-Band Power (SBP)?
- **Threshold Crossings (TC)**  
  - Capture *precise, discrete firing events* (like click counters).  
  - Great for detecting sharp transitions in speech movements.  
  - Especially useful for **consonants** that are short and crisp, like /t/, /p/, /k/.  
- **Spike-Band Power (SBP)**  
  - Measures *smooth, continuous population activity*.  
  - Tracks the ongoing “buzz” of neural engagement over time.  
  - Especially useful for **vowels**, which require sustained articulator positions (like /a/, /i/, /u/).  
- **Together** they provide a **richer, more robust signal**:  
  - TC = **when** neurons fired (timing of discrete events)  
  - SBP = **how strongly** populations were active (intensity of sustained activity)  

### 🗣️ How this helps phoneme prediction
Producing speech involves coordinating different articulators:
- /p/, /b/ → lips  
- /t/, /d/ → tongue tip  
- /k/, /g/ → tongue back  

The four cortical areas covered by the arrays (ventral 6v, area 4, 55b, dorsal 6v) capture different aspects of this motor control.  
By combining both TC and SBP features:
- The model can **pinpoint consonant onsets/offsets** via TC activity.  
- The model can **track vowel sustain and transitions** via SBP energy.  
- This improves the decoder’s ability to align neural activity to **phoneme sequences** and ultimately **sentence predictions**.

---

📊 **Summary:**  
- TC = “click counter” → good for **sharp, fast events** like consonants.  
- SBP = “background buzz” → good for **smooth, sustained activity** like vowels.  
- Using both ensures the model sees both **timing** and **intensity**, making speech decoding more accurate and robust.

---
## Exploratory Data Analysis (EDA)

This section summarizes key findings from an initial exploration of the **Brain-to-Text ’25** dataset.

### Folder Structure
```
data/
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
The top-level folder directory is `data`. Inside this directory is a single folder, `hdf5_data_final`, which consists of 45 sessions spanning 20 months. Each session is labeled in the form `t15.YYYY.MM.DD` (e.g., `t15.2023.08.13`). All sessions contain 3 `.hdf5` files (`data_train.hdf5, data_test.hdf5, data_val.hdf5`), with the exception of `t15.2023.08.11`, which only contains a single `data_train.hdf5`.

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
