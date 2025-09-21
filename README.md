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
- [x] Read through [dataset paper](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132)
- [~] Initial exploratory data analysis (EDA)
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

### 📂 Folder Structure
```
data/
└── hdf5_data_final/
    ├── t15.2023.08.11      # Exception: only contains train
    │   └── data_train.hdf5         
    ├── t15.2023.08.13      # Contains test, train, and val
    │   ├── data_test.hdf5
    │   ├── data_train.hdf5
    │   └── data_val.hdf5
    ├── t15.2023.08.18
    ...
    └── t15.2025.04.13
```
The top-level folder directory is `data`. Inside this directory is a single folder, `hdf5_data_final`, which consists of 45 sessions spanning 20 months. Each session is labeled in the form `t15.YYYY.MM.DD` (e.g., `t15.2023.08.13`). All sessions contain 3 `.hdf5` files (`data_train.hdf5, data_test.hdf5, data_val.hdf5`), with the exception of `t15.2023.08.11`, which only contains a single `data_train.hdf5`.

### 📄 File Structure
```
data_train.hdf5
├── trial_0001/
│   ├── input_features      (Dataset, shape [T, 512])
│   ├── seq_class_ids       (Dataset, shape [seq_len])
│   ├── transcription       (Dataset, string/bytes)
│   └── (attrs)
│       ├── n_time_steps    (int)
│       ├── seq_len         (int)
│       ├── sentence_label  (int)
│       ├── session         (string)
│       ├── block_num       (int)
│       └── trial_num       (int)
│
├── trial_0002/
│   └── ...
└── ...
```
Let's take a look at a single `data_train.hdf5` file. The `data_train.hdf5` file contains a list of trials, where each trial corresponds to a single sentence attempt (one recording). Each trial contains "Datasets" (stored as arrays) and "Attributes" `attrs` which contain small metadata key-value pairs attached to the trial.

### 🔑 Explanation of Each Field in a Trial

- **`input_features`** *(Dataset)*  
  - Shape: `[T, 512]` where `T` = number of 20 ms bins in this trial.  
  - Columns 0–255 = **Threshold Crossings (TC)** counts per electrode.  
  - Columns 256–511 = **Spike-Band Power (SBP)** values per electrode.  
  - Each row = one 20 ms snapshot of neural activity.

- **`seq_class_ids`** *(Dataset, optional)*  
  - Integer IDs representing the **phoneme sequence** aligned to this trial.  
  - Present in **train/val** splits, omitted in test splits.

- **`transcription`** *(Dataset, optional)*  
  - The ground-truth **sentence text** for this trial (bytes; decode to string).  
  - Present in **train/val** splits, omitted in test splits.

- **`n_time_steps`** *(Attribute)*  
  - Number of time bins (`T`) for this trial.  
  - Matches the first dimension of `input_features`.

- **`seq_len`** *(Attribute)*  
  - Length of the phoneme sequence (`seq_class_ids`).  

- **`sentence_label`** *(Attribute)*  
  - Stores a numeric or string ID that points to which sentence in the stimulus set this trial came from.
  - Useful for grouping multiple trials of the same sentence.

- **`session`** *(Attribute)*  
  - Code for the recording session (e.g., `t15.2021.04.12`).  

- **`block_num`** *(Attribute)*  
  - Block index within the session (sessions are often broken into multiple blocks).
  - A continuous chunk of recording with a set of sentences.  

- **`trial_num`** *(Attribute)*  
  - Trial index within the block.  

### 📄 Dataset Inputs and Outputs

Each `.hdf5` file contains many *trials* (one sentence attempt each).  
The contents differ slightly between **train/val** and **test** splits:

| Split       | Inputs (given)                                                                 | Outputs (provided)                                                                                           | Metadata (provided)                            |
|-------------|---------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| **Train**   | `input_features` `(T, 512)`<br>• 256 electrodes × 2 features (Threshold Crossings + Spike-Band Power)<br>• Each row = 20 ms bin | `seq_class_ids` (phoneme IDs)<br>`seq_len` (length of phoneme sequence)<br>`transcription` (sentence text)<br>`sentence_label` (sentence ID) | `n_time_steps`<br>`session`<br>`block_num`<br>`trial_num` |
| **Val**     | Same as Train                                                                  | Same as Train                                                                                                | Same as Train                                  |
| **Test**    | `input_features` `(T, 512)`                                                    | ❌ No phonemes<br>❌ No transcription<br>❌ No sentence_label                                                  | `n_time_steps`<br>`session`<br>`block_num`<br>`trial_num` |

### 🎯 Prediction target
- For each **test trial**, you are given only the `input_features`.  
- Your model must predict the **sentence transcription** (text).  
- Submissions are evaluated using **Character Error Rate (CER)** between your predictions and the hidden ground truth.

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
