# Brain-to-Text '25 Kaggle Competition

## Project Overview

This repository contains code and documentation for our team's solution to the [Brain-to-Text '25 Kaggle competition](https://www.kaggle.com/competitions/brain-to-text-25/overview). The challenge is to **decode intracortical neural activity during attempted speech into corresponding text**, pushing forward the field of neural signal decoding and brain-computer interfaces.

- **Team Name:** Neural Navigators
- **Team Members:** 5
- **Competition Dates:** September–December 2025

## Team & Communication

- **Team Members:** George Yu, Carl Shashuk, Rohini G, Nils Matteson, Daniel Yang
- **Communication:** [Slack Channel](https://data-science-hubgroup.slack.com/archives/C09EUS28ZEJ)

## Repository Structure
brain-to-text-25/
├── data/                # Local datasets (do not commit due to large file sizes)
├── notebooks/           # Exploration and modeling notebooks
├── src/                 # Scripts, models, and utilities
├── docs/                # Project docs and notes
├── requirements.txt     # Python dependencies
└── README.md            # Project overview

## Getting Started

1. **Join the competition** on Kaggle and download the dataset.
2. **Clone this repository** and set up the environment:
    ```
    git clone [repo-url]
    cd brain-to-text-25
    pip install -r requirements.txt
    ```
3. Place the competition dataset into the `data/` directory. *(Data should NOT be pushed to the repo)*

## Tasks and Progress

- [x] Competition registration & GitHub setup
- [ ] Read through [dataset paper](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132)
- [ ] Initial exploratory data analysis (EDA)
- [ ] Baseline model/notebook implementation
- [ ] Data pipeline and environment reproducibility

## Contributing Guidelines

- Work on feature branches and create pull requests for code review.
- Document code, experiments, and findings via Markdown or Jupyter notebooks.
- Use concise, meaningful commit messages.
- **Never push competition data**—keep all data in your local `data/` directory and listed in `.gitignore`.

## References

- [Brain-to-Text '25 Kaggle competition](https://www.kaggle.com/competitions/brain-to-text-25/overview)
- [An Accurate and Rapidly Calibrating Speech Neuroprosthesis](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132)
- [Baseline Algorithm from Brain-to-Text '24](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text)

---

