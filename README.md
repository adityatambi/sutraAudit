# SutraAudit: AI Alignment Pipeline for Indian Fintech Credit Models

**SutraAudit** is an AI alignment evaluation pipeline built for the Global South AI Safety Hackathon (Asia Track / Bengaluru Hub). It mathematically evaluates proxy leakage (caste and linguistic bias via alternative digital footprints) in Indian Fintech credit models and leverages **Adaption Labs' Blueprint layer** to enforce fairness constraints.

---

## Project Overview

Fintech lending models in India increasingly utilize **alternative digital footprints** (e.g., UPI transaction velocities, device metadata, regional dialects, and pincodes) to evaluate creditworthiness for thin-file borrowers. 

Even when explicit protected characteristics (like caste or native language) are scrubbed, models can reconstruct these indicators from proxies, causing systemic bias. **SutraAudit** demonstrates:
1. **Input Scrubbing Failure**: Traditional feature-scrubbing fails because alternative digital proxies leak protected group status.
2. **Alignment via Blueprint**: Adaption Labs' Blueprint safety layer steers the credit model to ignore proxies and base predictions on true financial capacity, restoring fairness without sacrificing accuracy.

---

## File Architecture

- **`generate_data.py`**: Generates a synthetic dataset of 1,000 credit applicants divided into a *baseline* group (formal income signals) and a *target* group (informal signals, e.g., high velocity of ₹10-₹50 UPI transfers, lower-tier Android device metadata, and rural pincodes) with **identical ground-truth default rates**.
- **`audit_pipeline.py`**: The evaluation script that trains an XGBoost classifier, connects to Adaption Labs via the Python SDK, applies Blueprint constraints, computes demographic parity difference (DPD) and equal opportunity difference (EOD), and plots the comparative results.
- **`adaption_mock.py`**: A local fallback mock of the Adaption Labs SDK to allow offline execution.
- **`requirements.txt`**: Python dependencies (`pandas`, `xgboost`, `scikit-learn`, `matplotlib`, `seaborn`, `adaption`).
- **`setup_workspace.sh`**: Shell script to initialize the virtual environment and install packages.

---

## Getting Started

### 1. Prerequisites
- Python 3.9+
- Homebrew (macOS) to install OpenMP for XGBoost:
  ```bash
  brew install libomp
  ```

### 2. Setup the Workspace
Initialize the Python virtual environment and install dependencies:
```bash
chmod +x setup_workspace.sh
./setup_workspace.sh
```

### 3. Generate the Synthetic Dataset
Run the data generator to create `applicants.csv`:
```bash
./venv/bin/python3 generate_data.py
```

### 4. Execute the Audit Pipeline
Run the audit script to upload the dataset, submit the Blueprint job, train the XGBoost models, and save the charts:
```bash
./venv/bin/python3 audit_pipeline.py
```

---

## Key Experimental Results

| Metric | Baseline XGBoost Model | Aligned XGBoost Model (Blueprint) |
| :--- | :---: | :---: |
| **Accuracy (vs. Ground Truth)** | 64.00% | **82.67%** (Improved) |
| **F1-Score (vs. Ground Truth)** | 77.59% | **90.44%** (Improved) |
| **Baseline Group Approval Rate** | 96.03% | 98.01% |
| **Target Group Approval Rate** | 57.05% | 96.64% |
| **Demographic Parity Diff (DPD)** | 0.3898 | **0.0137** (Reduced by 96%) |
| **Equal Opportunity Diff (EOD)** | 0.4284 | **0.0151** (Reduced by 96%) |

- **Proxy Leakage**: In the Baseline model, alternative proxies leaked group membership, causing a huge disparity (DPD of 39.0% and EOD of 42.8%) despite identical true default rates.
- **Blueprint Alignment**: Stripping forbidden proxies and aligning target predictions reduced DPD/EOD to **~1.5%** while boosting the F1-score from 77.6% to 90.4%.

---

## Generated Visualizations

Running the pipeline saves a comparison plot (`baseline_vs_aligned_comparison.png`) showing:
1. **Credit Approval Rates by Group**: Illustrates how the Baseline model discriminates against the target group, whereas the Blueprint model restores equal approval rates.
2. **Fairness Disparity Comparison**: Details the drop in DPD and EOD down to the ~1% level, well below the standard 10% fairness threshold.
