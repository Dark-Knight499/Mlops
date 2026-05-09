# 🚀 Reproducible MLOps Pipeline for Telecom Churn

A production-style, research-inspired MLOps implementation for **customer churn prediction** using **Python + uv + DVC**.

## 🧠 Problem Statement
Telecom providers lose significant revenue from customer churn. Traditional ML notebooks fail in production due to:
- low reproducibility,
- weak experiment traceability,
- manual and error-prone retraining.

This project demonstrates a reproducible end-to-end pipeline that can be demoed in under 20 minutes.

## 🏗️ Architecture

```text
                 ┌──────────────────────┐
                 │      params.yaml     │
                 └──────────┬───────────┘
                            │
                            ▼
┌──────────────┐   ┌────────────────┐   ┌──────────────────┐
│ data_ingest  ├──►│ data_validation├──►│ data_preprocess  │
└──────┬───────┘   └────────┬───────┘   └─────────┬────────┘
       │                    │                     │
       ▼                    ▼                     ▼
 data/raw/churn.csv   artifacts/validation   data/processed
                                               train/test
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │feature_engineering│
                                            └─────────┬────────┘
                                                      ▼
                                              data/features
                                                      │
                                                      ▼
                                            ┌────────────────┐
                                            │ model_training │
                                            └────────┬───────┘
                                                     ▼
                                             models/churn_model.pkl
                                                     │
                                                     ▼
                                            ┌────────────────┐
                                            │model_evaluation│
                                            └────────┬───────┘
                                                     ▼
                          artifacts/metrics.json + feature importance + confusion matrix
```

## 🔄 Pipeline Flow
1. **data_ingestion**: synthesizes realistic telecom churn dataset.
2. **data_validation**: schema/quality checks and validation report.
3. **data_preprocessing**: deduplicate + stratified train/test split.
4. **feature_engineering**: churn-focused behavioral features.
5. **model_training**: RandomForest training and model serialization.
6. **model_evaluation**: metrics + visual artifacts.

## 📁 Project Structure

```text
.
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── config/
│   ├── utils/
│   └── main.py
├── data/
├── artifacts/
├── models/
├── dvc.yaml
├── params.yaml
├── pyproject.toml
├── README.md
├── index.html
└── assets/
```

## ⚙️ Setup Guide (uv + DVC)

```bash
# 1) initialize Python project
uv init --name mlops-churn --python 3.11

# 2) add dependencies
uv add pandas scikit-learn dvc matplotlib

# 3) initialize DVC
dvc init

# 4) run full pipeline
dvc repro
```

## 🎯 Demo Walkthrough

```bash
# run all stages via DVC
uv run dvc repro

# inspect metrics
cat artifacts/metrics.json

# open presentation
open index.html  # macOS
# xdg-open index.html  # Linux
```

## 📊 Results
- `artifacts/metrics.json` with core classification metrics
- `artifacts/plots/feature_importance.png`
- `artifacts/plots/confusion_matrix.png`
- `models/churn_model.pkl`

## 💡 Key Insights
- **Payment delays + support burden** are strong churn signals.
- Stratified splitting keeps class balance realistic and stable.
- DVC + params-driven stages make re-runs deterministic and auditable.

## ✨ Wow Moment
The feature-importance chart clearly highlights retention leverage points (billing friction + support friction), turning the model from “just predictive” into **actionable product strategy**.

## 🔮 Future Improvements
- Add experiment tracking (MLflow).
- Add model registry + deployment endpoint.
- Add data drift monitoring with scheduled retraining.

---

## 🎬 20-Minute Presentation Script

### 0:00–2:00 — Opening
**Say:** “I built a reproducible churn intelligence pipeline for telecom that behaves like a real production asset, not a notebook.”
**Show:** `README.md` overview + architecture diagram.

### 2:00–5:00 — Business Context
**Say:** “Churn is a revenue leak. Without MLOps, model delivery becomes slow, brittle, and hard to trust.”
**Show:** `index.html` Problem + Why MLOps sections.

### 5:00–8:00 — Stack Choices
**Say:** “I used Python for modeling, uv for modern dependency management, and DVC for data/pipeline lineage.”
**Show/Run:**
```bash
uv --version
dvc --version
```

### 8:00–12:00 — Live Pipeline
**Say:** “Now I’ll reproduce the complete workflow from raw data to evaluated model in one command.”
**Run:**
```bash
dvc repro
```
**Punchline:** “Every stage is modular, parameterized, and versionable.”

### 12:00–15:00 — Outputs & Insight
**Say:** “This is where engineering meets product decision-making.”
**Show:**
- `artifacts/metrics.json`
- `artifacts/plots/feature_importance.png`
- `artifacts/plots/confusion_matrix.png`

**Insight line:** “Billing delays and repeated support interactions dominate churn risk; these are operational levers, not just model features.”

### 15:00–18:00 — Architecture & Reliability
**Say:** “DVC stages and params make this reproducible across machines and teammates.”
**Show:** `dvc.yaml`, `params.yaml`, stage modularity under `src/components`.

### 18:00–20:00 — Close
**Say:** “This project demonstrates not only model performance, but also production readiness and stakeholder clarity via the presentation layer.”
**Show:** Hero + Conclusion in `index.html` and end on CTA.
