# Car Price Prediction

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https://github.com/babaksoft/price-prediction/raw/refs/heads/master/pyproject.toml)
![Static Badge](https://img.shields.io/badge/task-regression-orange)
![Static Badge](https://img.shields.io/badge/framework-sklearn-orange)
![GitHub License](https://img.shields.io/github/license/babaksoft/price-prediction)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/babaksoft/price-prediction/build.yml)


## Project Overview

This project predicts **used car prices** based on technical, categorical, and
condition-related attributes.

The dataset originates from the
[Car Price Prediction Challenge on Kaggle](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge),
containing over 19,000 entries with mixed data types.

The main goal was to **design a robust and reproducible regression pipeline** that handles
extensive data cleaning, feature engineering, and hyperparameter tuning using modern MLOps
tools (DVC, MLflow).

Every stage — from ingestion to evaluation — is version-controlled for complete experiment
traceability.

The target variable (Price) is **log-transformed** to reduce skewness, stabilize variance,
and improve model learning.

## Business Framing

In a real-world dealership or car marketplace, an accurate car price prediction model
helps with:

- **Automatic pricing suggestions** for used car listings
- **Market consistency checks** — detecting overpriced or underpriced listings
- **Consumer transparency** — showing fair-value valuation ranges

From a business standpoint, the model acts as a **pricing decision-support tool**, not a
replacement for expert appraisals.

Each percentage point of error reduction directly translates into **fewer mispriced vehicles**,
higher trust in pricing credibility, and optimized profit margins for platforms relying on
algorithmic listing recommendations.

## Performance Requirements (Realistic Goals)

Since car prices can vary widely by make, model, and condition, perfect predictions
are unrealistic. Reasonable performance targets are described in both **statistical**
and **business-relevant** terms.

| Metric                | Acceptable | Good      | Excellent   |
|-----------------------|------------|-----------|-------------|
| MAE (USD, real scale) | ≤ 3,000    | ≤ 2,000   | ≤ 1,000     |
| MAPE (percentage)     | ≤ 25%      | ≤ 20%     | ≤ 15%       |
| R²                    | ≥ 0.7      | ≥ 0.8     | ≥ 0.9       |

For business fairness, **errors under 15–20%** on used car prices are considered
reliable — especially when prices span tens of thousands of dollars.

Note: Models are evaluated on **inverse-transformed predictions** to report realistic
figures.

## Dataset summary

(TODO: Provide a quick introduction of dataset features and target in tabular Markdown format)

