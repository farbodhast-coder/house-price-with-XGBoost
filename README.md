# 🏠 House Price Prediction with XGBoost

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5.0-green.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0-orange.svg)](https://scikit-learn.org/)

A robust machine learning pipeline for predicting house prices using **XGBoost**, featuring automated preprocessing, cross‑validation, and detailed performance metrics.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Feature Importance](#feature-importance)
- [Saving the Model](#saving-the-model)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 📌 Overview

This project implements an end‑to‑end regression solution for house price prediction. It loads a CSV dataset, preprocesses categorical variables, trains an XGBoost regressor with early stopping, and evaluates the model using multiple metrics (R², MAE, RMSE). The pipeline also performs 5‑fold cross‑validation and saves the trained model for future inference.

The code is written in Python using popular libraries: `pandas`, `numpy`, `scikit-learn`, and `xgboost`.

---

## 📊 Dataset

The script expects a CSV file named **`dataset.HousePrice.csv`** with the following columns:

| Column        | Type        | Description                              |
|---------------|-------------|------------------------------------------|
| `Home`        | Integer/ID  | Unique identifier for each home (dropped)|
| `SqFt`        | Integer     | Square footage of the home               |
| `Bedrooms`    | Integer     | Number of bedrooms                       |
| `Bathrooms`   | Integer     | Number of bathrooms                      |
| `Offers`      | Integer     | Number of offers received                |
| `Brick`       | String      | `Yes` / `No` (converted to 1/0)          |
| `Neighborhood`| String      | `East`, `North`, or `West` (label‑encoded)|
| `Price`       | Integer/float| Target variable – sale price             |

> **Note:** The script drops the `Home` column and treats `Price` as the target. All other columns are used as features.

---

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Clone the repository
```bash
git clone https://github.com/farbodhast-coder/house-price-with-XGBoost.git
cd house-price-with-XGBoost
