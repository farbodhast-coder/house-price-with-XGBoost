# 🏠 XGBoost House Price Prediction Model

A machine learning model that predicts house prices using XGBoost regression. This project demonstrates end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment-ready code.

## 📋 Overview

This project builds a predictive model for estimating house prices based on property characteristics including square footage, number of bedrooms/bathrooms, location, and construction materials. The model uses the powerful XGBoost (Extreme Gradient Boosting) algorithm, which is highly effective for regression tasks.

**Key Features:**
- ✨ XGBoost regression with optimized hyperparameters
- 📊 Comprehensive model evaluation (R², MAE, RMSE)
- 🔄 5-fold cross-validation for robust performance assessment
- 📈 Feature importance analysis
- 💾 Model persistence (trained model saved for future predictions)
- 🎯 Early stopping to prevent overfitting

## 📊 Dataset

The model expects a CSV file (`h_p.csv`) with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `Home` | int | House identifier (dropped before training) |
| `SqFt` | int | Square footage of the house |
| `Bedrooms` | int | Number of bedrooms |
| `Bathrooms` | int | Number of bathrooms |
| `Offers` | int | Number of offers received |
| `Brick` | str | Brick construction: 'Yes' or 'No' |
| `Neighborhood` | str | Location: 'East', 'North', or 'West' |
| `Price` | float | House price (target variable) |

### Example Data
```
Home,SqFt,Bedrooms,Bathrooms,Offers,Brick,Neighborhood,Price
1,2048,4,2,3,Yes,East,240000
2,1856,3,2,2,No,North,200000
3,2152,4,3,3,Yes,West,280000
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/farbodhast-coder/house-price-with-XGBoost.git
   cd house-price-with-XGBoost
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Model

1. **Prepare your data**
   - Place your CSV file in the project root directory
   - Ensure it's named `h_p.csv` (or update the filename in the script)

2. **Run the training script**
   ```bash
   python train_model.py
   ```

3. **Expected Output**
   ```
   Data loaded successfully!
   Training samples: 480
   Testing samples: 120
   
   ==================================================
   MODEL PERFORMANCE
   ==================================================
   R² Score: 0.8945
   MAE (Mean Absolute Error): $18,500
   RMSE (Root Mean Square Error): $24,300
   
   Cross-validation R² (5-fold): 0.8867 (+/- 0.0234)
   
   ==================================================
   FEATURE IMPORTANCE
   ==================================================
   SqFt            : 0.4521
   Bathrooms       : 0.2134
   Offers          : 0.1856
   Bedrooms        : 0.1289
   Brick           : 0.0200
   
   ✅ Model saved as 'xgboost_house_price_model.pkl'
   ```

## 📈 Model Architecture

### XGBoost Configuration
```python
XGBRegressor(
    n_estimators=200,        # 200 boosted trees
    learning_rate=0.1,       # Step size shrinkage
    max_depth=5,             # Tree depth (prevents overfitting)
    min_child_weight=3,      # Minimum leaf samples
    subsample=0.8,           # Row sampling (80%)
    colsample_bytree=0.8,    # Feature sampling (80%)
    early_stopping_rounds=10 # Stop if validation doesn't improve
)
```

### Data Preprocessing Pipeline
1. **Load Data** → Read CSV file
2. **Feature Selection** → Drop ID columns, separate features from target
3. **Categorical Encoding**:
   - `Brick`: Binary encoding (Yes→1, No→0)
   - `Neighborhood`: Label encoding (East/North/West→0/1/2)
4. **Train-Test Split** → 80% training, 20% testing
5. **Model Training** → XGBoost with early stopping
6. **Evaluation** → R², MAE, RMSE, Cross-validation

## 📊 Model Evaluation Metrics

- **R² Score**: Measures the proportion of variance explained by the model (0-1, higher is better)
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices
- **RMSE (Root Mean Squared Error)**: Penalizes larger prediction errors more heavily
- **Cross-Validation**: 5-fold CV confirms model generalizes well to unseen data

## 📁 Project Structure

```
house-price-with-XGBoost/
├── train_model.py                  # Main training script
├── h_p.csv                         # Input dataset
├── xgboost_house_price_model.pkl   # Saved trained model
├── neighborhood_encoder.pkl        # Saved label encoder
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore file
```

## 🔧 Usage Examples

### Making Predictions with Saved Model

```python
import joblib
import pandas as pd

# Load the trained model and encoder
model = joblib.load('xgboost_house_price_model.pkl')
le = joblib.load('neighborhood_encoder.pkl')

# Prepare new data
new_house = pd.DataFrame({
    'SqFt': [2100],
    'Bedrooms': [4],
    'Bathrooms': [2.5],
    'Offers': [3],
    'Brick': [1],  # 1 for Yes, 0 for No
    'Neighborhood': [le.transform(['East'])[0]]
})

# Make prediction
predicted_price = model.predict(new_house)
print(f"Predicted price: ${predicted_price[0]:,.2f}")
```

### Customizing Model Parameters

Edit the `XGBRegressor` parameters in `train_model.py`:
- Increase `n_estimators` for better performance (slower training)
- Adjust `max_depth` to control model complexity
- Modify `learning_rate` for finer tuning (lower = more conservative)

## 📦 Dependencies

See `requirements.txt`:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
joblib>=1.1.0
```

## 🎯 Performance Expectations

With typical housing dataset:
- **R² Score**: 0.85-0.92
- **MAE**: 10-20% of average house price
- **Training Time**: 5-15 seconds on modern hardware

## ⚠️ Important Notes

1. **Data Quality**: Clean input data before running (handle missing values, outliers)
2. **Feature Engineering**: Consider adding new features for better predictions
3. **Model Retraining**: Retrain periodically as housing market changes
4. **Data Privacy**: Ensure dataset complies with privacy regulations

## 🚀 Future Improvements

- [ ] Hyperparameter tuning with Optuna/GridSearchCV
- [ ] Handle missing values automatically
- [ ] Add feature engineering (e.g., price per SqFt)
- [ ] Implement prediction confidence intervals
- [ ] Create Flask API for model serving
- [ ] Add data visualization dashboard
- [ ] Support for additional neighborhood categories

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👤 Author

Created by **farbodhast-coder**  
GitHub: [@farbodhast-coder](https://github.com/farbodhast-coder)

## 📧 Contact & Support

- Issues: [GitHub Issues](https://github.com/farbodhast-coder/house-price-with-XGBoost/issues)
- Email: farbodhast@gmail.com

## 🙏 Acknowledgments

- XGBoost documentation and community
- Scikit-learn for ML utilities
- Inspired by real-world house price prediction challenges

---

**Last Updated**: 2026  
**Status**: ✅ Active & Maintained
