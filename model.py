import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Load your CSV file
df = pd.read_csv('dataset_HousePrice.csv')

# Display first few rows to verify
print("Data loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)

# Prepare features and target
# Assuming your columns are: Home, SqFt, Bedrooms, Bathrooms, Offers, Brick, Neighborhood, Price
X = df.drop(['Home', 'Price'], axis=1)  # Drop Home ID and target
y = df['Price']

# Encode categorical variables
# Convert 'Brick' (Yes/No) to 1/0
X['Brick'] = X['Brick'].map({'Yes': 1, 'No': 0})

# Encode 'Neighborhood' (East/North/West)
le = LabelEncoder()
X['Neighborhood'] = le.fit_transform(X['Neighborhood'])

print("\nFeatures after encoding:")
print(X.head())

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create XGBoost model
model = xgb.XGBRegressor(
    n_estimators=200,           # Number of trees
    learning_rate=0.1,          # Step size
    max_depth=5,                # Maximum tree depth
    min_child_weight=3,         # Minimum child weight
    subsample=0.8,              # Row sampling
    colsample_bytree=0.8,       # Feature sampling
    random_state=42,
    early_stopping_rounds=10    # Stop if no improvement
)

# Train the model
print("\nTraining XGBoost model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE (Mean Absolute Error): ${mean_absolute_error(y_test, y_pred):,.0f}")
print(f"RMSE (Root Mean Square Error): ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"\nCross-validation R² (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature importance
print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in feature_importance.iterrows():
    print(f"{row['Feature']:15} : {row['Importance']:.4f}")

# Sample prediction vs actual comparison
print("\n" + "="*50)
print("SAMPLE PREDICTIONS (First 10 test samples)")
print("="*50)
comparison = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Difference': y_test.values[:10] - y_pred[:10]
})
print(comparison)
print(f"\nAverage difference: ${comparison['Difference'].abs().mean():,.0f}")

# Optional: Save the model for future use
import joblib
joblib.dump(model, 'xgboost_house_price_model.pkl')
joblib.dump(le, 'neighborhood_encoder.pkl')
print("\n✅ Model saved as 'xgboost_house_price_model.pkl'")
