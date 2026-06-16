"""
XGBoost House Price Prediction Model
Training script for predicting house prices using XGBoost regression.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


def load_data(filepath):
    """Load and validate the housing dataset."""
    df = pd.read_csv(filepath)
    print("✅ Data loaded successfully!")
    print(f"Dataset shape: {df.shape} (rows, columns)")
    print("\nFirst few rows:")
    print(df.head())
    return df


def prepare_features(df):
    """
    Prepare features and target variable.
    
    Args:
        df: Input dataframe
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    # Separate features and target
    X = df.drop(['Home', 'Price'], axis=1)  # Drop ID and target
    y = df['Price']
    
    return X, y


def encode_categorical_features(X):
    """
    Encode categorical variables.
    
    - Brick: Binary encoding (Yes/No → 1/0)
    - Neighborhood: Label encoding (East/North/West → numeric)
    
    Args:
        X: Feature matrix with categorical columns
        
    Returns:
        X: Feature matrix with encoded values
        le: LabelEncoder object for 'Neighborhood' (for future predictions)
    """
    # Binary encode 'Brick' column
    X['Brick'] = X['Brick'].map({'Yes': 1, 'No': 0})
    
    # Label encode 'Neighborhood' column
    le = LabelEncoder()
    X['Neighborhood'] = le.fit_transform(X['Neighborhood'])
    
    print("\n✅ Categorical features encoded:")
    print(f"   - Brick: Yes/No → 1/0")
    print(f"   - Neighborhood: {list(le.classes_)} → {list(range(len(le.classes_)))}")
    print("\nFeatures after encoding:")
    print(X.head())
    
    return X, le


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"\n✅ Data split into train/test:")
    print(f"   Training samples: {len(X_train)} ({100*(1-test_size):.0f}%)")
    print(f"   Testing samples: {len(X_test)} ({100*test_size:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def create_model():
    """
    Create and configure XGBoost regression model.
    
    Returns:
        model: Configured XGBRegressor instance
    """
    model = xgb.XGBRegressor(
        n_estimators=200,           # Number of boosting rounds
        learning_rate=0.1,          # Shrinkage/eta
        max_depth=5,                # Maximum tree depth
        min_child_weight=3,         # Minimum sum of instance weight in child
        subsample=0.8,              # Fraction of samples for each tree
        colsample_bytree=0.8,       # Fraction of features for each tree
        random_state=42,            # Reproducibility seed
        early_stopping_rounds=10,   # Early stopping patience
        verbose=0                   # Suppress detailed output
    )
    
    return model


def train_model(model, X_train, X_test, y_train, y_test):
    """
    Train the XGBoost model.
    
    Args:
        model: XGBRegressor instance
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        
    Returns:
        model: Trained model
    """
    print("\n🚀 Training XGBoost model...")
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print("✅ Training complete!")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, X, y):
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        model: Trained model
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        X, y: Full dataset for cross-validation
    """
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Print results
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE")
    print("="*60)
    print(f"Training R² Score:  {train_r2:.4f}")
    print(f"Testing R² Score:   {test_r2:.4f}")
    print(f"MAE (Mean Absolute Error):     ${mae:,.2f}")
    print(f"RMSE (Root Mean Squared Error): ${rmse:,.2f}")
    
    # Cross-validation
    print("\n🔄 Cross-Validation (5-fold):")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"R² Scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"Mean R²:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return y_pred_test


def show_feature_importance(model, X):
    """
    Display feature importance rankings.
    
    Args:
        model: Trained model
        X: Feature matrix
    """
    print("\n" + "="*60)
    print("📈 FEATURE IMPORTANCE")
    print("="*60)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        bar_length = int(row['Importance'] * 50)
        bar = '█' * bar_length
        print(f"{row['Feature']:15} {bar} {row['Importance']:.4f}")


def show_predictions_sample(y_test, y_pred, n_samples=10):
    """
    Display sample predictions vs actual values.
    
    Args:
        y_test: Actual test values
        y_pred: Predicted values
        n_samples: Number of samples to display
    """
    print("\n" + "="*60)
    print(f"📋 SAMPLE PREDICTIONS (First {n_samples} test samples)")
    print("="*60)
    
    comparison = pd.DataFrame({
        'Actual': y_test.values[:n_samples],
        'Predicted': y_pred[:n_samples],
        'Error': y_test.values[:n_samples] - y_pred[:n_samples],
        'Error %': ((y_test.values[:n_samples] - y_pred[:n_samples]) / y_test.values[:n_samples] * 100)
    })
    
    print(comparison.to_string(index=False))
    print(f"\nAverage Absolute Error: ${comparison['Error'].abs().mean():,.2f}")


def save_model_artifacts(model, le, model_path='xgboost_house_price_model.pkl', 
                         encoder_path='neighborhood_encoder.pkl'):
    """
    Save trained model and encoder for future use.
    
    Args:
        model: Trained XGBRegressor
        le: LabelEncoder for Neighborhood
        model_path: Path to save the model
        encoder_path: Path to save the encoder
    """
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    
    print("\n" + "="*60)
    print("💾 MODEL SAVED")
    print("="*60)
    print(f"✅ Model saved as '{model_path}'")
    print(f"✅ Encoder saved as '{encoder_path}'")
    print("\nTo use the model later:")
    print("  model = joblib.load('xgboost_house_price_model.pkl')")
    print("  le = joblib.load('neighborhood_encoder.pkl')")


def main():
    """Main execution function."""
    
    # 1. Load data
    df = load_data('h_p.csv')
    
    # 2. Prepare features and target
    X, y = prepare_features(df)
    
    # 3. Encode categorical variables
    X, le = encode_categorical_features(X)
    
    # 4. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 5. Create model
    model = create_model()
    
    # 6. Train model
    model = train_model(model, X_train, X_test, y_train, y_test)
    
    # 7. Evaluate model
    y_pred = evaluate_model(model, X_train, X_test, y_train, y_test, X, y)
    
    # 8. Show feature importance
    show_feature_importance(model, X)
    
    # 9. Show sample predictions
    show_predictions_sample(y_test, y_pred)
    
    # 10. Save artifacts
    save_model_artifacts(model, le)


if __name__ == '__main__':
    main()
