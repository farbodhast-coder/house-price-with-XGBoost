"""
Example: Making predictions with the trained XGBoost house price model

This script demonstrates how to use the saved model and encoder
to make predictions on new house data.
"""

import joblib
import pandas as pd
import numpy as np


def main():
    """Load the trained model and make predictions on new data."""
    
    # Load the trained model and label encoder
    print("Loading trained model...")
    model = joblib.load('xgboost_house_price_model.pkl')
    le = joblib.load('neighborhood_encoder.pkl')
    print("✅ Model loaded successfully!\n")
    
    # Example 1: Single house prediction
    print("="*60)
    print("EXAMPLE 1: Predicting price for a single house")
    print("="*60)
    
    # Create a sample house
    house_1 = pd.DataFrame({
        'SqFt': [2100],
        'Bedrooms': [4],
        'Bathrooms': [2.5],
        'Offers': [3],
        'Brick': [1],  # 1 for Yes, 0 for No
        'Neighborhood': [le.transform(['East'])[0]]
    })
    
    print("\nHouse Details:")
    print(f"  Square Footage: {house_1['SqFt'].values[0]} sqft")
    print(f"  Bedrooms: {house_1['Bedrooms'].values[0]}")
    print(f"  Bathrooms: {house_1['Bathrooms'].values[0]}")
    print(f"  Offers: {house_1['Offers'].values[0]}")
    print(f"  Brick Construction: {'Yes' if house_1['Brick'].values[0] == 1 else 'No'}")
    print(f"  Neighborhood: East")
    
    price_1 = model.predict(house_1)[0]
    print(f"\n🏠 Predicted Price: ${price_1:,.2f}")
    
    # Example 2: Multiple houses
    print("\n" + "="*60)
    print("EXAMPLE 2: Predicting prices for multiple houses")
    print("="*60)
    
    # Create multiple sample houses
    houses_data = {
        'SqFt': [1800, 2200, 2500],
        'Bedrooms': [3, 4, 5],
        'Bathrooms': [2, 2.5, 3],
        'Offers': [2, 3, 4],
        'Brick': [0, 1, 1],  # No, Yes, Yes
        'Neighborhood': [
            le.transform(['North'])[0],
            le.transform(['East'])[0],
            le.transform(['West'])[0]
        ]
    }
    
    houses_df = pd.DataFrame(houses_data)
    
    print("\nPredicting prices for 3 houses...\n")
    
    predictions = model.predict(houses_df)
    
    # Display results
    results = pd.DataFrame({
        'SqFt': houses_data['SqFt'],
        'Bedrooms': houses_data['Bedrooms'],
        'Bathrooms': houses_data['Bathrooms'],
        'Offers': houses_data['Offers'],
        'Predicted_Price': predictions,
        'Price_Per_SqFt': predictions / houses_data['SqFt']
    })
    
    print(results.to_string(index=False))
    
    # Example 3: Interactive prediction
    print("\n" + "="*60)
    print("EXAMPLE 3: How to make a custom prediction")
    print("="*60)
    
    print("\nTo predict a price for your own house, follow this template:\n")
    
    template = """
import joblib
import pandas as pd

# Load the trained model and encoder
model = joblib.load('xgboost_house_price_model.pkl')
le = joblib.load('neighborhood_encoder.pkl')

# Create your house data
my_house = pd.DataFrame({
    'SqFt': [2200],                           # Square footage
    'Bedrooms': [4],                          # Number of bedrooms
    'Bathrooms': [2.5],                       # Number of bathrooms
    'Offers': [3],                            # Number of offers
    'Brick': [1],                             # 1 for Yes, 0 for No
    'Neighborhood': [le.transform(['East'])[0]]  # East, North, or West
})

# Make prediction
predicted_price = model.predict(my_house)[0]
print(f"Predicted Price: ${predicted_price:,.2f}")
    """
    
    print(template)
    
    # Example 4: Feature encoding reference
    print("\n" + "="*60)
    print("REFERENCE: Feature Encoding")
    print("="*60)
    
    print("\n✓ Brick Column:")
    print("  - 1 = Yes (has brick construction)")
    print("  - 0 = No (no brick construction)")
    
    print("\n✓ Neighborhood Column (Label Encoding):")
    for i, neighborhood in enumerate(le.classes_):
        print(f"  - {i} = {neighborhood}")
    
    print("\n✓ All other features (SqFt, Bedrooms, Bathrooms, Offers):")
    print("  - Use numeric values directly")


if __name__ == '__main__':
    main()
