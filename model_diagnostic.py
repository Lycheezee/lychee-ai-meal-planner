import pandas as pd
import pickle
import os

# Simple test to check model and data loading
print("=== Model Loading Diagnostic ===")

# Check if files exist
model_path = "models/best_primary_knn_model.pkl"
dataset_path = "dataset/daily_food_nutrition_dataset_cleaned.csv"

print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Dataset file exists: {os.path.exists(dataset_path)}")

if os.path.exists(model_path):
    print(f"Model file size: {os.path.getsize(model_path)} bytes")

# Test dataset loading
try:
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully - Shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")
    print("Sample data:")
    print(df[['id', 'food_item', 'calories']].head(3))
except Exception as e:
    print(f"Error loading dataset: {e}")

# Test model loading
try:
    print("\nTesting model loading...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    print(f"Model loaded successfully")
    print(f"Model type: {type(model_data)}")
    
    if isinstance(model_data, dict):
        print(f"Model keys: {list(model_data.keys())}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("This explains why the service falls back to training a new model")

print("\n=== End Diagnostic ===")
