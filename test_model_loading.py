import pickle
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Test model loading
model_path = "models/best_primary_knn_model.pkl"

print(f"Testing model loading from: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
print(f"File size: {os.path.getsize(model_path)} bytes")

try:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Successfully loaded model data")
    print(f"Type of model_data: {type(model_data)}")
    
    if isinstance(model_data, dict):
        print(f"Model data keys: {list(model_data.keys())}")
        for key, value in model_data.items():
            print(f"  {key}: {type(value)}")
    else:
        print(f"Model data is not a dictionary: {model_data}")
        
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()

# Test dataset loading
try:
    df = pd.read_csv("dataset/daily_food_nutrition_dataset_cleaned.csv")
    print(f"\nDataset loaded successfully")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())
except Exception as e:
    print(f"Error loading dataset: {e}")
