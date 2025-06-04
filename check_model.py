import pickle
import os

try:
    # Check if model file exists
    model_path = "models/best_primary_knn_model.pkl"
    if os.path.exists(model_path):
        print("Model file found")
        
        # Load and inspect the model
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        print("Data type:", type(data).__name__)
        
        if isinstance(data, dict):
            print("Dictionary keys:", list(data.keys()))
        else:
            print("Direct object type:", type(data).__name__)
    else:
        print("Model file not found")
        
except Exception as e:
    print("Error:", str(e))
