import pandas as pd
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def recreate_model():
    """Recreate and save the KNN model properly"""
    
    print("=== Recreating KNN Model ===")
    
    # Load the dataset
    try:
        df = pd.read_csv("dataset/daily_food_nutrition_dataset_cleaned.csv")
        print(f"Dataset loaded: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Prepare features
    features = ["calories", "proteins", "carbohydrates", "fats", "fibers", "sugars", "sodium", "cholesterol"]
    X = df[features]
    
    print(f"Features shape: {X.shape}")
    print(f"Features: {features}")
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Scaler fitted")
    
    # Create and fit KNN model
    knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
    knn.fit(X_scaled)
    print("✅ KNN model fitted")
    
    # Create model data dictionary
    model_data = {
        'model': knn,
        'scaler': scaler,
        'features': features,
        'dataset_shape': df.shape
    }
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the model
    model_path = "models/best_primary_knn_model.pkl"
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Model saved successfully to {model_path}")
        
        # Verify the saved model
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)
        print("✅ Model verification successful")
        print(f"Loaded model keys: {list(loaded_data.keys())}")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    print("=== Model Recreation Complete ===")

if __name__ == "__main__":
    recreate_model()
