import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pickle
import joblib
import os

class SimilarMealPlanService:
    def __init__(self):
        # Load data and model using cleaned dataset
        self.df = pd.read_csv("dataset/daily_food_nutrition_dataset_cleaned.csv")
        self.features = ["calories", "proteins", "carbohydrates", "fats", "fibers", "sugars", "sodium", "cholesterol"]
        X = self.df[self.features]        # Load pre-trained KNN model and scaler
        model_path = "models/best_primary_knn_model.pkl"
        joblib_path = "models/best_primary_knn_model.joblib"
        
        try:
            # Try loading with joblib first (recommended for scikit-learn)
            if os.path.exists(joblib_path):
                print(f"Loading pre-trained KNN model from {joblib_path}")
                model_data = joblib.load(joblib_path)
                
                # Extract model and scaler from saved data
                if isinstance(model_data, dict):
                    self.knn = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    if self.scaler is not None:
                        self.X_scaled = self.scaler.transform(X)
                    else:
                        # Fallback: create new scaler if not saved
                        self.scaler = StandardScaler()
                        self.X_scaled = self.scaler.fit_transform(X)
                else:
                    # If model_data is just the model object
                    self.knn = model_data
                    self.scaler = StandardScaler()
                    self.X_scaled = self.scaler.fit_transform(X)
                
                print("✅ Pre-trained model loaded successfully with joblib")
                
            elif os.path.exists(model_path):
                print(f"Loading pre-trained KNN model from {model_path}")
                
                # Try different pickle protocols
                model_data = None
                for protocol in [None, 0, 1, 2, 3, 4, 5]:
                    try:
                        with open(model_path, 'rb') as f:
                            if protocol is None:
                                model_data = pickle.load(f)
                            else:
                                model_data = pickle.load(f)
                        break
                    except Exception as protocol_error:
                        if protocol is None:
                            print(f"Default protocol failed: {protocol_error}")
                        continue
                
                if model_data is None:
                    raise Exception("Failed to load with any pickle protocol")
                
                # Extract model and scaler from saved data
                if isinstance(model_data, dict):
                    self.knn = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    if self.scaler is not None:
                        self.X_scaled = self.scaler.transform(X)
                    else:
                        # Fallback: create new scaler if not saved
                        self.scaler = StandardScaler()
                        self.X_scaled = self.scaler.fit_transform(X)
                else:
                    # If model_data is just the model object
                    self.knn = model_data
                    self.scaler = StandardScaler()
                    self.X_scaled = self.scaler.fit_transform(X)
                
                print("✅ Pre-trained model loaded successfully with pickle")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path} or {joblib_path}")
                
        except Exception as e:
            print(f"⚠️ Failed to load pre-trained model: {e}")
            print("🔄 Falling back to training new model...")
            
            # Fallback: train new model
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(X)
            self.knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
            self.knn.fit(self.X_scaled)
            
            # Save the new model with joblib for better compatibility
            try:
                os.makedirs("models", exist_ok=True)
                model_data = {
                    'model': self.knn,
                    'scaler': self.scaler,
                    'features': self.features,
                    'dataset_shape': self.df.shape
                }
                joblib.dump(model_data, joblib_path)
                print(f"✅ New model saved with joblib to {joblib_path}")
            except Exception as save_error:
                print(f"⚠️ Failed to save new model: {save_error}")

        # Load start foods from first meal plan
        try:
            firstMealDf = pd.read_csv("results/first_meal_plan.csv")
            self.start_foods = firstMealDf["food_item"].dropna().tolist()
        except FileNotFoundError:
            self.start_foods = []

    def generate_meal_plan(self, start_food, days=30):
        meal_plan = [start_food]
        current_food = start_food
        visited = set()
        visited.add(current_food.lower())

        for _ in range(days - 1):
            idx = self.df[self.df["food_item"].str.lower() == current_food.lower()].index
            if len(idx) == 0:
                meal_plan.append("Not Found")
                break
            
            idx = idx[0]
            distances, indices = self.knn.kneighbors([self.X_scaled[idx]])
            
            found = False
            for neighbor_idx in indices[0][1:]:
                neighbor_food = self.df.iloc[neighbor_idx]['food_item']
                if neighbor_food.lower() not in visited:
                    meal_plan.append(neighbor_food)
                    current_food = neighbor_food
                    visited.add(neighbor_food.lower())
                    found = True
                    break
            
            if not found:
                meal_plan.append(current_food)
        
        while len(meal_plan) < days:
            meal_plan.append(meal_plan[-1])
        
        return meal_plan
    
    def generate_similar_meal_plans(self, days=30, start_foods=[]):
        meal_plans = []
        
        # Convert food IDs to food names if necessary
        processed_start_foods = []
        for food in start_foods:
            # Check if food is an ID (24-character hexadecimal string) or exists in ID column
            if len(food) == 24 and all(c in '0123456789abcdef' for c in food.lower()):
                # It's likely a food ID, convert to food name
                food_row = self.df[self.df['id'] == food]
                if not food_row.empty:
                    food_name = food_row.iloc[0]['food_item']
                    processed_start_foods.append(food_name)
                else:
                    # ID not found, keep original value
                    processed_start_foods.append(food)
            else:
                # Check if it exists in the ID column anyway
                food_row = self.df[self.df['id'] == food]
                if not food_row.empty:
                    food_name = food_row.iloc[0]['food_item']
                    processed_start_foods.append(food_name)
                else:
                    # Assume it's already a food name
                    processed_start_foods.append(food)
        
        # Generate dates starting from today
        start_date = datetime.now()
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
              # Create meals for this day using all start foods
            meals = []
            for food in processed_start_foods:
                # Get the food for this specific day from the meal plan
                food_plan = self.generate_meal_plan(food, days)
                if day < len(food_plan):
                    current_food = food_plan[day]
                    
                    # Find the food ID for the current food
                    food_row = self.df[self.df['food_item'] == current_food]
                    food_id = food_row.iloc[0]['id'] if not food_row.empty else ""
                    
                    meals.append({
                        "foodId": food_id,
                        "status": "not_completed"
                    })
            
            # Create daily plan
            daily_plan = {
                "date": current_date.isoformat(),
                "meals": meals
            }
            
            meal_plans.append(daily_plan)
        
        return meal_plans

# Create singleton instance
similar_meal_service = SimilarMealPlanService()
