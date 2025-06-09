import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pickle
import joblib
import os
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarMealPlanService:
    def __init__(self):
        # Update dataset path to use the new dataset
        self.df = pd.read_csv("dataset/product_dataset/final_usable_food_dataset.csv")
        self.features = ["calories", "proteins", "carbohydrates", "fats", "fibers", "sugars", "sodium", "cholesterol"]
        
        # Data cleaning
        self.df[self.features] = self.df[self.features].fillna(0)
        
        try:
            # Try to load pre-trained model if it exists
            self.scaler = joblib.load("models/food_feature_scaler.pkl")
            self.knn = joblib.load("models/knn_food_similarity_model.pkl")
            self.X_scaled = self.scaler.transform(self.df[self.features])
            logger.info("✅ Successfully loaded pre-trained models")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load pre-trained model: {e}")
            logger.info("🔄 Falling back to training new model...")
            
            # Fallback: train new model
            X = self.df[self.features]
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(X)
            self.knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
            self.knn.fit(self.X_scaled)
        
        # Initialize RL service
        self.rl_service = None
        self._init_rl_service()
        
        self.start_foods = []
    
    def _init_rl_service(self):
        """Initialize the RL service for enhanced meal planning."""
        try:
            import sys
            sys.path.append("rl")
            from rl_meal_service import RLMealPlanService
            
            self.rl_service = RLMealPlanService(
                data_path="dataset/product_dataset/final_usable_food_dataset.csv",
                model_path="models/rl_meal_planner"
            )
            logger.info("✅ RL service initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize RL service: {e}")
            logger.info("🔄 Will use traditional KNN-based approach")
            

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
      def generate_similar_meal_plans(self, days=30, start_foods=[], use_rl=True):
        """Generate meal plans using RL if available, otherwise fallback to KNN."""
        
        # If RL service is available and requested, use it
        if use_rl and self.rl_service is not None:
            try:
                logger.info("Using RL-based meal planning")
                return self.rl_service.generate_similar_meal_plans(
                    days=days,
                    start_foods=start_foods,
                    use_rl=True
                )
            except Exception as e:
                logger.warning(f"RL meal planning failed: {e}, falling back to KNN")
        
        # Fallback to traditional KNN-based approach
        logger.info("Using traditional KNN-based meal planning")
        meal_plans = []
        
        # Convert food IDs to food names if necessary
        processed_start_foods = []
        for food in start_foods:
            # Check if it's already a food name in our dataset
            if self.df['food_item'].str.lower().eq(food.lower()).any():
                processed_start_foods.append(food)
            else:
                # Try to find by partial match
                matches = self.df[self.df['food_item'].str.contains(food, case=False, na=False)]
                if not matches.empty:
                    processed_start_foods.append(matches.iloc[0]['food_item'])
                else:
                    processed_start_foods.append(food)  # Keep original if no match
        
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
                    
                    # Create a simple hash-based ID since dataset doesn't have IDs
                    food_id = str(hash(current_food) % 10000000)
                    
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
    
    def get_rl_model_info(self):
        """Get information about the RL model."""
        if self.rl_service:
            return self.rl_service.get_model_info()
        return {"rl_available": False}
    
    def train_rl_model(self, **kwargs):
        """Train the RL model with custom parameters."""
        try:
            import sys
            sys.path.append("rl")
            from train_rl_model import RLMealPlanTrainer
            
            trainer = RLMealPlanTrainer(
                data_path="dataset/product_dataset/final_usable_food_dataset.csv"
            )
            
            # Train with custom parameters
            model = trainer.train_model(**kwargs)
            
            # Reinitialize RL service with new model
            self._init_rl_service()
            
            logger.info("RL model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train RL model: {e}")
            return False

# Create singleton instance
similar_meal_service = SimilarMealPlanService()
