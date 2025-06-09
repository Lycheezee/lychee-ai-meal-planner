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
    def __init__(self):        # Update dataset path to use the final dataset
        self.df = pd.read_csv("dataset/process_dataset/final_usable_food_dataset.csv")
        self.features = ["calories", "proteins", "carbohydrates", "fats", "fibers", "sugars", "sodium", "cholesterol"]
        
        # Data cleaning
        self.df[self.features] = self.df[self.features].fillna(0)
        
        X = self.df[self.features]
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        self.knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
        self.knn.fit(self.X_scaled)
        
        self.start_foods = []

    def generate_similar_meal_plans(self, days=30, start_foods=[]):
        """Generate meal plans using traditional KNN-based approach."""
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

    def generate_meal_plan(self, input_food, days=1):
        """Generate a meal plan based on the input food using KNN."""
        food_matches = self.df[self.df['food_item'].str.lower().eq(input_food.lower())]
        
        if food_matches.empty:
            food_matches = self.df[self.df['food_item'].str.contains(input_food, case=False, na=False)]
        
        if food_matches.empty:
            return [input_food] * days
        
        food_index = food_matches.index[0]
        food_features = self.X_scaled[food_index].reshape(1, -1)
        
        distances, indices = self.knn.kneighbors(food_features)
        
        similar_foods = []
        for i in indices[0]:
            similar_foods.append(self.df.iloc[i]['food_item'])
        
        # Generate meal plan for specified days by cycling through similar foods
        meal_plan = []
        for day in range(days):
            food_index = day % len(similar_foods)
            meal_plan.append(similar_foods[food_index])
        
        return meal_plan

# Create singleton instance
similar_meal_service = SimilarMealPlanService()
