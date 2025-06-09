# filepath: D:\Code\Lychee\lychee-meal-planners\systems\rl\rl_meal_service.py
import pandas as pd
import numpy as np
import os
import pickle
from typing import List, Dict, Optional, Any
from stable_baselines3 import PPO, A2C, DQN
from meal_plan_env import MealPlanEnv
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLMealPlanService:
    """Service class for RL-based meal planning."""
    
    def __init__(self, 
                 data_path: str = "dataset/product_dataset/final_usable_food_dataset.csv",
                 model_path: str = "models/rl_meal_planner"):
        
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.config = None
        self.df = None
        self.env = None
        
        # Load data and model
        self._load_data()
        self._load_model()
    
    def _load_data(self):
        """Load and prepare the food dataset."""
        try:
            logger.info(f"Loading data from {self.data_path}...")
            self.df = pd.read_csv(self.data_path)
            
            # Basic data cleaning
            nutrition_cols = ["calories", "proteins", "carbohydrates", "fats", "fibers", "sugars", "sodium", "cholesterol"]
            self.df[nutrition_cols] = self.df[nutrition_cols].fillna(0)
            
            # Remove rows with all zero nutrition values
            self.df = self.df[~(self.df[nutrition_cols] == 0).all(axis=1)]
            
            logger.info(f"Loaded {len(self.df)} food items")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _load_model(self):
        """Load the trained RL model."""
        try:
            # Load configuration
            config_path = f"{self.model_path}_config.pkl"
            if os.path.exists(config_path):
                with open(config_path, 'rb') as f:
                    self.config = pickle.load(f)
                logger.info("Configuration loaded successfully")
            else:
                # Default configuration
                self.config = {
                    'algorithm': 'PPO',
                    'max_steps_per_episode': 7,
                    'nutritional_targets': {
                        "calories": 2000.0,
                        "proteins": 150.0,
                        "carbohydrates": 250.0,
                        "fats": 65.0,
                        "fibers": 25.0,
                        "sugars": 50.0,
                        "sodium": 2300.0,
                        "cholesterol": 300.0
                    }
                }
                logger.warning("Using default configuration")
            
            # Load model
            if os.path.exists(f"{self.model_path}.zip"):
                algorithm = self.config.get('algorithm', 'PPO')
                
                if algorithm.upper() == 'PPO':
                    self.model = PPO.load(self.model_path)
                elif algorithm.upper() == 'A2C':
                    self.model = A2C.load(self.model_path)
                elif algorithm.upper() == 'DQN':
                    self.model = DQN.load(self.model_path)
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")
                
                logger.info(f"Model loaded successfully: {algorithm}")
            else:
                logger.warning(f"Model file not found: {self.model_path}.zip")
                self.model = None
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def _create_environment(self, initial_meals: List[str] = None, nutritional_targets: Dict[str, float] = None):
        """Create the meal planning environment."""
        targets = nutritional_targets or self.config['nutritional_targets']
        
        env = MealPlanEnv(
            df=self.df,
            nutritional_targets=targets,
            max_steps=self.config['max_steps_per_episode']
        )
        
        return env
    
    def generate_meal_plan_rl(self, 
                             initial_meals: List[str] = None,
                             days: int = 7,
                             nutritional_targets: Dict[str, float] = None,
                             deterministic: bool = True) -> List[str]:
        """Generate a meal plan using the RL model."""
        
        if self.model is None:
            logger.warning("RL model not available, falling back to random selection")
            return self._fallback_meal_plan(initial_meals, days)
        
        try:
            # Create environment
            env = self._create_environment(initial_meals, nutritional_targets)
            
            # Reset environment with initial meals
            obs, _ = env.reset(options={'initial_meals': initial_meals} if initial_meals else None)
            
            # Generate meal plan
            meal_plan = []
            done = False
            step = 0
            
            while not done and step < days:
                # Predict action
                action, _ = self.model.predict(obs, deterministic=deterministic)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Add selected meal
                meal_plan.append(info['selected_meal'])
                
                done = terminated or truncated
                step += 1
            
            # If we need more days, repeat the pattern or use fallback
            while len(meal_plan) < days:
                if len(meal_plan) > 0:
                    # Repeat the last meal or use a pattern
                    meal_plan.append(meal_plan[-1])
                else:
                    meal_plan.append(self._get_random_meal())
            
            logger.info(f"Generated meal plan with {len(meal_plan)} meals using RL")
            return meal_plan[:days]
            
        except Exception as e:
            logger.error(f"Error generating RL meal plan: {e}")
            return self._fallback_meal_plan(initial_meals, days)
    
    def _fallback_meal_plan(self, initial_meals: List[str] = None, days: int = 7) -> List[str]:
        """Fallback meal plan generation using random selection."""
        meal_plan = []
        
        if initial_meals:
            # Use initial meals as starting point
            for i in range(days):
                if i < len(initial_meals):
                    meal_plan.append(initial_meals[i])
                else:
                    meal_plan.append(self._get_random_meal())
        else:
            # Random selection
            for _ in range(days):
                meal_plan.append(self._get_random_meal())
        
        return meal_plan
    
    def _get_random_meal(self) -> str:
        """Get a random meal from the dataset."""
        if self.df is not None and len(self.df) > 0:
            return self.df.sample(1)['food_item'].iloc[0]
        return "Default Meal"
    
    def generate_similar_meal_plans(self, 
                                   days: int = 30, 
                                   start_foods: List[str] = None,
                                   use_rl: bool = True,
                                   nutritional_targets: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Generate meal plans using RL or fallback method."""
        
        meal_plans = []
        start_date = datetime.now()
        
        # Process start foods (convert IDs to names if needed)
        processed_start_foods = self._process_start_foods(start_foods or [])
        
        if use_rl and self.model is not None:
            # Use RL to generate meal plans
            logger.info("Generating meal plans using RL model")
            
            # Generate a base meal plan using RL
            base_meal_plan = self.generate_meal_plan_rl(
                initial_meals=processed_start_foods,
                days=days,
                nutritional_targets=nutritional_targets,
                deterministic=False  # Add some randomness for variety
            )
            
            # Create daily plans
            for day in range(days):
                current_date = start_date + timedelta(days=day)
                
                meals = []
                if day < len(base_meal_plan):
                    meal_name = base_meal_plan[day]
                    food_id = self._get_food_id(meal_name)
                    
                    meals.append({
                        "foodId": food_id,
                        "status": "not_completed"
                    })
                
                daily_plan = {
                    "date": current_date.isoformat(),
                    "meals": meals
                }
                
                meal_plans.append(daily_plan)
        
        else:
            # Fallback to original method
            logger.info("Generating meal plans using fallback method")
            meal_plans = self._generate_fallback_plans(days, processed_start_foods, start_date)
        
        return meal_plans
    
    def _process_start_foods(self, start_foods: List[str]) -> List[str]:
        """Process start foods (convert IDs to names if needed)."""
        processed_foods = []
        
        for food in start_foods:
            # Check if it's already a food name in our dataset
            if self.df['food_item'].str.lower().eq(food.lower()).any():
                processed_foods.append(food)
            else:
                # Try to find by partial match
                matches = self.df[self.df['food_item'].str.contains(food, case=False, na=False)]
                if not matches.empty:
                    processed_foods.append(matches.iloc[0]['food_item'])
                else:
                    processed_foods.append(food)  # Keep original if no match
        
        return processed_foods
    
    def _get_food_id(self, food_name: str) -> str:
        """Get food ID for a given food name."""
        # Since the dataset doesn't have IDs, we'll use the index or create a hash
        matches = self.df[self.df['food_item'] == food_name]
        if not matches.empty:
            return str(hash(food_name) % 10000000)  # Simple hash-based ID
        return ""
    
    def _generate_fallback_plans(self, days: int, start_foods: List[str], start_date: datetime) -> List[Dict[str, Any]]:
        """Generate meal plans using fallback method."""
        meal_plans = []
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            meals = []
            
            # Use start foods or random selection
            if start_foods and day < len(start_foods):
                food_name = start_foods[day]
            elif start_foods:
                food_name = start_foods[day % len(start_foods)]
            else:
                food_name = self._get_random_meal()
            
            food_id = self._get_food_id(food_name)
            
            meals.append({
                "foodId": food_id,
                "status": "not_completed"
            })
            
            daily_plan = {
                "date": current_date.isoformat(),
                "meals": meals
            }
            
            meal_plans.append(daily_plan)
        
        return meal_plans
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_loaded': self.model is not None,
            'algorithm': self.config.get('algorithm', 'Unknown') if self.config else 'Unknown',
            'config': self.config,
            'data_size': len(self.df) if self.df is not None else 0
        }