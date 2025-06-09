# filepath: D:\Code\Lychee\lychee-meal-planners\systems\rl\meal_plan_env.py
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import List, Tuple, Dict, Any
import random

class MealPlanEnv(gym.Env):
    """
    Gymnasium environment for meal planning using reinforcement learning.
    
    The agent learns to select meals that:
    1. Meet nutritional targets
    2. Provide variety (avoid repetition)
    3. Consider similarity to initial meals
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 nutritional_targets: Dict[str, float] = None,
                 max_steps: int = 7,
                 similarity_weight: float = 0.3,
                 variety_weight: float = 0.2,
                 nutrition_weight: float = 0.5):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.max_steps = max_steps
        self.similarity_weight = similarity_weight
        self.variety_weight = variety_weight
        self.nutrition_weight = nutrition_weight
        
        # Define nutritional features
        self.nutrition_features = ["calories", "proteins", "carbohydrates", "fats", "fibers", "sugars", "sodium", "cholesterol"]
        
        # Default nutritional targets (daily values)
        if nutritional_targets is None:
            self.nutritional_targets = {
                "calories": 2000.0,
                "proteins": 150.0,
                "carbohydrates": 250.0,
                "fats": 65.0,
                "fibers": 25.0,
                "sugars": 50.0,
                "sodium": 2300.0,
                "cholesterol": 300.0
            }
        else:
            self.nutritional_targets = nutritional_targets
            
        # Convert targets to numpy array for faster computation
        self.target_array = np.array([self.nutritional_targets[feature] for feature in self.nutrition_features], dtype=np.float32)
        
        # Normalize nutrition data
        self.nutrition_data = self.df[self.nutrition_features].fillna(0).values.astype(np.float32)
        
        # State space: [cumulative_nutrition (8), step_count (1), initial_meal_similarity (1)]
        state_dim = len(self.nutrition_features) + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space: select one food item
        self.action_space = spaces.Discrete(len(self.df))
        
        # Initialize state variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cumulative_nutrition = np.zeros(len(self.nutrition_features), dtype=np.float32)
        self.selected_meals = []
        self.selected_indices = set()
        
        # Set initial meals if provided in options
        if options and 'initial_meals' in options:
            self.initial_meals = options['initial_meals']
            self.initial_nutrition = self._calculate_average_nutrition(self.initial_meals)
        else:
            self.initial_meals = []
            self.initial_nutrition = np.zeros(len(self.nutrition_features), dtype=np.float32)
            
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if action >= len(self.df):
            action = action % len(self.df)
            
        # Get selected meal
        selected_meal = self.df.iloc[action]
        meal_nutrition = self.nutrition_data[action]
        
        # Update cumulative nutrition
        self.cumulative_nutrition += meal_nutrition
        
        # Add to selected meals
        self.selected_meals.append(selected_meal['food_item'])
        self.selected_indices.add(action)
        
        # Calculate reward
        reward = self._calculate_reward(action, meal_nutrition)
        
        # Update step count
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Prepare info
        info = {
            'selected_meal': selected_meal['food_item'],
            'cumulative_nutrition': self.cumulative_nutrition.copy(),
            'nutrition_score': self._calculate_nutrition_score(),
            'variety_score': self._calculate_variety_score(),
            'similarity_score': self._calculate_similarity_score(meal_nutrition)
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        # Normalize cumulative nutrition by targets
        normalized_nutrition = self.cumulative_nutrition / (self.target_array + 1e-8)
        
        # Add step progress
        step_progress = self.current_step / self.max_steps
        
        # Add similarity to initial meals
        similarity_score = self._calculate_similarity_score(self.cumulative_nutrition) if len(self.initial_meals) > 0 else 0.0
        
        observation = np.concatenate([
            normalized_nutrition,
            [step_progress],
            [similarity_score]
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_reward(self, action: int, meal_nutrition: np.ndarray) -> float:
        """Calculate reward for the selected action."""
        # Nutrition reward (how close to targets)
        nutrition_reward = self._calculate_nutrition_score()
        
        # Variety reward (avoid repetition)
        variety_reward = self._calculate_variety_score()
        
        # Similarity reward (similar to initial meals)
        similarity_reward = self._calculate_similarity_score(meal_nutrition)
        
        # Combine rewards
        total_reward = (
            self.nutrition_weight * nutrition_reward +
            self.variety_weight * variety_reward +
            self.similarity_weight * similarity_reward
        )
        
        return total_reward
    
    def _calculate_nutrition_score(self) -> float:
        """Calculate how well current nutrition meets targets."""
        if self.current_step == 0:
            return 0.0
            
        # Calculate percentage of targets met
        progress = self.cumulative_nutrition / (self.target_array + 1e-8)
        
        # Reward being close to 1.0 (100% of target)
        # Penalize being too far above or below
        target_ratio = 1.0 / self.max_steps * self.current_step  # Expected progress
        score = -np.mean(np.abs(progress - target_ratio))
        
        return score
    
    def _calculate_variety_score(self) -> float:
        """Calculate variety score (higher is better)."""
        if len(self.selected_indices) <= 1:
            return 1.0
            
        # Reward selecting different meals
        unique_meals = len(self.selected_indices)
        total_meals = self.current_step
        
        return unique_meals / total_meals if total_meals > 0 else 0.0
    
    def _calculate_similarity_score(self, meal_nutrition: np.ndarray) -> float:
        """Calculate similarity to initial meals."""
        if len(self.initial_meals) == 0:
            return 0.0
            
        # Calculate cosine similarity between meal and average initial nutrition
        meal_norm = np.linalg.norm(meal_nutrition)
        initial_norm = np.linalg.norm(self.initial_nutrition)
        
        if meal_norm == 0 or initial_norm == 0:
            return 0.0
            
        similarity = np.dot(meal_nutrition, self.initial_nutrition) / (meal_norm * initial_norm)
        return max(0.0, similarity)  # Ensure non-negative
    
    def _calculate_average_nutrition(self, meal_names: List[str]) -> np.ndarray:
        """Calculate average nutrition of given meals."""
        if not meal_names:
            return np.zeros(len(self.nutrition_features), dtype=np.float32)
            
        nutrition_sum = np.zeros(len(self.nutrition_features), dtype=np.float32)
        count = 0
        
        for meal_name in meal_names:
            # Find meal in dataframe
            meal_rows = self.df[self.df['food_item'].str.lower() == meal_name.lower()]
            if not meal_rows.empty:
                meal_nutrition = meal_rows.iloc[0][self.nutrition_features].fillna(0).values.astype(np.float32)
                nutrition_sum += meal_nutrition
                count += 1
                
        return nutrition_sum / count if count > 0 else nutrition_sum
    
    def get_meal_plan(self) -> List[str]:
        """Get the current meal plan."""
        return self.selected_meals.copy()
    
    def render(self, mode='human'):
        """Render the current state."""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Selected meals: {self.selected_meals}")
            print(f"Cumulative nutrition: {dict(zip(self.nutrition_features, self.cumulative_nutrition))}")
            print(f"Target nutrition: {self.nutritional_targets}")
            print("---")