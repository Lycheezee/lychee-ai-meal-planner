# filepath: D:\Code\Lychee\lychee-meal-planners\systems\rl\train_rl_model.py
import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from meal_plan_env import MealPlanEnv
import pickle
from datetime import datetime

class RLMealPlanTrainer:
    """Trainer class for RL-based meal planning."""
    
    def __init__(self, 
                 data_path: str = "dataset/product_dataset/final_usable_food_dataset.csv",
                 model_save_path: str = "models/rl_meal_planner",
                 log_path: str = "logs/rl_training"):
        
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.log_path = log_path
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        
        # Load and prepare data
        self.df = self._load_and_prepare_data()
        
        # Default training configuration
        self.config = {
            'algorithm': 'PPO',
            'total_timesteps': 100000,
            'learning_rate': 3e-4,
            'batch_size': 64,
            'n_steps': 2048,
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
        
    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare the food dataset."""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Basic data cleaning
        nutrition_cols = ["calories", "proteins", "carbohydrates", "fats", "fibers", "sugars", "sodium", "cholesterol"]
        df[nutrition_cols] = df[nutrition_cols].fillna(0)
        
        # Remove rows with all zero nutrition values
        df = df[~(df[nutrition_cols] == 0).all(axis=1)]
        
        print(f"Loaded {len(df)} food items")
        return df
    
    def create_environment(self, initial_meals=None):
        """Create the meal planning environment."""
        env = MealPlanEnv(
            df=self.df,
            nutritional_targets=self.config['nutritional_targets'],
            max_steps=self.config['max_steps_per_episode']
        )
        
        if initial_meals:
            env.reset(options={'initial_meals': initial_meals})
        
        return env
    
    def train_model(self, algorithm='PPO', total_timesteps=100000, **kwargs):
        """Train the RL model."""
        print(f"Training {algorithm} model...")
        
        # Update config with provided parameters
        self.config.update(kwargs)
        self.config['algorithm'] = algorithm
        self.config['total_timesteps'] = total_timesteps
        
        # Create environment
        def make_env():
            env = self.create_environment()
            env = Monitor(env, self.log_path)
            return env
        
        # Create vectorized environment
        env = DummyVecEnv([make_env])
        
        # Create model based on algorithm
        if algorithm.upper() == 'PPO':
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.config['learning_rate'],
                n_steps=self.config['n_steps'],
                batch_size=self.config['batch_size'],
                verbose=1,
                tensorboard_log=self.log_path
            )
        elif algorithm.upper() == 'A2C':
            model = A2C(
                "MlpPolicy",
                env,
                learning_rate=self.config['learning_rate'],
                verbose=1,
                tensorboard_log=self.log_path
            )
        elif algorithm.upper() == 'DQN':
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=self.config['learning_rate'],
                batch_size=self.config['batch_size'],
                verbose=1,
                tensorboard_log=self.log_path
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Create evaluation environment
        eval_env = DummyVecEnv([make_env])
        
        # Create callback for evaluation
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.model_save_path}_best",
            log_path=self.log_path,
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Train the model
        print(f"Starting training for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name=f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Save the final model
        model.save(self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
        
        # Save configuration
        config_path = f"{self.model_save_path}_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        print(f"Configuration saved to {config_path}")
        
        return model
    
    def evaluate_model(self, model_path=None, num_episodes=10, initial_meals=None):
        """Evaluate the trained model."""
        if model_path is None:
            model_path = self.model_save_path
            
        print(f"Loading model from {model_path}...")
        
        # Determine algorithm from config
        config_path = f"{model_path}_config.pkl"
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            algorithm = config.get('algorithm', 'PPO')
        else:
            algorithm = 'PPO'  # Default
        
        # Load model
        if algorithm.upper() == 'PPO':
            model = PPO.load(model_path)
        elif algorithm.upper() == 'A2C':
            model = A2C.load(model_path)
        elif algorithm.upper() == 'DQN':
            model = DQN.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Create environment
        env = self.create_environment(initial_meals)
        
        # Evaluate
        episode_rewards = []
        meal_plans = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset(options={'initial_meals': initial_meals} if initial_meals else None)
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            meal_plans.append(env.get_meal_plan())
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}")
            print(f"Meal plan: {env.get_meal_plan()}")
            print("---")
        
        avg_reward = np.mean(episode_rewards)
        print(f"Average reward over {num_episodes} episodes: {avg_reward:.3f}")
        
        return {
            'average_reward': avg_reward,
            'episode_rewards': episode_rewards,
            'meal_plans': meal_plans
        }

def main():
    """Main training function."""
    trainer = RLMealPlanTrainer()
    
    # Train the model
    model = trainer.train_model(
        algorithm='PPO',
        total_timesteps=50000,
        learning_rate=3e-4
    )
    
    # Evaluate the model
    results = trainer.evaluate_model(num_episodes=5)
    
    print("Training completed!")
    print(f"Average evaluation reward: {results['average_reward']:.3f}")

if __name__ == "__main__":
    main()