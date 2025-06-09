#!/usr/bin/env python3
"""
Quick test script for RL meal planning functionality.
"""

import sys
import os
sys.path.append('rl')

def test_rl_environment():
    """Test the RL environment."""
    print("=" * 50)
    print("Testing RL Environment")
    print("=" * 50)
    
    try:
        import pandas as pd
        from meal_plan_env import MealPlanEnv
        
        # Load dataset
        df = pd.read_csv("dataset/product_dataset/final_usable_food_dataset.csv")
        print(f"✅ Dataset loaded: {len(df)} food items")
        
        # Create environment
        env = MealPlanEnv(df, max_steps=3)  # Short episode for testing
        print("✅ Environment created successfully")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Environment reset: observation shape = {obs.shape}")
        
        # Test step
        action = 0  # Select first food item
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Step executed: reward = {reward:.3f}, selected = {info['selected_meal']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_rl_trainer():
    """Test the RL trainer."""
    print("\n" + "=" * 50)
    print("Testing RL Trainer")
    print("=" * 50)
    
    try:
        from train_rl_model import RLMealPlanTrainer
        
        # Create trainer
        trainer = RLMealPlanTrainer()
        print("✅ Trainer created successfully")
        
        # Test short training
        print("🚀 Starting quick training (1000 timesteps)...")
        model = trainer.train_model(
            algorithm='PPO',
            total_timesteps=1000,  # Very short for testing
            learning_rate=3e-4
        )
        print("✅ Training completed successfully")
        
        # Test evaluation
        print("📊 Testing model evaluation...")
        results = trainer.evaluate_model(num_episodes=2)
        print(f"✅ Evaluation completed: avg reward = {results['average_reward']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer test failed: {e}")
        return False

def test_rl_service():
    """Test the RL service."""
    print("\n" + "=" * 50)
    print("Testing RL Service")
    print("=" * 50)
    
    try:
        from rl_meal_service import RLMealPlanService
        
        # Create service
        service = RLMealPlanService()
        print("✅ RL service created successfully")
        
        # Test model info
        info = service.get_model_info()
        print(f"✅ Model info: loaded = {info['model_loaded']}")
        
        # Test meal plan generation (with fallback if no model)
        meal_plan = service.generate_meal_plan_rl(
            initial_meals=["Chicken Breast", "Brown Rice"],
            days=5
        )
        print(f"✅ Generated meal plan: {len(meal_plan)} meals")
        print(f"   First 3 meals: {meal_plan[:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        return False

def test_integration():
    """Test integration with similar_meal_service."""
    print("\n" + "=" * 50)
    print("Testing Integration")
    print("=" * 50)
    
    try:
        sys.path.append('services')
        from similar_meal_service import SimilarMealPlanService
        
        # Create service
        service = SimilarMealPlanService()
        print("✅ Similar meal service created successfully")
        
        # Test RL model info
        rl_info = service.get_rl_model_info()
        print(f"✅ RL info: available = {rl_info.get('rl_available', False)}")
        
        # Test meal plan generation
        meal_plans = service.generate_similar_meal_plans(
            days=5,
            start_foods=["Chicken Breast"],
            use_rl=True
        )
        print(f"✅ Generated {len(meal_plans)} daily meal plans")
        print(f"   First day: {meal_plans[0]['date'][:10]} with {len(meal_plans[0]['meals'])} meals")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 RL Meal Planning Quick Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Environment", test_rl_environment()))
    results.append(("Trainer", test_rl_trainer()))
    results.append(("Service", test_rl_service()))
    results.append(("Integration", test_integration()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:15} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed! RL implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
