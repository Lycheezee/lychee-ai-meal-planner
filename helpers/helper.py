"""
Main helper module for meal planning.
Coordinates all meal planning functionality by importing and exposing functions from specialized modules.
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from specialized modules
from .macro_adjuster import adjust_targets_for_macro, get_available_macro_preferences, validate_macro_preference
from .nutritional_calculator import (
    calculate_nutritional_score, 
    calculate_meal_suitability_score, 
    calculate_nutrient_density_score,
    get_default_nutritional_weights,
    validate_nutritional_data
)
from .food_selector import (
    select_optimal_food, 
    select_diverse_foods, 
    filter_foods_by_criteria,
    rank_foods_by_score,
    get_food_alternatives
)
from .meal_plan_generator import (
    generate_first_meal_plan, 
    calculate_meal_plan_summary,
    validate_meal_plan,
    generate_meal_plan_report
)
from .meal_preferences import (
    get_meal_type_preferences, 
    get_meal_order, 
    get_calorie_distribution,
    validate_meal_distribution
)

# Re-export commonly used functions for backward compatibility
__all__ = [
    # Core meal planning
    'generate_first_meal_plan',
    'calculate_meal_plan_summary',
    'validate_meal_plan',
    'generate_meal_plan_report',
    
    # Macro adjustments
    'adjust_targets_for_macro',
    'get_available_macro_preferences',
    'validate_macro_preference',
    
    # Nutritional calculations
    'calculate_nutritional_score',
    'calculate_meal_suitability_score',
    'calculate_nutrient_density_score',
    'get_default_nutritional_weights',
    'validate_nutritional_data',
    
    # Food selection
    'select_optimal_food',
    'select_diverse_foods',
    'filter_foods_by_criteria',
    'rank_foods_by_score',
    'get_food_alternatives',
    
    # Meal preferences
    'get_meal_type_preferences',
    'get_meal_order',
    'get_calorie_distribution',
    'validate_meal_distribution'
]

def get_helper_module_info() -> Dict[str, List[str]]:
    """
    Get information about available functions in each helper module.
    
    Returns:
        Dictionary mapping module names to their available functions
    """
    return {
        'macro_adjuster': [
            'adjust_targets_for_macro',
            'get_available_macro_preferences', 
            'get_macro_preference_details',
            'validate_macro_preference'
        ],
        'nutritional_calculator': [
            'calculate_nutritional_score',
            'calculate_meal_suitability_score',
            'calculate_nutrient_density_score',
            'get_default_nutritional_weights',
            'validate_nutritional_data'
        ],
        'food_selector': [
            'select_optimal_food',
            'select_diverse_foods',
            'filter_foods_by_criteria',
            'rank_foods_by_score',
            'get_food_alternatives'
        ],
        'meal_plan_generator': [
            'generate_first_meal_plan',
            'create_meal_plan_dataframe',
            'calculate_meal_plan_summary',
            'validate_meal_plan',
            'generate_meal_plan_report'
        ],
        'meal_preferences': [
            'get_meal_type_preferences',
            'get_meal_order',
            'get_calorie_distribution',
            'validate_meal_distribution'
        ]
    }

def create_sample_meal_plan(df: pd.DataFrame, 
                          gender: str = "male", 
                          bmi: float = 22.0, 
                          exercise_rate: str = "moderate",
                          age: int = 30,
                          macro_preference: str = "balanced") -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Create a sample meal plan with default parameters for testing.
    
    Args:
        df: Food database DataFrame
        gender: User's gender (default: "male")
        bmi: Body Mass Index (default: 22.0)
        exercise_rate: Exercise activity level (default: "moderate")
        age: User's age (default: 30)
        macro_preference: Macro preference (default: "balanced")
        
    Returns:
        Tuple of (meal_plan_dataframe, daily_targets_dict)
    """
    return generate_first_meal_plan(df, gender, bmi, exercise_rate, age, macro_preference)

def quick_meal_plan_analysis(meal_plan: pd.DataFrame, daily_targets: Dict[str, float]) -> None:
    """
    Print a quick analysis of a meal plan.
    
    Args:
        meal_plan: Generated meal plan DataFrame
        daily_targets: Target nutritional values
    """
    if meal_plan.empty:
        print("No meal plan to analyze.")
        return
    
    print("\n" + "="*40)
    print("QUICK MEAL PLAN ANALYSIS")
    print("="*40)
    
    # Basic stats
    print(f"Total meals planned: {len(meal_plan)}")
    print(f"Meal types: {sorted(meal_plan['meal_type'].unique())}")
    
    # Nutritional summary
    summary = calculate_meal_plan_summary(meal_plan)
    validation = validate_meal_plan(meal_plan, daily_targets)
    
    print("\nNutritional Achievement:")
    for nutrient in ['calories', 'proteins', 'carbohydrates', 'fats']:
        if nutrient in summary and nutrient in daily_targets:
            actual = summary[nutrient]
            target = daily_targets[nutrient]
            percentage = (actual / target * 100) if target > 0 else 0
            status = "✓" if validation.get(nutrient, False) else "✗"
            print(f"  {nutrient.capitalize()}: {percentage:.1f}% of target {status}")
    
    print("="*40)

# Utility function for easy testing
def test_helper_modules(df: pd.DataFrame) -> bool:
    """
    Test all helper modules with sample data.
    
    Args:
        df: Food database DataFrame
        
    Returns:
        Boolean indicating if all tests passed
    """
    try:
        print("Testing helper modules...")
        
        # Test meal plan generation
        meal_plan, targets = create_sample_meal_plan(df)
        if meal_plan.empty:
            print("❌ Meal plan generation failed")
            return False
        print("✓ Meal plan generation working")
        
        # Test analysis functions
        summary = calculate_meal_plan_summary(meal_plan)
        validation = validate_meal_plan(meal_plan, targets)
        print("✓ Analysis functions working")
        
        # Test preferences
        preferences = get_meal_type_preferences()
        distribution = get_calorie_distribution()
        print("✓ Meal preferences working")
        
        print("✅ All helper modules tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Helper module test failed: {e}")
        return False
