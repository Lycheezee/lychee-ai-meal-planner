"""
Meal preferences and distribution configurations for meal planning.
Contains meal type distributions, preferences, and constraints.
"""

from typing import Dict, Any

# Number of food items per meal
num_foods = 2

def get_meal_type_preferences() -> Dict[str, Dict[str, Any]]:
    """
    Get meal type preferences including calorie limits and nutritional focus.
    
    Returns:
        Dict containing preferences for each meal type
    """
    return {
        'Breakfast': {
            'max_calories': 600,
            'min_protein': 15,
            'min_carbs': 30,
            'fiber_bonus': 1.2,
            'preferred_nutrients': ['proteins', 'carbohydrates', 'fibers'],
            'description': 'Energy-focused start to the day'
        },
        'Lunch': {
            'max_calories': 800,
            'min_protein': 25,
            'min_carbs': 40,
            'fiber_bonus': 1.1,
            'preferred_nutrients': ['proteins', 'carbohydrates', 'fats'],
            'description': 'Balanced midday meal'
        },
        'Snack': {
            'max_calories': 300,
            'min_protein': 5,
            'min_carbs': 15,
            'fiber_bonus': 1.3,
            'preferred_nutrients': ['fibers', 'proteins'],
            'description': 'Light nutritional boost'
        },
        'Dinner': {
            'max_calories': 700,
            'min_protein': 30,
            'min_carbs': 25,
            'fiber_bonus': 1.15,
            'preferred_nutrients': ['proteins', 'fats', 'fibers'],
            'description': 'Satisfying end-of-day meal'
        }
    }

def get_calorie_distribution() -> Dict[str, float]:
    """Get the calorie distribution across meal types."""
    return meal_distribution.copy()

def validate_meal_distribution() -> bool:
    """Validate that meal distribution sums to 1.0 (100%)."""
    total = sum(meal_distribution.values())
    return abs(total - 1.0) < 0.01  # Allow for small floating point errors
