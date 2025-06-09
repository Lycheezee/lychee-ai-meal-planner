"""
Nutritional calculations and scoring for meal planning.
Contains functions for calculating nutritional scores and meal suitability.
"""

import pandas as pd
from typing import Dict
from .meal_preferences import get_meal_type_preferences

def calculate_nutritional_score(food_row: pd.Series, target_nutrition: Dict[str, float], 
                               meal_type: str, weights: Dict[str, float] = None) -> float:
    """
    Calculate a comprehensive nutritional score for food selection.
    
    Args:
        food_row: Pandas Series containing food nutritional data
        target_nutrition: Target nutritional values for the meal
        meal_type: Type of meal (Breakfast, Lunch, Snack, Dinner)
        weights: Optional custom weights for different nutrients
        
    Returns:
        Float score between 0 and 1 representing nutritional fit
    """
    if weights is None:
        weights = {
            'calories': 0.3,
            'protein': 0.25,
            'carbohydrates': 0.2,
            'fat': 0.15,
            'fiber': 0.1
        }
    
    score = 0.0
    
    # Calorie alignment score
    if target_nutrition.get('calories', 0) > 0:
        calorie_ratio = min(food_row['calories'] / target_nutrition['calories'], 2.0)
        calorie_score = 1.0 - abs(1.0 - calorie_ratio)
        score += weights['calories'] * max(0, calorie_score)
    
    # Protein score (higher is generally better)
    if target_nutrition.get('proteins', 0) > 0:
        protein_ratio = food_row['proteins'] / target_nutrition['proteins']
        protein_score = min(protein_ratio, 1.5) / 1.5  # Cap at 1.5x target
        score += weights['protein'] * protein_score
    
    # Carbohydrate alignment
    if target_nutrition.get('carbohydrates', 0) > 0:
        carb_ratio = food_row['carbohydrates'] / target_nutrition['carbohydrates']
        carb_score = 1.0 - abs(1.0 - min(carb_ratio, 2.0))
        score += weights['carbohydrates'] * max(0, carb_score)
    
    # Fat alignment
    if target_nutrition.get('fats', 0) > 0:
        fat_ratio = food_row['fats'] / target_nutrition['fats']
        fat_score = 1.0 - abs(1.0 - min(fat_ratio, 2.0))
        score += weights['fat'] * max(0, fat_score)
    
    # Fiber bonus (higher is better)
    if target_nutrition.get('fibers', 0) > 0:
        fiber_score = min(food_row['fibers'] / target_nutrition['fibers'], 1.0)
        score += weights['fiber'] * fiber_score
    
    return score

def calculate_meal_suitability_score(food_row: pd.Series, meal_type: str) -> float:
    """
    Calculate how suitable a food is for a specific meal type based on nutritional content.
    
    Args:
        food_row: Pandas Series containing food nutritional data
        meal_type: Type of meal (Breakfast, Lunch, Snack, Dinner)
        
    Returns:
        Float score between 0 and 1 representing meal suitability
    """
    preferences = get_meal_type_preferences()[meal_type]
    score = 0.5  # Base score
    
    # Check calorie appropriateness
    if food_row['calories'] <= preferences['max_calories']:
        score += 0.2
    else:
        # Penalize foods that are too high in calories for the meal
        excess_ratio = food_row['calories'] / preferences['max_calories']
        score -= min(0.3, (excess_ratio - 1) * 0.2)
    
    # Check protein content
    if food_row['proteins'] >= preferences['min_protein']:
        score += 0.2
    
    # Check carbohydrate content
    if food_row['carbohydrates'] >= preferences['min_carbs']:
        score += 0.1
    
    # Fiber bonus
    if food_row['fibers'] > 3:  # Good fiber content
        score *= preferences['fiber_bonus']
    
    return max(0, min(1.0, score))  # Clamp between 0 and 1

def calculate_nutrient_density_score(food_row: pd.Series) -> float:
    """
    Calculate nutrient density score based on beneficial nutrients per calorie.
    
    Args:
        food_row: Pandas Series containing food nutritional data
        
    Returns:
        Float score representing nutrient density
    """
    if food_row['calories'] <= 0:
        return 0.0
    
    # Calculate beneficial nutrients per calorie
    protein_density = food_row['proteins'] / food_row['calories']
    fiber_density = food_row['fibers'] / food_row['calories']
    
    # Penalize high sugar content
    sugar_penalty = min(food_row.get('sugars', 0) / food_row['calories'], 0.3)
    
    # Penalize high sodium content
    sodium_penalty = min(food_row.get('sodium', 0) / (food_row['calories'] * 100), 0.2)
    
    density_score = (protein_density * 10 + fiber_density * 20) - sugar_penalty - sodium_penalty
    
    return max(0, min(1.0, density_score))

def get_default_nutritional_weights() -> Dict[str, float]:
    """Get default weights for nutritional scoring."""
    return {
        'calories': 0.3,
        'protein': 0.25,
        'carbohydrates': 0.2,
        'fat': 0.15,
        'fiber': 0.1
    }

def validate_nutritional_data(food_row: pd.Series) -> bool:
    """
    Validate that food nutritional data is complete and reasonable.
    
    Args:
        food_row: Pandas Series containing food nutritional data
        
    Returns:
        Bool indicating if data is valid
    """
    required_fields = ['calories', 'proteins', 'carbohydrates', 'fats', 'fibers']
    
    # Check for missing fields
    for field in required_fields:
        if field not in food_row or pd.isna(food_row[field]):
            return False
    
    # Check for negative values
    for field in required_fields:
        if food_row[field] < 0:
            return False
    
    # Check for unreasonable values
    if food_row['calories'] > 1000:  # Very high calorie food
        return False
    
    if food_row['proteins'] > 100:  # More than 100g protein per serving
        return False
    
    return True
