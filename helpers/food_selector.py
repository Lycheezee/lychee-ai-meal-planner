"""
Food selection logic for meal planning.
Contains functions for selecting optimal foods based on nutritional requirements and meal types.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from .nutritional_calculator import calculate_nutritional_score, calculate_meal_suitability_score

def select_optimal_food(df: pd.DataFrame, meal_type: str, 
                       target_nutrition: Dict[str, float], selected_foods: List[str] = None) -> Optional[pd.Series]:
    """
    Select optimal food item for a specific meal type based on nutritional content.
    
    Args:
        df: DataFrame containing food data
        meal_type: Type of meal (Breakfast, Lunch, Snack, Dinner)
        target_nutrition: Target nutritional values for the meal
        selected_foods: List of already selected food items to avoid repetition
        
    Returns:
        Pandas Series of the selected food item or None if no suitable food found
    """
    if selected_foods is None:
        selected_foods = []
    
    # Filter foods that haven't been selected yet
    available_foods = df[~df['food_item'].isin(selected_foods)]
    
    if available_foods.empty:
        # If no new foods available, allow repetition
        available_foods = df.copy()
    
    if available_foods.empty:
        return None
    
    # Calculate scores for each food
    scores = []
    for idx, food_row in available_foods.iterrows():
        nutritional_score = calculate_nutritional_score(food_row, target_nutrition, meal_type)
        meal_suitability = calculate_meal_suitability_score(food_row, meal_type)
        # Combine nutritional score with meal suitability
        combined_score = nutritional_score * 0.6 + meal_suitability * 0.4
        scores.append((idx, combined_score))
    
    # Select the food with the highest combined score
    best_food_idx = max(scores, key=lambda x: x[1])[0]
    return available_foods.loc[best_food_idx]

def select_diverse_foods(df: pd.DataFrame, meal_type: str, 
                        target_nutrition: Dict[str, float], 
                        num_foods: int = 2, selected_foods: List[str] = None) -> List[pd.Series]:
    """
    Select multiple diverse foods for a meal to meet nutritional targets.
    
    Args:
        df: DataFrame containing food data
        meal_type: Type of meal
        target_nutrition: Target nutritional values
        num_foods: Number of foods to select
        selected_foods: Previously selected foods to avoid
        
    Returns:
        List of selected food items
    """
    if selected_foods is None:
        selected_foods = []
    
    selected_for_meal = []
    remaining_targets = target_nutrition.copy()
    
    for i in range(num_foods):
        # Calculate targets for this food item
        foods_remaining = num_foods - i
        if foods_remaining > 0:
            item_targets = {k: v / foods_remaining for k, v in remaining_targets.items()}
        else:
            item_targets = remaining_targets.copy()
        
        # Select optimal food for current targets
        selected_food = select_optimal_food(df, meal_type, item_targets, 
                                          selected_foods + [f['food_item'] for f in selected_for_meal])
        
        if selected_food is not None:
            selected_for_meal.append(selected_food)
            
            # Update remaining targets
            for nutrient in remaining_targets:
                if nutrient in selected_food:
                    remaining_targets[nutrient] = max(0, remaining_targets[nutrient] - selected_food[nutrient])
    
    return selected_for_meal

def filter_foods_by_criteria(df: pd.DataFrame, criteria: Dict[str, any]) -> pd.DataFrame:
    """
    Filter foods based on specific criteria.
    
    Args:
        df: DataFrame containing food data
        criteria: Dictionary of filtering criteria
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Filter by calorie range
    if 'min_calories' in criteria:
        filtered_df = filtered_df[filtered_df['calories'] >= criteria['min_calories']]
    if 'max_calories' in criteria:
        filtered_df = filtered_df[filtered_df['calories'] <= criteria['max_calories']]
    
    # Filter by protein content
    if 'min_protein' in criteria:
        filtered_df = filtered_df[filtered_df['proteins'] >= criteria['min_protein']]
    
    # Filter by carbohydrate content
    if 'max_carbs' in criteria:
        filtered_df = filtered_df[filtered_df['carbohydrates'] <= criteria['max_carbs']]
    
    # Filter by fat content
    if 'max_fats' in criteria:
        filtered_df = filtered_df[filtered_df['fats'] <= criteria['max_fats']]
    
    # Filter by fiber content
    if 'min_fiber' in criteria:
        filtered_df = filtered_df[filtered_df['fibers'] >= criteria['min_fiber']]
    
    # Exclude specific foods
    if 'exclude_foods' in criteria:
        filtered_df = filtered_df[~filtered_df['food_item'].isin(criteria['exclude_foods'])]
    
    # Include only specific foods
    if 'include_only' in criteria:
        filtered_df = filtered_df[filtered_df['food_item'].isin(criteria['include_only'])]
    
    return filtered_df

def rank_foods_by_score(df: pd.DataFrame, meal_type: str, 
                       target_nutrition: Dict[str, float], top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Rank foods by their combined nutritional and suitability scores.
    
    Args:
        df: DataFrame containing food data
        meal_type: Type of meal
        target_nutrition: Target nutritional values
        top_n: Number of top foods to return
        
    Returns:
        List of tuples (food_name, score) sorted by score
    """
    food_scores = []
    
    for idx, food_row in df.iterrows():
        nutritional_score = calculate_nutritional_score(food_row, target_nutrition, meal_type)
        meal_suitability = calculate_meal_suitability_score(food_row, meal_type)
        combined_score = nutritional_score * 0.6 + meal_suitability * 0.4
        food_scores.append((food_row['food_item'], combined_score))
    
    # Sort by score in descending order and return top N
    food_scores.sort(key=lambda x: x[1], reverse=True)
    return food_scores[:top_n]

def get_food_alternatives(df: pd.DataFrame, original_food: str, 
                         meal_type: str, num_alternatives: int = 5) -> List[str]:
    """
    Get alternative foods similar to the original food for the same meal type.
    
    Args:
        df: DataFrame containing food data
        original_food: Name of the original food
        meal_type: Type of meal
        num_alternatives: Number of alternatives to return
        
    Returns:
        List of alternative food names
    """
    # Find the original food
    original_row = df[df['food_item'] == original_food]
    if original_row.empty:
        return []
    
    original_row = original_row.iloc[0]
    
    # Use original food's nutrition as target
    target_nutrition = {
        'calories': original_row['calories'],
        'proteins': original_row['proteins'],
        'carbohydrates': original_row['carbohydrates'],
        'fats': original_row['fats'],
        'fibers': original_row['fibers']
    }
    
    # Exclude the original food from alternatives
    alternative_df = df[df['food_item'] != original_food]
    
    # Rank and return top alternatives
    alternatives = rank_foods_by_score(alternative_df, meal_type, target_nutrition, num_alternatives)
    return [food[0] for food in alternatives]
