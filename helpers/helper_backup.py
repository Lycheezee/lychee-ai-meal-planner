import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .fuzzy_system import fuzzy_calorie_adjustment
from constant import female_targets, male_targets, macro_preferences
 
def adjust_targets_for_macro(targets, preference):
    """Adjust nutritional targets based on macro preferences"""
    pref = preference.lower()
    if pref not in macro_preferences:
        print(f"Warning: Preference '{preference}' not found. Using 'balanced'.")
        pref = 'balanced'

    adjusted = targets.copy()
    # Adjust macros only for Protein, Carbs, and Fat
    for macro in ["proteins", "carbohydrates", "fats"]:
        adjusted[macro] = round(adjusted[macro] * macro_preferences[pref].get(macro, 1.0), 1)

    # Optional: Adjust calories roughly to match macro changes (assuming calories come mainly from these)
    # 1g Protein = 4 kcal, 1g Carb = 4 kcal, 1g Fat = 9 kcal
    calories = (
        adjusted["proteins"] * 4 +
        adjusted["carbohydrates"] * 4 +
        adjusted["fats"] * 9
    )
    adjusted["calories"] = round(calories)

    return adjusted




def calculate_nutritional_score(food_row: pd.Series, target_nutrition: Dict[str, float], 
                               meal_type: str, weights: Dict[str, float] = None) -> float:
    """Calculate a comprehensive nutritional score for food selection"""
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
    calorie_ratio = min(food_row['calories'] / target_nutrition['calories'], 2.0)
    calorie_score = 1.0 - abs(1.0 - calorie_ratio)
    score += weights['calories'] * max(0, calorie_score)
    
    # Protein score (higher is generally better)
    protein_ratio = food_row['proteins'] / target_nutrition['proteins']
    protein_score = min(protein_ratio, 1.5) / 1.5  # Cap at 1.5x target
    score += weights['protein'] * protein_score
    
    # Carbohydrate alignment
    carb_ratio = food_row['carbohydrates'] / target_nutrition['carbohydrates']
    carb_score = 1.0 - abs(1.0 - min(carb_ratio, 2.0))
    score += weights['carbohydrates'] * max(0, carb_score)
    
    # Fat alignment
    fat_ratio = food_row['fats'] / target_nutrition['fats']
    fat_score = 1.0 - abs(1.0 - min(fat_ratio, 2.0))
    score += weights['fat'] * max(0, fat_score)
    
    # Fiber bonus (higher is better)
    fiber_score = min(food_row['fibers'] / target_nutrition['fibers'], 1.0)
    score += weights['fiber'] * fiber_score
    
    return score


def select_optimal_food(df: pd.DataFrame, meal_type: str, 
                       target_nutrition: Dict[str, float], selected_foods: List[str] = None) -> pd.Series:
    """Select optimal food item for a specific meal type based on nutritional content"""
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


def calculate_meal_suitability_score(food_row: pd.Series, meal_type: str) -> float:
    """Calculate how suitable a food is for a specific meal type based on nutritional content"""
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


def generate_first_meal_plan(df: pd.DataFrame, gender: str, bmi: float, 
                           exercise_rate: str, age: int, macro_preference: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Generate an optimized meal plan based on advanced nutritional algorithms.
    
    This function implements a comprehensive meal planning approach that:
    1. Adjusts nutritional targets based on personal characteristics
    2. Distributes nutrition optimally across meal types
    3. Selects foods based on meal suitability and nutritional value
    4. Ensures dietary variety and balance
    
    Args:
        df: Food database DataFrame
        gender: User's gender (male/female)
        bmi: Body Mass Index
        exercise_rate: Exercise activity level
        age: User's age
        macro_preference: Preferred macronutrient distribution
        
    Returns:
        pd.DataFrame: Optimized meal plan with nutritional information
    """
      # Step 1: Calculate personalized nutritional targets using the enhanced fuzzy system
    calorie_adjustment = fuzzy_calorie_adjustment(bmi, exercise_rate, age)
    print(f"Calorie adjustment factor based on BMI, activity level, and age: {calorie_adjustment:.2f}")
    
    # Get base targets according to gender
    daily_targets = male_targets.copy() if gender.lower() == 'male' else female_targets.copy()
    
    # Adjust for macro preference (this should come before the calorie adjustment)
    daily_targets = adjust_targets_for_macro(daily_targets, macro_preference)
    
    # Apply personal adjustment factor to all nutritional targets
    for key in daily_targets:
        daily_targets[key] *= calorie_adjustment
    
    meal_plans = []
    selected_foods = []
    
    # Step 3: Generate meals for each meal type
    for meal_type, calorie_ratio in meal_distribution.items():
        # Calculate nutritional targets for this meal
        meal_targets = {
            'calories': daily_targets['calories'] * calorie_ratio,
            'proteins': daily_targets['proteins'] * calorie_ratio,
            'carbohydrates': daily_targets['carbohydrates'] * calorie_ratio,
            'fats': daily_targets['fats'] * calorie_ratio,
            'fibers': daily_targets['fibers'] * calorie_ratio
        }
          # Get meal type preferences
        meal_preferences = get_meal_type_preferences()
        
        # Select 2-3 food items for this meal based on calorie distribution
        foods_for_meal = []
        remaining_targets = meal_targets.copy()
        
        for i in range(num_foods):
            # Adjust targets for this food item (divide remaining by remaining foods)
            foods_remaining = num_foods - i
            item_targets = {k: v / foods_remaining for k, v in remaining_targets.items()}
            
            # Select optimal food for this meal type
            selected_food = select_optimal_food(df, meal_type, item_targets, selected_foods)
            if selected_food is not None:
                # Add meal type information
                food_record = selected_food.copy()
                food_record['meal_type'] = meal_type
                foods_for_meal.append(food_record)
                selected_foods.append(selected_food['food_item'])
                
                # Update remaining targets
                remaining_targets['calories'] -= selected_food['calories']
                remaining_targets['proteins'] -= selected_food['proteins']
                remaining_targets['carbohydrates'] -= selected_food['carbohydrates']
                remaining_targets['fats'] -= selected_food['fats']
                remaining_targets['fibers'] -= selected_food['fibers']
        
        meal_plans.extend(foods_for_meal)
    
    # Step 4: Create final meal plan DataFrame
    if meal_plans:
        final_meal_plan = pd.DataFrame(meal_plans)        # Reorder columns for better presentation
        column_order = ['meal_type', 'food_item', 'calories', 
                       'proteins', 'carbohydrates', 'fats', 'fibers', 
                       'sugars', 'sodium', 'cholesterol']
        
        # Only include columns that exist in the dataframe
        available_columns = [col for col in column_order if col in final_meal_plan.columns]
        final_meal_plan = final_meal_plan[available_columns]
          # Sort by meal type order
        meal_order = ['Breakfast', 'Lunch', 'Snack', 'Dinner']
        final_meal_plan['meal_type'] = pd.Categorical(final_meal_plan['meal_type'], 
                                                     categories=meal_order, ordered=True)
        final_meal_plan = final_meal_plan.sort_values('meal_type').reset_index(drop=True)
        
        return final_meal_plan, daily_targets
    else:
        print("Warning: No suitable foods found for meal plan generation.")
        return pd.DataFrame(), daily_targets
