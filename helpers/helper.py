import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .fuzzy_system import fuzzy_calorie_adjustment
from constant import female_targets, male_targets, macro_preferences, activity_calories, reference_weight
 
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


def get_meal_distribution() -> Dict[str, float]:
    """Define calorie distribution across meal types based on nutritional research"""
    return {
        'Breakfast': 0.25,  # 25% of daily calories
        'Lunch': 0.35,      # 35% of daily calories  
        'Dinner': 0.30,     # 30% of daily calories
        'Snack': 0.10       # 10% of daily calories
    }


def get_food_suitability_scores() -> Dict[str, Dict[str, float]]:
    """Define food category suitability for different meal types based on nutritional guidelines"""
    return {
        'Breakfast': {
            'Grains': 0.9,      # High suitability for breakfast
            'Dairy': 0.8,
            'Fruits': 0.7,
            'Meat': 0.6,        # Moderate (eggs, breakfast meats)
            'Beverages': 0.7,
            'Vegetables': 0.5,
            'Snacks': 0.3
        },
        'Lunch': {
            'Meat': 0.9,        # High protein for lunch
            'Vegetables': 0.8,
            'Grains': 0.7,
            'Dairy': 0.6,
            'Fruits': 0.5,
            'Beverages': 0.6,
            'Snacks': 0.4
        },
        'Dinner': {
            'Meat': 0.9,
            'Vegetables': 0.9,  # High vegetable content for dinner
            'Grains': 0.6,
            'Dairy': 0.5,
            'Fruits': 0.4,
            'Beverages': 0.5,
            'Snacks': 0.3
        },
        'Snack': {
            'Snacks': 0.9,      # Obvious choice for snacks
            'Fruits': 0.8,      # Healthy snack option
            'Dairy': 0.6,       # Yogurt, cheese
            'Beverages': 0.7,
            'Grains': 0.4,
            'Vegetables': 0.5,
            'Meat': 0.3
        }
    }


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
    calorie_ratio = min(food_row['Calories (kcal)'] / target_nutrition['calories'], 2.0)
    calorie_score = 1.0 - abs(1.0 - calorie_ratio)
    score += weights['calories'] * max(0, calorie_score)
    
    # Protein score (higher is generally better)
    protein_ratio = food_row['Protein (g)'] / target_nutrition['proteins']
    protein_score = min(protein_ratio, 1.5) / 1.5  # Cap at 1.5x target
    score += weights['protein'] * protein_score
    
    # Carbohydrate alignment
    carb_ratio = food_row['Carbohydrates (g)'] / target_nutrition['carbohydrates']
    carb_score = 1.0 - abs(1.0 - min(carb_ratio, 2.0))
    score += weights['carbohydrates'] * max(0, carb_score)
    
    # Fat alignment
    fat_ratio = food_row['Fat (g)'] / target_nutrition['fats']
    fat_score = 1.0 - abs(1.0 - min(fat_ratio, 2.0))
    score += weights['fat'] * max(0, fat_score)
    
    # Fiber bonus (higher is better)
    fiber_score = min(food_row['Fiber (g)'] / target_nutrition['fibers'], 1.0)
    score += weights['fiber'] * fiber_score
    
    return score


def select_optimal_food(df: pd.DataFrame, category: str, meal_type: str, 
                       target_nutrition: Dict[str, float], selected_foods: List[str] = None) -> pd.Series:
    """Select optimal food item from a category for a specific meal type"""
    if selected_foods is None:
        selected_foods = []
    
    # Filter foods from the category that haven't been selected yet
    category_foods = df[(df['Category'] == category) & (~df['Food_Item'].isin(selected_foods))]
    
    if category_foods.empty:
        # If no new foods available, allow repetition but from different meals
        category_foods = df[df['Category'] == category]
    
    if category_foods.empty:
        return None
    
    # Get suitability scores
    suitability_scores = get_food_suitability_scores()
    category_suitability = suitability_scores[meal_type].get(category, 0.5)
    
    # Calculate nutritional scores for each food
    scores = []
    for idx, food_row in category_foods.iterrows():
        nutritional_score = calculate_nutritional_score(food_row, target_nutrition, meal_type)
        # Combine nutritional score with meal suitability
        combined_score = nutritional_score * 0.7 + category_suitability * 0.3
        scores.append((idx, combined_score))
    
    # Select the food with the highest combined score
    best_food_idx = max(scores, key=lambda x: x[1])[0]
    return category_foods.loc[best_food_idx]


def generate_first_meal_plan(df: pd.DataFrame, gender: str, bmi: float, 
                           exercise_rate: str, age: int, macro_preference: str) -> pd.DataFrame:
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
    calorie_adjustment, bmi_mem, ex_mem, age_mem = fuzzy_calorie_adjustment(bmi, exercise_rate, age)
    print(f"Fuzzy system outputs - BMI membership: {bmi_mem}, Exercise membership: {ex_mem}, Age membership: {age_mem}")
    print(f"Calorie adjustment factor based on BMI, activity level, and age: {calorie_adjustment:.2f}, ")
    
    # Get base targets according to gender
    daily_targets = male_targets.copy() if gender.lower() == 'male' else female_targets.copy()
    
    # Adjust for macro preference (this should come before the calorie adjustment)
    daily_targets = adjust_targets_for_macro(daily_targets, macro_preference)
    
    # Apply personal adjustment factor to all nutritional targets
    for key in daily_targets:
        daily_targets[key] *= calorie_adjustment
    
    # Step 2: Distribute targets across meal types
    meal_distribution = get_meal_distribution()
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
        
        # Get available categories and their suitability for this meal
        suitability_scores = get_food_suitability_scores()
        meal_suitability = suitability_scores[meal_type]
        
        # Sort categories by suitability for this meal type
        sorted_categories = sorted(meal_suitability.items(), key=lambda x: x[1], reverse=True)
        
        # Select 2-3 food items for this meal based on calorie distribution
        foods_for_meal = []
        remaining_targets = meal_targets.copy()
        
        # Determine number of foods based on meal type and calorie allocation
        if meal_type == 'Snack':
            num_foods = 1
        elif meal_type == 'Breakfast':
            num_foods = 2
        else:  # Lunch, Dinner
            num_foods = 3
        
        for i in range(num_foods):
            if i < len(sorted_categories):
                category = sorted_categories[i][0]
                
                # Adjust targets for this food item (divide remaining by remaining foods)
                foods_remaining = num_foods - i
                item_targets = {k: v / foods_remaining for k, v in remaining_targets.items()}
                
                # Select optimal food from this category
                selected_food = select_optimal_food(df, category, meal_type, item_targets, selected_foods)
                
                if selected_food is not None:
                    # Add meal type information
                    food_record = selected_food.copy()
                    food_record['Meal_Type'] = meal_type
                    foods_for_meal.append(food_record)
                    selected_foods.append(selected_food['Food_Item'])
                    
                    # Update remaining targets
                    remaining_targets['calories'] -= selected_food['Calories (kcal)']
                    remaining_targets['proteins'] -= selected_food['Protein (g)']
                    remaining_targets['carbohydrates'] -= selected_food['Carbohydrates (g)']
                    remaining_targets['fats'] -= selected_food['Fat (g)']
                    remaining_targets['fibers'] -= selected_food['Fiber (g)']
        
        meal_plans.extend(foods_for_meal)
    
    # Step 4: Create final meal plan DataFrame
    if meal_plans:
        final_meal_plan = pd.DataFrame(meal_plans)
        
        # Reorder columns for better presentation
        column_order = ['Meal_Type', 'Category', 'Food_Item', 'Calories (kcal)', 
                       'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)', 
                       'Sugars (g)', 'Sodium (mg)', 'Cholesterol (mg)']
        
        # Only include columns that exist in the dataframe
        available_columns = [col for col in column_order if col in final_meal_plan.columns]
        final_meal_plan = final_meal_plan[available_columns]
        
        # Sort by meal type order
        meal_order = ['Breakfast', 'Lunch', 'Snack', 'Dinner']
        final_meal_plan['Meal_Type'] = pd.Categorical(final_meal_plan['Meal_Type'], 
                                                     categories=meal_order, ordered=True)
        final_meal_plan = final_meal_plan.sort_values('Meal_Type').reset_index(drop=True)
        
        return final_meal_plan
    else:
        print("Warning: No suitable foods found for meal plan generation.")
        return pd.DataFrame()
