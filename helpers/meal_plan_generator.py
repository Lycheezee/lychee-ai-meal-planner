"""
Main meal plan generation logic.
Contains the primary function for generating comprehensive meal plans.
"""

import pandas as pd
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .fuzzy_system import fuzzy_calorie_adjustment
from .macro_adjuster import adjust_targets_for_macro
from .food_selector import select_optimal_food

from constant import female_targets, male_targets

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
        Tuple of (meal_plan_dataframe, daily_targets_dict)
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
    
    # Step 2: Find foods that meet daily nutritional targets
    meal_plans = []
    selected_foods = []
    
    # Select foods to meet the entire daily target directly
    foods_for_daily_plan = select_foods_for_daily_targets(
        df=df,
        daily_targets=daily_targets,
        max_foods=8  # Reasonable number of foods per day
    )
    
    # Add all selected foods to meal plan
    for food_record in foods_for_daily_plan:
        meal_plans.append(food_record)
    
    # Step 4: Create final meal plan DataFrame
    if meal_plans:
        final_meal_plan = create_meal_plan_dataframe(meal_plans)
        
        # Add nutrition comparison
        comparison_results = compare_meal_plan_to_targets(final_meal_plan, daily_targets)
        
        return final_meal_plan, daily_targets
    else:
        print("Warning: No suitable foods found for meal plan generation.")
        return pd.DataFrame(), daily_targets
    
def compare_meal_plan_to_targets(meal_plan: pd.DataFrame, daily_targets: Dict[str, float]) -> Dict[str, Dict]:
    """
    Sum all nutrition values from the meal plan and compare them to daily targets.
    
    Args:
        meal_plan: DataFrame containing the selected foods
        daily_targets: Target nutritional values for the day
        
    Returns:
        Dictionary containing totals, targets, differences, and percentages
    """
    print("\n" + "="*60)
    print("NUTRITION COMPARISON: MEAL PLAN vs DAILY TARGETS")
    print("="*60)
    
    if meal_plan.empty:
        print("❌ No meal plan to analyze - meal plan is empty")
        return {}
    
    # Calculate totals from meal plan
    nutritional_columns = ['calories', 'proteins', 'carbohydrates', 'fats', 'fibers', 'sugars', 'sodium', 'cholesterol']
    totals = {}
    
    print(f"📊 Analyzing {len(meal_plan)} selected foods:")
    for _, food in meal_plan.iterrows():
        food_name = food.get('food_item', 'Unknown Food')
        calories = food.get('calories', 0) if pd.notna(food.get('calories', 0)) else 0
        print(f"  • {food_name}: {calories:.1f} cal")
    
    print(f"\n🧮 Nutritional Totals:")
    print("-" * 60)
    
    comparison_results = {}
    print(meal_plan.columns)
    for nutrient in nutritional_columns:
        if nutrient in meal_plan.columns:
            # Sum all values for this nutrient, handling NaN values
            total = meal_plan[nutrient].fillna(0).sum()
            totals[nutrient] = round(total, 2)
            
            # Get target value
            target = daily_targets.get(nutrient, 0)
            
            # Calculate difference and percentage
            difference = total - target
            percentage = (total / target * 100) if target > 0 else 0
              # Determine status
            if target > 0:
                if nutrient in ['calories', 'carbohydrates', 'fats']:
                    # Stricter criteria for critical nutrients
                    if 95 <= percentage <= 105:
                        status = "🎯 Perfect"
                    elif 90 <= percentage <= 110:
                        status = "✓ Good"
                    elif 85 <= percentage <= 115:
                        status = "⚠️ Acceptable"
                    else:
                        status = "❌ Poor"
                else:
                    # Regular criteria for other nutrients
                    if 90 <= percentage <= 110:
                        status = "✅ Perfect"
                    elif 80 <= percentage <= 120:
                        status = "✓ Good"
                    elif percentage < 80:
                        status = "⚠️ Low"
                    else:
                        status = "⚠️ High"
            else:
                status = "❓ No target"
            
            # Store results
            comparison_results[nutrient] = {
                'total': total,
                'target': target,
                'difference': difference,
                'percentage': percentage,
                'status': status
            }
            
            # Print comparison
            print(f"{nutrient.capitalize():<15}: {total:>8.1f} / {target:>8.1f} ({percentage:>6.1f}%) {status}")
        else:
            print(f"{nutrient.capitalize():<15}: Column not found in meal plan")
            comparison_results[nutrient] = {
                'total': 0,
                'target': daily_targets.get(nutrient, 0),
                'difference': -daily_targets.get(nutrient, 0),
                'percentage': 0,
                'status': "❌ Missing"
            }
    
    # Calculate overall achievement
    valid_percentages = [result['percentage'] for result in comparison_results.values() 
                        if result['target'] > 0 and result['status'] != "❌ Missing"]
    
    if valid_percentages:
        avg_achievement = sum(valid_percentages) / len(valid_percentages)
        targets_in_range = sum(1 for result in comparison_results.values() 
                              if result['target'] > 0 and 80 <= result['percentage'] <= 120)
        total_targets = sum(1 for result in comparison_results.values() if result['target'] > 0)
        
        print(f"\n📈 OVERALL PERFORMANCE:")
        print("-" * 60)
        print(f"Average Achievement: {avg_achievement:.1f}%")
        print(f"Targets in Range (80-120%): {targets_in_range}/{total_targets}")
        
        if avg_achievement >= 90:
            overall_status = "🎯 Excellent meal plan!"
        elif avg_achievement >= 80:
            overall_status = "👍 Good meal plan"
        elif avg_achievement >= 70:
            overall_status = "👌 Acceptable meal plan"
        else:
            overall_status = "⚠️ Needs improvement"
        
        print(f"Overall Status: {overall_status}")
    
    print("="*60)
    
    return comparison_results

def create_meal_plan_dataframe(meal_plans: List[pd.Series]) -> pd.DataFrame:
    """
    Create a well-formatted DataFrame from the meal plan data.
    
    Args:
        meal_plans: List of pandas Series containing meal plan data
        
    Returns:
        Formatted DataFrame with proper column ordering and sorting
    """
    final_meal_plan = pd.DataFrame(meal_plans)

      # Reorder columns for better presentation
    column_order = ['food_item', 'calories', '_id', 'proteins', 'carbohydrates', 'fats', 'fibers', 'sugars', 'sodium', 'cholesterol']
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in column_order if col in final_meal_plan.columns]
    final_meal_plan = final_meal_plan[available_columns]
    
    # Sort by food name since we're not using meal types anymore
    if 'food_item' in final_meal_plan.columns:
        final_meal_plan = final_meal_plan.sort_values('food_item').reset_index(drop=True)
    print(final_meal_plan.columns)
    return final_meal_plan

def calculate_meal_plan_summary(meal_plan: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate nutritional summary of the complete meal plan.
    
    Args:
        meal_plan: DataFrame containing the meal plan
        
    Returns:
        Dictionary with total nutritional values
    """
    if meal_plan.empty:
        return {}
    
    summary = {}
    nutritional_columns = ['calories', 'proteins', 'carbohydrates', 'fats', 'fibers', 'sugars', 'sodium', 'cholesterol']
    
    for column in nutritional_columns:
        if column in meal_plan.columns:
            summary[column] = round(meal_plan[column].sum(), 2)
    
    return summary

def validate_meal_plan(meal_plan: pd.DataFrame, daily_targets: Dict[str, float], 
                      tolerance: float = 0.2) -> Dict[str, bool]:
    """
    Validate that the meal plan meets nutritional targets within tolerance.
    
    Args:
        meal_plan: Generated meal plan DataFrame
        daily_targets: Target nutritional values
        tolerance: Acceptable deviation from targets (0.2 = 20%)
        
    Returns:
        Dictionary indicating which targets are met
    """
    if meal_plan.empty:
        return {key: False for key in daily_targets.keys()}
    
    summary = calculate_meal_plan_summary(meal_plan)
    validation_results = {}
    
    for nutrient, target in daily_targets.items():
        if nutrient in summary:
            actual = summary[nutrient]
            min_acceptable = target * (1 - tolerance)
            max_acceptable = target * (1 + tolerance)
            validation_results[nutrient] = min_acceptable <= actual <= max_acceptable
        else:
            validation_results[nutrient] = False
    
    return validation_results

def generate_meal_plan_report(meal_plan: pd.DataFrame, daily_targets: Dict[str, float]) -> str:
    """
    Generate a formatted report of the meal plan.
    
    Args:
        meal_plan: Generated meal plan DataFrame
        daily_targets: Target nutritional values
        
    Returns:
        Formatted string report
    """
    if meal_plan.empty:
        return "No meal plan generated."
    
    summary = calculate_meal_plan_summary(meal_plan)
    validation = validate_meal_plan(meal_plan, daily_targets)
    
    report = ["=" * 50]
    report.append("MEAL PLAN REPORT")
    report.append("=" * 50)
      # Food breakdown
    report.append(f"\nSELECTED FOODS ({len(meal_plan)} items):")
    for _, food in meal_plan.iterrows():
        calories = food['calories'] if 'calories' in food and pd.notna(food['calories']) else 0
        report.append(f"  - {food['food_item']} ({calories:.0f} cal)")
    
    # Nutritional summary
    report.append("\n" + "=" * 50)
    report.append("NUTRITIONAL SUMMARY")
    report.append("=" * 50)
    
    for nutrient in ['calories', 'proteins', 'carbohydrates', 'fats', 'fibers']:
        if nutrient in summary and nutrient in daily_targets:
            actual = summary[nutrient]
            target = daily_targets[nutrient]
            status = "✓" if validation.get(nutrient, False) else "✗"
            percentage = (actual / target * 100) if target > 0 else 0
            report.append(f"{nutrient.capitalize()}: {actual:.1f}/{target:.1f} ({percentage:.1f}%) {status}")
    
    return "\n".join(report)

def select_foods_for_daily_targets(df: pd.DataFrame, daily_targets: Dict[str, float], max_foods: int = 8) -> List[pd.Series]:
    """
    Select foods that collectively meet the daily nutritional targets with precision.
    Prioritizes staying within 95-105% for calories, carbohydrates, and fats.
    
    Args:
        df: Food database DataFrame
        daily_targets: Target nutritional values for the entire day
        max_foods: Maximum number of foods to select
        
    Returns:
        List of selected food items
    """
    selected_foods = []
    current_totals = {nutrient: 0.0 for nutrient in daily_targets.keys()}
    selected_food_names = []
    
    # Critical nutrients that must stay within 95-105%
    critical_nutrients = ['calories', 'carbohydrates', 'fats']
    
    print(f"\n🎯 TARGET PRECISION MODE for {critical_nutrients}")
    print(f"Daily targets: {daily_targets}")
    
    for i in range(max_foods):
        print(f"\n--- Selecting food {i+1}/{max_foods} ---")
        
        # Calculate remaining needs
        remaining_needs = {}
        for nutrient, target in daily_targets.items():
            remaining_needs[nutrient] = max(0, target - current_totals[nutrient])
        
        print(f"Current totals: {current_totals}")
        print(f"Remaining needs: {remaining_needs}")
        
        # Filter out already selected foods
        available_foods = df[~df['food_item'].isin(selected_food_names)]
        if available_foods.empty:
            print("No more available foods")
            break
        
        # Find best food with precision scoring
        best_food = None
        best_score = -999999
        
        for idx, food_row in available_foods.iterrows():
            # Calculate what totals would be after adding this food
            projected_totals = current_totals.copy()
            for nutrient in daily_targets.keys():
                if nutrient in food_row:
                    food_nutrient = food_row[nutrient] if pd.notna(food_row[nutrient]) else 0
                    projected_totals[nutrient] += food_nutrient
            
            # Calculate precision score
            score = 0
            penalty = 0
            
            for nutrient, target in daily_targets.items():
                if target > 0:
                    projected_value = projected_totals[nutrient]
                    percentage = (projected_value / target) * 100
                    
                    if nutrient in critical_nutrients:
                        # Heavy penalty for going outside 95-105% range for critical nutrients
                        if 95 <= percentage <= 105:
                            score += 100  # Perfect range bonus
                        elif 90 <= percentage <= 110:
                            score += 50   # Acceptable range
                        elif 85 <= percentage <= 115:
                            score += 10   # Tolerable range
                        else:
                            penalty += 1000  # Heavy penalty for going too far
                            
                        # Extra penalty for overshooting critical nutrients
                        if percentage > 105:
                            penalty += (percentage - 105) * 50
                    else:
                        # Less strict scoring for other nutrients
                        if 80 <= percentage <= 120:
                            score += 20
                        elif 70 <= percentage <= 130:
                            score += 10
                        else:
                            penalty += 100
            
            final_score = score - penalty
            
            if final_score > best_score:
                best_score = final_score
                best_food = food_row
        
        if best_food is not None:
            selected_foods.append(best_food)
            selected_food_names.append(best_food['food_item'])
            
            # Update current totals
            for nutrient in daily_targets.keys():
                if nutrient in best_food:
                    food_nutrient = best_food[nutrient] if pd.notna(best_food[nutrient]) else 0
                    current_totals[nutrient] += food_nutrient
            
            print(f"Selected: {best_food['food_item']} (score: {best_score:.1f})")
            
            # Check if we're in acceptable range for critical nutrients
            all_critical_in_range = True
            for nutrient in critical_nutrients:
                if nutrient in daily_targets and daily_targets[nutrient] > 0:
                    percentage = (current_totals[nutrient] / daily_targets[nutrient]) * 100
                    if not (95 <= percentage <= 105):
                        all_critical_in_range = False
                        break
            
            # Stop early if all critical nutrients are in perfect range
            if all_critical_in_range and i >= 3:  # Minimum 4 foods
                print(f"🎯 Perfect range achieved for critical nutrients after {i+1} foods!")
                break
        else:
            print("No suitable food found")
            break
    
    print(f"\n✅ Selected {len(selected_foods)} foods total")
    return selected_foods
