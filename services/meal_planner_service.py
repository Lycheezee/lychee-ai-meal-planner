import pandas as pd
from helpers.meal_plan_generator import generate_first_meal_plan
from helpers.deap_meal_generator import generate_deap_meal_plan
from helpers.calculate_bmi import calculate_bmi
from helpers.calculate_age import calculate_age
from helpers.fuzzy_system import fuzzy_calorie_adjustment
from helpers.macro_adjuster import adjust_targets_for_macro
from constant import female_targets, male_targets
from pandas.api.types import CategoricalDtype

def generate_meal_plan_api(
    height: float,
    weight: float,
    gender: str,
    exercise_rate: str,
    dob: str,
    macro_preference: str,
    use_deap: bool = True
):
    """
    Generate meal plan using either DEAP genetic algorithm or traditional method.
    
    Args:
        height: User's height in cm
        weight: User's weight in kg  
        gender: User's gender (male/female)
        exercise_rate: Exercise activity level
        dob: Date of birth
        macro_preference: Macro preference (balanced, high_protein, etc.)
        use_deap: Whether to use DEAP genetic algorithm (default: True)
        
    Returns:
        Tuple of (transformed_meal_plan, daily_targets)
    """
    # Use the final cleaned dataset
    df = pd.read_csv("./dataset/process_dataset/final_usable_food_dataset.csv")

    age = calculate_age(dob)
    bmi = calculate_bmi(height, weight)
    
    # Calculate personalized nutritional targets
    calorie_adjustment = fuzzy_calorie_adjustment(bmi, exercise_rate, age)
    print(f"Calorie adjustment factor: {calorie_adjustment:.2f}")
    
    # Get base targets according to gender
    daily_targets = male_targets.copy() if gender.lower() == 'male' else female_targets.copy()
    
    # Adjust for macro preference
    daily_targets = adjust_targets_for_macro(daily_targets, macro_preference)
    
    # Apply personal adjustment factor to all nutritional targets
    for key in daily_targets:
        daily_targets[key] *= calorie_adjustment
    
    print(f"Final daily targets: {daily_targets}")
    
    if use_deap:
        print("🧬 Using DEAP Genetic Algorithm for meal planning...")
        meal_plan = generate_deap_meal_plan(df, daily_targets)
    else:
        print("🔄 Using traditional greedy algorithm for meal planning...")
        meal_plan, _ = generate_first_meal_plan(df, gender, bmi, exercise_rate, age, macro_preference)
    
    if meal_plan.empty:
        print("❌ No meal plan generated! Falling back to traditional method...")
        meal_plan, daily_targets = generate_first_meal_plan(df, gender, bmi, exercise_rate, age, macro_preference)
    
    # Use the complete meal_plan instead of just selecting food_item
    result_df = meal_plan.copy()

    transformed_meal_plan = []
    for _, row in result_df.iterrows():
        transformed_item = {
            "foodId": row.get("_id", ""),
            "name": row.get("food_item", ""),
            "fats": float(row.get("fats", 0)),
            "calories": float(row.get("calories", 0)),
            "sugars": float(row.get("sugars", 0)),
            "proteins": float(row.get("proteins", 0)),
            "fibers": float(row.get("fibers", 0)),
            "sodium": float(row.get("sodium", 0)),
            "cholesterol": float(row.get("cholesterol", 0)),
            "carbohydrates": float(row.get("carbohydrates", 0))
        }
        transformed_meal_plan.append(transformed_item)

    return transformed_meal_plan, daily_targets
