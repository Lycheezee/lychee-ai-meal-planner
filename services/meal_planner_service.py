import pandas as pd
from helpers.helper import generate_first_meal_plan
from helpers.calculate_bmi import calculate_bmi
from helpers.calculate_age import calculate_age
from pandas.api.types import CategoricalDtype

def generate_meal_plan_api(
    height: float,
    weight: float,
    gender: str,
    exercise_rate: str,
    dob: str,
    macro_preference: str
):
    df = pd.read_csv("./dataset/daily_food_nutrition_dataset_with_ids.csv")
    
    age = calculate_age(dob)
    bmi = calculate_bmi(weight, height)
    
    meal_plan, daily_targets = generate_first_meal_plan(df, gender, bmi, exercise_rate, age, macro_preference)

    # Removed total row calculation and concatenation.
    # Reorder and export (optional)
    meal_order = ['Breakfast', 'Lunch', 'Snack', 'Dinner']
    meal_type_order = CategoricalDtype(categories=meal_order, ordered=True)
    
    result_df = meal_plan[['Meal_Type', 'Category', 'Food_Item']].copy()
    result_df['Meal_Type'] = result_df['Meal_Type'].astype(meal_type_order)
    result_df = result_df.sort_values('Meal_Type').reset_index(drop=True)
    result_df.to_csv("results/first_meal_plan.csv", index=False)

    return meal_plan.to_dict(orient="records"), daily_targets
