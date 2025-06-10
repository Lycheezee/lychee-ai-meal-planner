import pandas as pd
from helpers.meal_plan_generator import generate_first_meal_plan
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
    # Use the final cleaned dataset
    df = pd.read_csv("./dataset/process_dataset/final_usable_food_dataset.csv")

    age = calculate_age(dob)
    bmi = calculate_bmi(height,weight)
    
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
