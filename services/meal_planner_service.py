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
    # Use the cleaned dataset with new column names
    df = pd.read_csv("./dataset/daily_food_nutrition_dataset_cleaned.csv")
    print(df)
    age = calculate_age(dob)
    bmi = calculate_bmi(weight, height)
    
    meal_plan, daily_targets = generate_first_meal_plan(df, gender, bmi, exercise_rate, age, macro_preference)

    # Reorder and export (optional)
    meal_order = ['Breakfast', 'Lunch', 'Snack', 'Dinner']
    meal_type_order = CategoricalDtype(categories=meal_order, ordered=True)
    
    result_df = meal_plan[['meal_type', 'category', 'food_item']].copy()
    result_df['meal_type'] = result_df['meal_type'].astype(meal_type_order)
    result_df = result_df.sort_values('meal_type').reset_index(drop=True)
    result_df.to_csv("results/first_meal_plan.csv", index=False)

    # Transform the meal plan data to the new format
    transformed_meal_plan = []
    for _, row in meal_plan.iterrows():
        transformed_item = {
            "foodId": row.get("id", ""),
            "foodItem": row.get("food_item", ""),
            "mealType": row.get("meal_type", ""),
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
