import pandas as pd

from helpers.fuzzy_system import fuzzy_calorie_adjustment
from constant import female_targets, male_targets, macro_preferences
 
def adjust_targets_for_macro(targets, preference):
    pref = preference.lower()
    if pref not in macro_preferences:
        print(f"Warning: Preference '{preference}' not found. Using 'balanced'.")
        pref = 'balanced'

    adjusted = targets.copy()

    # Adjust macros only for Protein, Carbs, and Fat
    for macro in ["Protein (g)", "Carbohydrates (g)", "Fat (g)"]:
        adjusted[macro] = round(adjusted[macro] * macro_preferences[pref].get(macro, 1.0), 1)

    # Optional: Adjust calories roughly to match macro changes (assuming calories come mainly from these)
    # 1g Protein = 4 kcal, 1g Carb = 4 kcal, 1g Fat = 9 kcal
    calories = (
        adjusted["Protein (g)"] * 4 +
        adjusted["Carbohydrates (g)"] * 4 +
        adjusted["Fat (g)"] * 9
    )
    adjusted["Calories (kcal)"] = round(calories)

    return adjusted
 
def generate_first_meal_plan(df, gender, bmi, exercise_rate, age, macro_preference):
    adjustment = fuzzy_calorie_adjustment(bmi, exercise_rate, age)


    daily_targets = male_targets if gender.lower() == 'male' else female_targets
    daily_targets = adjust_targets_for_macro(daily_targets, macro_preference)

    for key in daily_targets:
        daily_targets[key] *= adjustment

    plan = []
    selected_categories = set()
    categories = df['Category'].unique()

    for category in categories:
        foods = df[df['Category'] == category]
        if not foods.empty:
            plan.append(foods.sample(1))
            selected_categories.add(category)

    meal_plan = pd.concat(plan)

    return meal_plan