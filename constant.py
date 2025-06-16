# Dietary Guidelines for Americans 2020–2025 (U.S. Health.gov) — energy, macronutrient percentage ranges
# Protein and Fiber Fact Sheets (NIH Office of Dietary Supplements) — RDA for protein and fiber
# WHO Guidelines on Sugars Intake — recommended limit for added sugars
# FDA Guidance on Sodium — daily sodium limit
# American Heart Association — dietary cholesterol recommendation
# Calorie ranges updated based on research data for ages 18-60, weight ranges 50-90kg (male) and 45-85kg (female)

# Default targets (average across all age groups and activity levels)
male_targets = {
    'calories': 1794.29,
    'proteins': 67,        # 15% of calories, 4 kcal/g
    'carbohydrates': 247,   # 55% of calories, 4 kcal/g
    'fats': 60,            # 30% of calories, 9 kcal/g
    'fibers': 38,      # Based on dietary guidelines for adult men
    'sugars': 50,      # Limit based on WHO recommendations
    'sodium': 2300,    # FDA guidance in mg
    'cholesterol': 300 # AHA recommendation in mg
}

female_targets = {
    'calories': 1434.29,
    'proteins': 54,        # 15% of calories, 4 kcal/g
    'carbohydrates': 197,   # 55% of calories, 4 kcal/g
    'fats': 48,            # 30% of calories, 9 kcal/g
    'fibers': 25,      # Based on dietary guidelines for adult women
    'sugars': 50,      # Limit based on WHO recommendations
    'sodium': 2300,    # FDA guidance in mg
    'cholesterol': 300 # AHA recommendation in mg
}

test_targets = {
    'calories': 1614.29,
    'proteins': 61,
    'carbohydrates': 222,
    'fats': 54,
    'fibers': 32,
    'sugars': 50,
    'sodium': 2300, 
    'cholesterol': 300 
}

# Macro preferences adjustment factors (multiplicative)
macro_preferences = {
    "high protein": {
        "proteins": 1.75,        # 35% protein vs 20% baseline
        "carbohydrates": 0.7,   # 35% carbs vs 50%
        "fats": 1.0,             # 30% fat = baseline
    },
    "low carb": {
        "proteins": 1.5,         # 30% protein vs 20%
        "carbohydrates": 0.6,   # 30% carbs vs 50%
        "fats": 1.33,            # 40% fat vs 30%
    },
    "balanced": {
        "proteins": 1.0,         # 20%
        "carbohydrates": 1.0,   # 50%
        "fats": 1.0,             # 30%
    },
    "keto": {
        "proteins": 0.75,        # 15% protein vs 20%
        "carbohydrates": 0.2,   # 10% carbs vs 50%
        "fats": 2.5,             # 75% fat vs 30%
    },
    "high carb": {
        "proteins": 1.0,         # 20%
        "carbohydrates": 1.2,   # 60% carbs vs 50%
        "fats": 0.67,            # 20% fat vs 30%
    },
    "low fat": {
        "proteins": 1.25,        # 25% protein vs 20%
        "carbohydrates": 1.2,   # 60% carbs vs 50%
        "fats": 0.5,             # 15% fat vs 30%
    },
    "mediterranean": {
        "proteins": 0.75,        # 15% protein vs 20%
        "carbohydrates": 1.0,   # 50% carbs = baseline
        "fats": 1.17,            # 35% fat vs 30%
    }
}