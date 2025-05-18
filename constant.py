# Dietary Guidelines for Americans 2020–2025 (U.S. Health.gov) — energy, macronutrient percentage ranges
# Protein and Fiber Fact Sheets (NIH Office of Dietary Supplements) — RDA for protein and fiber
# WHO Guidelines on Sugars Intake — recommended limit for added sugars
# FDA Guidance on Sodium — daily sodium limit
# American Heart Association — dietary cholesterol recommendation

male_targets = {
    'Calories (kcal)': 2500,
    'Protein (g)': 56,
    'Carbohydrates (g)': 350,
    'Fat (g)': 80,
    'Fiber (g)': 38,
    'Sugars (g)': 50,
    'Sodium (mg)': 2300,
    'Cholesterol (mg)': 300
}

female_targets = {
    'Calories (kcal)': 2000,
    'Protein (g)': 46,
    'Carbohydrates (g)': 275,
    'Fat (g)': 65,
    'Fiber (g)': 25,
    'Sugars (g)': 50,
    'Sodium (mg)': 2300,
    'Cholesterol (mg)': 300
}

# Macro preferences adjustment factors (multiplicative)
macro_preferences = {
    "high protein": {
        "Protein (g)": 1.75,        # 35% protein vs 20% baseline
        "Carbohydrates (g)": 0.7,   # 35% carbs vs 50%
        "Fat (g)": 1.0,             # 30% fat = baseline
    },
    "low carb": {
        "Protein (g)": 1.5,         # 30% protein vs 20%
        "Carbohydrates (g)": 0.6,   # 30% carbs vs 50%
        "Fat (g)": 1.33,            # 40% fat vs 30%
    },
    "balanced": {
        "Protein (g)": 1.0,         # 20%
        "Carbohydrates (g)": 1.0,   # 50%
        "Fat (g)": 1.0,             # 30%
    },
    "keto": {
        "Protein (g)": 0.75,        # 15% protein vs 20%
        "Carbohydrates (g)": 0.2,   # 10% carbs vs 50%
        "Fat (g)": 2.5,             # 75% fat vs 30%
    },
    "high carb": {
        "Protein (g)": 1.0,         # 20%
        "Carbohydrates (g)": 1.2,   # 60% carbs vs 50%
        "Fat (g)": 0.67,            # 20% fat vs 30%
    },
    "low fat": {
        "Protein (g)": 1.25,        # 25% protein vs 20%
        "Carbohydrates (g)": 1.2,   # 60% carbs vs 50%
        "Fat (g)": 0.5,             # 15% fat vs 30%
    },
    "mediterranean": {
        "Protein (g)": 0.75,        # 15% protein vs 20%
        "Carbohydrates (g)": 1.0,   # 50% carbs = baseline
        "Fat (g)": 1.17,            # 35% fat vs 30%
    }
}


target_adjustment = {
    'Low': 0.8,
    'Low+': 0.9,
    'Med': 1.0,
    'High-': 1.1,
    'High': 1.2
}


gender_adjusment = 0.02