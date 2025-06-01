# Dietary Guidelines for Americans 2020–2025 (U.S. Health.gov) — energy, macronutrient percentage ranges
# Protein and Fiber Fact Sheets (NIH Office of Dietary Supplements) — RDA for protein and fiber
# WHO Guidelines on Sugars Intake — recommended limit for added sugars
# FDA Guidance on Sodium — daily sodium limit
# American Heart Association — dietary cholesterol recommendation

male_targets = {
    'calories': 2500,
    'proteins': 56,
    'carbohydrates': 350,
    'fats': 80,
    'fibers': 38,
    'sugars': 50,
    'sodium': 2300,
    'cholesterol': 300
}

female_targets = {
    'calories': 2000,
    'proteins': 46,
    'carbohydrates': 275,
    'fats': 65,
    'fibers': 25,
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


target_adjustment = {
    'Low': 0.8,
    'Low+': 0.9,
    'Med': 1.0,
    'High-': 1.1,
    'High': 1.2
}


gender_adjusment = 0.02