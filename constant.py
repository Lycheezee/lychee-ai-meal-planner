# Dietary Guidelines for Americans 2020–2025 (U.S. Health.gov) — energy, macronutrient percentage ranges
# Protein and Fiber Fact Sheets (NIH Office of Dietary Supplements) — RDA for protein and fiber
# WHO Guidelines on Sugars Intake — recommended limit for added sugars
# FDA Guidance on Sodium — daily sodium limit
# American Heart Association — dietary cholesterol recommendation
# Calorie ranges updated based on research data for ages 18-60, weight ranges 50-90kg (male) and 45-85kg (female)

# Base calories for different age groups and activity levels based on Vietnamese dietary guidelines
activity_calories = {
    'male': {
        # Children & Youth (1-18 years)
        'youth': {
            'light': 2503,      # Light activity
            'moderate': 2948,   # Moderate activity - 1-2 years: 948 kcal/day, 17-18 years: 3410 kcal/day
            'vigorous': 3410    # Vigorous activity
        },
        # Adults (18-29.9 years)
        'young_adult': {
            'light': 3050,      # Light activity (PAL 1.75)
            'moderate': 3600,   # Moderate activity (PAL 1.75)
            'vigorous': 4200    # Vigorous activity (PAL 1.75)
        },
        # Adults (30-59.9 years)
        'adult': {
            'light': 2950,      # Light activity (PAL 1.75)
            'moderate': 3400,   # Moderate activity (PAL 1.75) 
            'vigorous': 3950    # Vigorous activity (PAL 1.75)
        },
        # Elderly (60+ years)
        'elderly': {
            'light': 2450,      # Light activity (PAL 1.75)
            'moderate': 2850,   # Moderate activity (PAL 1.75)
            'vigorous': 3300    # Vigorous activity (PAL 1.75)
        }
    },
    'female': {
        # Children & Youth (1-18 years)
        'youth': {
            'light': 2100,      # Light activity
            'moderate': 2450,   # Moderate activity - 1-2 years: 865 kcal/day, 17-18 years: 2503 kcal/day  
            'vigorous': 2800    # Vigorous activity
        },
        # Adults (18-29.9 years)
        'young_adult': {
            'light': 2400,      # Light activity (PAL 1.75)
            'moderate': 2800,   # Moderate activity (PAL 1.75)
            'vigorous': 3250    # Vigorous activity (PAL 1.75)
        },
        # Adults (30-59.9 years)  
        'adult': {
            'light': 2250,      # Light activity (PAL 1.75)
            'moderate': 2600,   # Moderate activity (PAL 1.75)
            'vigorous': 3000    # Vigorous activity (PAL 1.75)
        },
        # Elderly (60+ years)
        'elderly': {
            'light': 2100,      # Light activity (PAL 1.75)
            'moderate': 2450,   # Moderate activity (PAL 1.75)
            'vigorous': 2800    # Vigorous activity (PAL 1.75)
        }
    }
}

# Default targets (average across all age groups and activity levels)
male_targets = {
    'calories': 3218,  # Average of all male activity_calories: (2954+3617+3433+2867)/4
    'proteins': 80,    # ~10% of calories - 1g protein = 4 kcal
    'carbohydrates': 402,  # ~50% of calories - 1g carb = 4 kcal
    'fats': 107,       # ~30% of calories - 1g fat = 9 kcal
    'fibers': 38,      # Based on dietary guidelines for adult men
    'sugars': 50,      # Limit based on WHO recommendations
    'sodium': 2300,    # FDA guidance in mg
    'cholesterol': 300 # AHA recommendation in mg
}

female_targets = {
    'calories': 2584,  # Average of all female activity_calories: (2450+2817+2617+2450)/4
    'proteins': 65,    # ~10% of calories - 1g protein = 4 kcal
    'carbohydrates': 323,  # ~50% of calories - 1g carb = 4 kcal
    'fats': 86,        # ~30% of calories - 1g fat = 9 kcal
    'fibers': 25,      # Based on dietary guidelines for adult women
    'sugars': 50,      # Limit based on WHO recommendations
    'sodium': 2300,    # FDA guidance in mg
    'cholesterol': 300 # AHA recommendation in mg
}

test_targets = {
    'calories': 2901,
    'proteins': 73,
    'carbohydrates': 363,
    'fats': 97,
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