# Dietary Guidelines for Americans 2020–2025 (U.S. Health.gov) — energy, macronutrient percentage ranges
# Protein and Fiber Fact Sheets (NIH Office of Dietary Supplements) — RDA for protein and fiber
# WHO Guidelines on Sugars Intake — recommended limit for added sugars
# FDA Guidance on Sodium — daily sodium limit
# American Heart Association — dietary cholesterol recommendation
# Calorie ranges updated based on research data for ages 18-60, weight ranges 50-90kg (male) and 45-85kg (female)

# Base calories for different activity levels based on physical activity level (PAL)
activity_calories = {
    'male': {
        # Sedentary/light activity (PAL 1.40-1.69)
        'sedentary': {
            'min': 1700,    # Lower bound (50kg)
            'max': 3050,    # Upper bound (90kg)
            'mid': 2375     # Midpoint value
        },
        # Active/moderately active (PAL 1.70-1.99)
        'moderate': {
            'min': 2050,    # Lower bound (50kg)
            'max': 3600,    # Upper bound (90kg)
            'mid': 2825     # Midpoint value
        },
        # Vigorous/vigorously active (PAL 2.00-2.40)
        'active': {
            'min': 2350,    # Lower bound (50kg)
            'max': 4200,    # Upper bound (90kg)
            'mid': 3275     # Midpoint value
        }
    },
    'female': {
        # Sedentary/light activity (PAL 1.40-1.69)
        'sedentary': {
            'min': 1600,    # Lower bound (45kg)
            'max': 2800,    # Upper bound (85kg)
            'mid': 2200     # Midpoint value
        },
        # Active/moderately active (PAL 1.70-1.99)
        'moderate': {
            'min': 1800,    # Lower bound (45kg)
            'max': 3300,    # Upper bound (85kg)
            'mid': 2550     # Midpoint value
        },
        # Vigorous/vigorously active (PAL 2.00-2.40)
        'active': {
            'min': 2000,    # Lower bound (45kg)
            'max': 3850,    # Upper bound (85kg)
            'mid': 2925     # Midpoint value
        }
    }
}

# Default targets (moderate activity level with mid-range weight)
male_targets = {
    'calories': 2825,  # Based on moderate activity level midpoint
    'proteins': 70,    # ~10% of calories - 1g protein = 4 kcal
    'carbohydrates': 353,  # ~50% of calories - 1g carb = 4 kcal
    'fats': 94,        # ~30% of calories - 1g fat = 9 kcal
    'fibers': 38,      # Based on dietary guidelines for adult men
    'sugars': 50,      # Limit based on WHO recommendations
    'sodium': 2300,    # FDA guidance in mg
    'cholesterol': 300 # AHA recommendation in mg
}

female_targets = {
    'calories': 2550,  # Based on moderate activity level midpoint
    'proteins': 64,    # ~10% of calories - 1g protein = 4 kcal
    'carbohydrates': 319,  # ~50% of calories - 1g carb = 4 kcal
    'fats': 85,        # ~30% of calories - 1g fat = 9 kcal
    'fibers': 25,      # Based on dietary guidelines for adult women
    'sugars': 50,      # Limit based on WHO recommendations
    'sodium': 2300,    # FDA guidance in mg
    'cholesterol': 300 # AHA recommendation in mg
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


# Weight-based adjustment factors
weight_adjustment = {
    'very_light': 0.8,    # Lower end of weight range 
    'light': 0.9,         # Lower-middle of weight range
    'medium': 1.0,        # Middle of weight range
    'heavy': 1.1,         # Upper-middle of weight range
    'very_heavy': 1.2     # Upper end of weight range
}

# Age-based adjustment factors
age_adjustment = {
    'youth': 1.10,        # Ages 18-24, higher metabolism
    'young_adult': 1.05,  # Ages 25-35
    'adult': 1.00,        # Ages 36-50, reference level
    'middle_aged': 0.95,  # Ages 51-65
    'senior': 0.90,       # Ages 66-75
    'elderly': 0.85       # Ages 76+
}

# Activity level adjustment factors
activity_adjustment = {
    'sedentary': 0.85,     # Sedentary (PAL 1.40-1.55)
    'light': 0.95,         # Light activity (PAL 1.56-1.69)
    'moderate': 1.0,       # Moderate activity (PAL 1.70-1.85) - reference
    'active': 1.15,        # Active (PAL 1.86-1.99)
    'very_active': 1.30    # Very active (PAL 2.00-2.40)
}

# Gender-specific reference weights (kg)
reference_weight = {
    'male': 70,   # Reference weight for men
    'female': 60  # Reference weight for women
}