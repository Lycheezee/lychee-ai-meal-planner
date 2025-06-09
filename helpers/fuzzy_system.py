from typing import Dict, List, Tuple

# 1. Membership definitions using trapezoidal/triangular helper

def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b <= x <= c:
        return 1.0
    if c < x < d:
        return (d - x) / (d - c)
    return 0.0

# 2. Fuzzy variables membership functions

BMI_CATEGORIES = {
    'Severely_Underweight': lambda bmi: trapmf(bmi, -1, -1, 14.5, 16.5),
    'Underweight':           lambda bmi: trapmf(bmi, 16.5, 17.5, 18.5, 19.5),
    'Normal':                lambda bmi: trapmf(bmi, 18.5, 21.7, 24.9, 26.5),
    'Overweight':            lambda bmi: trapmf(bmi, 24.9, 27.5, 29.9, 31.5),
    'Obese_I':               lambda bmi: trapmf(bmi, 29.9, 32.5, 34.9, 36.5),
    'Obese_II':              lambda bmi: trapmf(bmi, 34.9, 36.5, 50.0, 50.0),
}

EXERCISE_CATEGORIES = {
    'Sedentary':  lambda pal: trapmf(pal, 1.0, 1.0, 1.45, 1.55),
    'Light':      lambda pal: trapmf(pal, 1.40, 1.56, 1.69, 1.75),
    'Moderate':   lambda pal: trapmf(pal, 1.70, 1.75, 1.85, 1.90),
    'Active':     lambda pal: trapmf(pal, 1.86, 1.90, 1.99, 2.05),
    'Very_Active':lambda pal: trapmf(pal, 1.99, 2.05, 2.40, 2.40),
}

AGE_CATEGORIES = {
    'Youth':        lambda age: trapmf(age, 18, 18, 22, 25),
    'Young_Adult':  lambda age: trapmf(age, 25, 30, 35, 35),
    'Adult':        lambda age: trapmf(age, 36, 40, 46, 50),
    'Middle_Aged':  lambda age: trapmf(age, 51, 55, 61, 65),
    'Senior':       lambda age: trapmf(age, 66, 70, 75, 75),
    'Elderly':      lambda age: trapmf(age, 76, 80, 90, 90),
}

# 3. Rule base as a list of (bmi_cat, ex_cat, age_cat, adjustment)
Rules: List[Tuple[str, str, str, float]] = [
    # Severely underweight: big boosts
    ('Severely_Underweight', 'Any', 'Any',         1.30),
    # Underweight: moderate boost
    ('Underweight',           'Any', 'Any',         1.20),
    # Normal: maintenance or slight mod
    ('Normal',                'Sedentary', 'Middle_Aged', 0.85),
    ('Normal',                'Light',     'Any',         0.95),
    ('Normal',                'Moderate',  'Any',         1.00),
    ('Normal',                'Active',    'Any',         1.05),
    ('Normal',                'Very_Active','Any',         1.10),
    # Overweight and above: reductions
    ('Overweight',            'Any',       'Any',         0.85),
    ('Obese_I',               'Any',       'Any',         0.75),
    ('Obese_II',              'Any',       'Any',         0.70),
]

# 4. Fuzzy inference engine
def fuzzy_calorie_adjustment(bmi: float, pal: float, age: float) -> float:
    # 4a. Compute all memberships
    bmi_mem = {cat: fn(bmi) for cat, fn in BMI_CATEGORIES.items()}
    ex_mem  = {cat: fn(pal)  for cat, fn in EXERCISE_CATEGORIES.items()}
    age_mem = {cat: fn(age) for cat, fn in AGE_CATEGORIES.items()}

    # 4b. Evaluate rules
    numerator = 0.0
    denominator = 0.0
    for bmi_cat, ex_cat, age_cat, adj in Rules:
        # 'Any' matches sum of all categories
        mem = bmi_mem[bmi_cat]
        if ex_cat != 'Any': mem = min(mem, ex_mem[ex_cat])
        if age_cat != 'Any': mem = min(mem, age_mem[age_cat])
        numerator   += mem * adj
        denominator += mem

    # 4c. Defuzzify by weighted average
    if denominator < 1e-6:
        return 1.0
    result = numerator / denominator
    return round(max(0.6, min(1.5, result)), 5)

# Example usage:
if __name__ == '__main__':
    print(fuzzy_calorie_adjustment(bmi=23, pal=1.2, age=23))
