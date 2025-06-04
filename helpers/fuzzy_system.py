from constant import weight_adjustment, age_adjustment, activity_adjustment, reference_weight, activity_calories

def bmi_membership(bmi):
    """
    Calculate membership degrees for different BMI categories based on WHO standards.
    
    The World Health Organization BMI ranges:
    - Severely Underweight: < 16.5
    - Underweight: 16.5-18.4
    - Normal weight: 18.5-24.9
    - Overweight: 25.0-29.9
    - Obese Class I: 30.0-34.9
    - Obese Class II: 35.0-39.9
    - Obese Class III: ≥ 40.0
    """
    return {
        'Severely_Underweight': max(0, min(1, (16.5 - bmi) / 3)) if bmi < 16.5 else 0,
        'Underweight': max(0, min((bmi - 16.5) / 2, (18.5 - bmi) / 2)) if 16.5 <= bmi <= 18.5 else 0,
        'Normal': max(0, min((bmi - 18.5) / 2, (24.9 - bmi) / 2)) if 18.5 <= bmi <= 24.9 else 0,
        'Overweight': max(0, min((bmi - 25.0) / 2, (29.9 - bmi) / 2)) if 25.0 <= bmi <= 29.9 else 0,
        'Obese_I': max(0, min((bmi - 30.0) / 2, (34.9 - bmi) / 2)) if 30.0 <= bmi <= 34.9 else 0,
        'Obese_II': max(0, min((bmi - 35.0) / 2, (39.9 - bmi) / 2)) if 35.0 <= bmi <= 39.9 else 0,
        'Obese_III': max(0, min(1, (bmi - 40.0) / 5)) if bmi >= 40.0 else 0
    }

def exercise_membership(exercise_rate):
    """
    Map exercise rates to physical activity levels based on PAL values.
    
    Physical Activity Level (PAL) categories:
    - Sedentary: PAL 1.40-1.55
    - Light activity: PAL 1.56-1.69
    - Moderate activity: PAL 1.70-1.85
    - Active: PAL 1.86-1.99
    - Very Active: PAL 2.00-2.40
    
    These values align with the research data for ages 18-60.
    """
    mapping = {
        'sedentary': {
            'Sedentary': 1.0, 
            'Light': 0.0, 
            'Moderate': 0.0, 
            'Active': 0.0, 
            'Very_Active': 0.0
        },
        'light': {
            'Sedentary': 0.2, 
            'Light': 0.8, 
            'Moderate': 0.0, 
            'Active': 0.0, 
            'Very_Active': 0.0
        },
        'moderate': {
            'Sedentary': 0.0, 
            'Light': 0.2, 
            'Moderate': 0.8, 
            'Active': 0.0, 
            'Very_Active': 0.0
        },
        'active': {
            'Sedentary': 0.0, 
            'Light': 0.0, 
            'Moderate': 0.2, 
            'Active': 0.8, 
            'Very_Active': 0.0
        },
        'very active': {
            'Sedentary': 0.0, 
            'Light': 0.0, 
            'Moderate': 0.0, 
            'Active': 0.2, 
            'Very_Active': 0.8
        }
    }
    return mapping.get(exercise_rate.lower(), {'Sedentary': 0.0, 'Light': 0.0, 'Moderate': 1.0, 'Active': 0.0, 'Very_Active': 0.0})

def age_membership(age):
    """
    Returns degrees of membership [0–1] in six overlapping age groups that align 
    with metabolic and nutritional research:
    
      - Youth       (18–24) - Higher metabolism, growth phase completion
      - Young Adult (25–35) - Peak physical condition
      - Adult       (36–50) - Stable metabolism
      - Middle-Aged (51–65) - Metabolic slowdown begins
      - Senior      (66–75) - Significant metabolic changes
      - Elderly     (76+)   - Decreased caloric needs
      
    Based on research data for nutritional requirements across age groups.
    """
    mem = {}

    # Youth (18-24): Trapezoid with plateau from 19-22
    if age < 18: mem['Youth'] = 0
    elif age < 19: mem['Youth'] = (age - 18)
    elif age <= 22: mem['Youth'] = 1
    elif age < 25: mem['Youth'] = (24 - age) / 2
    else: mem['Youth'] = 0

    # Young Adult (25-35): Triangle with peak at 30
    if age < 25: mem['Young_Adult'] = 0
    elif age < 30: mem['Young_Adult'] = (age - 25) / 5
    elif age < 36: mem['Young_Adult'] = (35 - age) / 5
    else: mem['Young_Adult'] = 0

    # Adult (36-50): Trapezoid with plateau from 40-46
    if age < 36: mem['Adult'] = 0
    elif age < 40: mem['Adult'] = (age - 36) / 4
    elif age <= 46: mem['Adult'] = 1
    elif age < 51: mem['Adult'] = (50 - age) / 4
    else: mem['Adult'] = 0

    # Middle-Aged (51-65): Trapezoid with plateau from 55-61
    if age < 51: mem['Middle_Aged'] = 0
    elif age < 55: mem['Middle_Aged'] = (age - 51) / 4
    elif age <= 61: mem['Middle_Aged'] = 1
    elif age < 66: mem['Middle_Aged'] = (65 - age) / 4
    else: mem['Middle_Aged'] = 0

    # Senior (66-75): Triangle with peak at 70
    if age < 66: mem['Senior'] = 0
    elif age < 70: mem['Senior'] = (age - 66) / 4
    elif age < 76: mem['Senior'] = (75 - age) / 5
    else: mem['Senior'] = 0

    # Elderly (76+): Rising curve with plateau at 85+
    if age < 76: mem['Elderly'] = 0
    elif age < 85: mem['Elderly'] = (age - 76) / 9
    else: mem['Elderly'] = 1

    return mem


# References:
# - Advanced Nutrition and Dietetics Research
# - https://www.mdpi.com/2072-6643/16/21/3637 
# - Metabolic adaptation across age groups and activity levels

def calculate_weight_factor(bmi, gender):
    """Calculate a weight factor based on BMI and gender reference weights"""
    if gender.lower() == 'male':
        ref_weight = reference_weight['male']
        if bmi < 18.5:  # Underweight
            return weight_adjustment['light']
        elif bmi < 25:  # Normal
            return weight_adjustment['medium']
        elif bmi < 30:  # Overweight
            return weight_adjustment['heavy']
        else:  # Obese
            return weight_adjustment['very_heavy']
    else:  # Female
        ref_weight = reference_weight['female']
        if bmi < 18.5:  # Underweight
            return weight_adjustment['light'] * 0.95  # Women typically need fewer calories
        elif bmi < 25:  # Normal
            return weight_adjustment['medium'] * 0.95
        elif bmi < 30:  # Overweight
            return weight_adjustment['heavy'] * 0.95
        else:  # Obese
            return weight_adjustment['very_heavy'] * 0.95

def fuzzy_calorie_adjustment(bmi, exercise_rate, age):
    """
    Calculate a personalized calorie adjustment factor using fuzzy logic.
    
    This refined implementation uses the detailed data from the research table
    and incorporates more nuanced rules for different combinations of:
    - BMI categories (7 levels from severely underweight to obese class III)
    - Activity levels (5 levels from sedentary to very active)
    - Age groups (6 groups from youth to elderly)
    
    Returns:
        float: Adjustment factor to multiply with base caloric needs
    """
    # Get membership degrees for all input variables
    bmi_mem = bmi_membership(bmi)
    ex_mem = exercise_membership(exercise_rate)
    age_mem = age_membership(age)

    # Enhanced rule base with more comprehensive coverage
    rules = [
        # ----- SEVERELY UNDERWEIGHT CASES ----- #
        # Need significant calorie increases regardless of activity level
        (bmi_mem['Severely_Underweight'] * ex_mem['Sedentary']  * age_mem['Youth'],        1.30),
        (bmi_mem['Severely_Underweight'] * ex_mem['Sedentary']  * age_mem['Young_Adult'],  1.25),
        (bmi_mem['Severely_Underweight'] * ex_mem['Sedentary']  * age_mem['Adult'],        1.20),
        (bmi_mem['Severely_Underweight'] * ex_mem['Sedentary']  * age_mem['Middle_Aged'],  1.15),
        (bmi_mem['Severely_Underweight'] * ex_mem['Sedentary']  * age_mem['Senior'],       1.10),
        (bmi_mem['Severely_Underweight'] * ex_mem['Light']      * age_mem['Youth'],        1.35),
        (bmi_mem['Severely_Underweight'] * ex_mem['Moderate']   * age_mem['Youth'],        1.40),
        (bmi_mem['Severely_Underweight'] * ex_mem['Active']     * age_mem['Youth'],        1.45),
        (bmi_mem['Severely_Underweight'] * ex_mem['Very_Active']* age_mem['Youth'],        1.50),
        
        # ----- UNDERWEIGHT CASES ----- #
        # Moderate calorie increases
        (bmi_mem['Underweight'] * ex_mem['Sedentary']  * age_mem['Youth'],        1.20),
        (bmi_mem['Underweight'] * ex_mem['Sedentary']  * age_mem['Young_Adult'],  1.15),
        (bmi_mem['Underweight'] * ex_mem['Sedentary']  * age_mem['Adult'],        1.10),
        (bmi_mem['Underweight'] * ex_mem['Sedentary']  * age_mem['Middle_Aged'],  1.05),
        (bmi_mem['Underweight'] * ex_mem['Sedentary']  * age_mem['Senior'],       1.05),
        (bmi_mem['Underweight'] * ex_mem['Light']      * age_mem['Youth'],        1.25),
        (bmi_mem['Underweight'] * ex_mem['Moderate']   * age_mem['Youth'],        1.30),
        (bmi_mem['Underweight'] * ex_mem['Active']     * age_mem['Youth'],        1.35),
        
        # ----- NORMAL WEIGHT CASES ----- #
        # Maintenance or slight adjustments based on age and activity
        (bmi_mem['Normal'] * ex_mem['Sedentary']   * age_mem['Youth'],        0.95),
        (bmi_mem['Normal'] * ex_mem['Sedentary']   * age_mem['Young_Adult'],  0.90),
        (bmi_mem['Normal'] * ex_mem['Sedentary']   * age_mem['Adult'],        0.90),
        (bmi_mem['Normal'] * ex_mem['Sedentary']   * age_mem['Middle_Aged'],  0.85),
        (bmi_mem['Normal'] * ex_mem['Sedentary']   * age_mem['Senior'],       0.85),
        (bmi_mem['Normal'] * ex_mem['Light']       * age_mem['Youth'],        1.00),
        (bmi_mem['Normal'] * ex_mem['Light']       * age_mem['Young_Adult'],  0.95),
        (bmi_mem['Normal'] * ex_mem['Light']       * age_mem['Adult'],        0.95),
        (bmi_mem['Normal'] * ex_mem['Light']       * age_mem['Middle_Aged'],  0.90),
        (bmi_mem['Normal'] * ex_mem['Moderate']    * age_mem['Youth'],        1.05),
        (bmi_mem['Normal'] * ex_mem['Moderate']    * age_mem['Young_Adult'],  1.00),
        (bmi_mem['Normal'] * ex_mem['Moderate']    * age_mem['Adult'],        1.00),
        (bmi_mem['Normal'] * ex_mem['Active']      * age_mem['Youth'],        1.10),
        (bmi_mem['Normal'] * ex_mem['Active']      * age_mem['Young_Adult'],  1.05),
        (bmi_mem['Normal'] * ex_mem['Very_Active'] * age_mem['Youth'],        1.15),
        
        # ----- OVERWEIGHT CASES ----- #
        # Slight to moderate reductions
        (bmi_mem['Overweight'] * ex_mem['Sedentary']   * age_mem['Youth'],        0.85),
        (bmi_mem['Overweight'] * ex_mem['Sedentary']   * age_mem['Young_Adult'],  0.80),
        (bmi_mem['Overweight'] * ex_mem['Sedentary']   * age_mem['Adult'],        0.80),
        (bmi_mem['Overweight'] * ex_mem['Sedentary']   * age_mem['Middle_Aged'],  0.75),
        (bmi_mem['Overweight'] * ex_mem['Sedentary']   * age_mem['Senior'],       0.75),
        (bmi_mem['Overweight'] * ex_mem['Light']       * age_mem['Youth'],        0.90),
        (bmi_mem['Overweight'] * ex_mem['Light']       * age_mem['Young_Adult'],  0.85),
        (bmi_mem['Overweight'] * ex_mem['Light']       * age_mem['Adult'],        0.85),
        (bmi_mem['Overweight'] * ex_mem['Moderate']    * age_mem['Youth'],        0.95),
        (bmi_mem['Overweight'] * ex_mem['Moderate']    * age_mem['Young_Adult'],  0.90),
        (bmi_mem['Overweight'] * ex_mem['Active']      * age_mem['Youth'],        1.00),
        
        # ----- OBESE CLASS I CASES ----- #
        # Moderate reductions
        (bmi_mem['Obese_I'] * ex_mem['Sedentary']   * age_mem['Youth'],        0.75),
        (bmi_mem['Obese_I'] * ex_mem['Sedentary']   * age_mem['Young_Adult'],  0.70),
        (bmi_mem['Obese_I'] * ex_mem['Sedentary']   * age_mem['Adult'],        0.70),
        (bmi_mem['Obese_I'] * ex_mem['Sedentary']   * age_mem['Middle_Aged'],  0.65),
        (bmi_mem['Obese_I'] * ex_mem['Sedentary']   * age_mem['Senior'],       0.65),
        (bmi_mem['Obese_I'] * ex_mem['Light']       * age_mem['Youth'],        0.80),
        (bmi_mem['Obese_I'] * ex_mem['Moderate']    * age_mem['Youth'],        0.85),
        
        # ----- OBESE CLASS II CASES ----- #
        # More significant reductions
        (bmi_mem['Obese_II'] * ex_mem['Sedentary']   * age_mem['Youth'],        0.70),
        (bmi_mem['Obese_II'] * ex_mem['Sedentary']   * age_mem['Young_Adult'],  0.65),
        (bmi_mem['Obese_II'] * ex_mem['Light']       * age_mem['Youth'],        0.75),
        
        # ----- OBESE CLASS III CASES ----- #
        # Most significant reductions with medical supervision implied
        (bmi_mem['Obese_III'] * ex_mem['Sedentary']   * age_mem['Youth'],        0.65),
        (bmi_mem['Obese_III'] * ex_mem['Sedentary']   * age_mem['Young_Adult'],  0.60),
        (bmi_mem['Obese_III'] * ex_mem['Sedentary']   * age_mem['Adult'],        0.60),
    ]
    
    # Calculate the weighted average of all rule outputs
    num = sum(w * out for w, out in rules)
    den = sum(w for w, out in rules)
    
    # Default to 1.0 if no rules fire significantly
    if den < 0.1:  # Minimal rule activation
        return 1.0
    
    # Return the defuzzified result, rounded to 2 decimal places
    adjustment = round(num / den, 2)
    
    # Safety limits: never go below 0.6 or above 1.5
    return max(0.6, min(1.5, adjustment)), bmi_mem, ex_mem, age_mem

