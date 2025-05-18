from constant import gender_adjusment

def bmi_membership(bmi):
    return {
        'Underweight': max(0, min(1, (18.5 - bmi) / 3)),
        'Normal': max(0, 1 - abs(bmi - 22) / 3.5),
        'Overweight': max(0, 1 - abs(bmi - 27) / 3),
        'Obese': max(0, min(1, (bmi - 30) / 5))
    }

def exercise_membership(exercise_rate):
    mapping = {
        'sedentary': {'Sedentary': 1},
        'light': {'Light': 1},
        'moderate': {'Moderate': 1},
        'active': {'Active': 1},
        'very active': {'Very Active': 1}
    }
    return mapping.get(exercise_rate.lower(), {'Sedentary': 1})

def age_membership(age):
    """
    Returns degrees of membership [0–1] in six overlapping age groups:
      - Youth       (15–24)
      - Young Adult (20–30)
      - Adult       (30–45)
      - Middle-Aged (45–60)
      - Senior      (60–75)
      - Elderly     (75+)
    Ranges based on StatsCan (15–24 youth, 25–64 adults, 65+ seniors) :contentReference[oaicite:3]{index=3}
    and Integris Health (20–39 Adult, 40–59 Middle, 60+ Senior) :contentReference[oaicite:4]{index=4}.
    """
    mem = {}

    # Youth: trapezoid rising from 15 to plateau at 18–22, falling to 24
    if age < 15: mem['Youth'] = 0
    elif age <= 18: mem['Youth'] = (age - 15) / (18 - 15)
    elif age <= 22: mem['Youth'] = 1
    elif age <= 24: mem['Youth'] = (24 - age) / (24 - 22)
    else: mem['Youth'] = 0

    # Young Adult: triangle 20–30, peak at 25
    mem['Young Adult'] = max(0, min((age - 20) / 5, (30 - age) / 5))

    # Adult: triangle 30–45, peak at 37.5
    mem['Adult'] = max(0, min((age - 30) / 7.5, (45 - age) / 7.5))

    # Middle-Aged: triangle 45–60, peak at 52.5
    mem['Middle-Aged'] = max(0, min((age - 45) / 7.5, (60 - age) / 7.5))

    # Senior: triangle 60–75, peak at 67.5
    mem['Senior'] = max(0, min((age - 60) / 7.5, (75 - age) / 7.5))

    # Elderly: trapezoid rising from 75, plateau from 80 onwards
    if age < 75: mem['Elderly'] = 0
    elif age <= 80: mem['Elderly'] = (age - 75) / (80 - 75)
    else: mem['Elderly'] = 1

    return mem


# Citeria
# https://www.mdpi.com/2072-6643/16/21/3637#:~:text=environment%20and%20avoid%20weight%20gain,items%20that%20generate%20greater%20satiation
# https://chatgpt.com/share/6825e7ab-3054-8001-8d8d-f977d42ec220

def fuzzy_calorie_adjustment(bmi, exercise_rate, age):
    bmi_mem = bmi_membership(bmi)
    ex_mem  = exercise_membership(exercise_rate)
    age_mem = age_membership(age)

    rules =rules = [
        # Strong reductions for high risk
        (bmi_mem['Obese']       * ex_mem.get('Sedentary',0)  * age_mem['Middle-Aged'], 0.70),
        (bmi_mem['Obese']       * ex_mem.get('Sedentary',0)  * age_mem['Senior'],      0.75),
        (bmi_mem['Obese']       * ex_mem.get('Light',0)      * age_mem['Senior'],      0.78),
        (bmi_mem['Obese']       * ex_mem.get('Light',0)      * age_mem['Adult'],       0.80),
        (bmi_mem['Obese']       * ex_mem.get('Moderate',0)   * age_mem['Adult'],       0.90),
        (bmi_mem['Obese']       * ex_mem.get('Active',0)     * age_mem['Elderly'],     0.85),
        (bmi_mem['Overweight']  * ex_mem.get('Sedentary',0)  * age_mem['Middle-Aged'], 0.85),
        (bmi_mem['Overweight']  * ex_mem.get('Moderate',0)   * age_mem['Senior'],      0.88),
        (bmi_mem['Overweight']  * ex_mem.get('Light',0)      * age_mem['Youth'],       0.95),
        (bmi_mem['Normal']      * ex_mem.get('Sedentary',0)  * age_mem['Adult'],       0.95),
        (bmi_mem['Normal']      * ex_mem.get('Light',0)      * age_mem['Elderly'],     0.92),
        (bmi_mem['Overweight']  * ex_mem.get('Active',0)     * age_mem['Elderly'],     0.92),

        # Near maintenance values
        (bmi_mem['Normal']      * ex_mem.get('Sedentary',0)  * age_mem['Elderly'],     0.90),
        (bmi_mem['Normal']      * ex_mem.get('Light',0)      * age_mem['Middle-Aged'], 0.98),
        (bmi_mem['Normal']      * ex_mem.get('Moderate',0)   * age_mem['Adult'],       1.00),
        (bmi_mem['Normal']      * ex_mem.get('Moderate',0)   * age_mem['Young Adult'], 1.00),
        (bmi_mem['Overweight']  * ex_mem.get('Active',0)     * age_mem['Adult'],       1.00),

        # Slight increase scenarios
        (bmi_mem['Normal']      * ex_mem.get('Active',0)     * age_mem['Senior'],      1.05),
        (bmi_mem['Normal']      * ex_mem.get('Active',0)     * age_mem['Youth'],       1.05),
        (bmi_mem['Underweight'] * ex_mem.get('Sedentary',0)  * age_mem['Elderly'],     1.10),
        (bmi_mem['Underweight'] * ex_mem.get('Moderate',0)   * age_mem['Senior'],      1.10),
        (bmi_mem['Underweight'] * ex_mem.get('Active',0)     * age_mem['Elderly'],     1.10),

        # Moderate boosts
        (bmi_mem['Underweight'] * ex_mem.get('Sedentary',0)  * age_mem['Youth'],       1.15),
        (bmi_mem['Underweight'] * ex_mem.get('Moderate',0)   * age_mem['Young Adult'], 1.20),

        # Higher boosts for younger and more active
        (bmi_mem['Underweight'] * ex_mem.get('Active',0)     * age_mem['Youth'],       1.25),
    ]


    num = sum(w * out for w, out in rules)
    den = sum(w for w, out in rules)
    return round(num / den, 2) if den else 1.0

