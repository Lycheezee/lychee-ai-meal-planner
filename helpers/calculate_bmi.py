def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)