"""
Macro preference adjustments for meal planning.
Handles adjusting nutritional targets based on user's macro preferences.
"""

import sys
import os
from typing import Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constant import macro_preferences

def adjust_targets_for_macro(targets: Dict[str, float], preference: str) -> Dict[str, float]:
    """
    Adjust nutritional targets based on macro preferences.
    
    Args:
        targets: Base nutritional targets dictionary
        preference: Macro preference (e.g., 'balanced', 'high_protein', 'low_carb')
        
    Returns:
        Dict with adjusted nutritional targets
    """
    pref = preference.lower()
    if pref not in macro_preferences:
        print(f"Warning: Preference '{preference}' not found. Using 'balanced'.")
        pref = 'balanced'

    adjusted = targets.copy()
    
    # Adjust macros only for Protein, Carbs, and Fat
    for macro in ["proteins", "carbohydrates", "fats"]:
        if macro in adjusted:
            adjusted[macro] = round(adjusted[macro] * macro_preferences[pref].get(macro, 1.0), 1)

    # Optional: Adjust calories roughly to match macro changes 
    # (assuming calories come mainly from these macros)
    # 1g Protein = 4 kcal, 1g Carb = 4 kcal, 1g Fat = 9 kcal
    calories = (
        adjusted.get("proteins", 0) * 4 +
        adjusted.get("carbohydrates", 0) * 4 +
        adjusted.get("fats", 0) * 9
    )
    adjusted["calories"] = round(calories)

    return adjusted

def get_available_macro_preferences() -> list:
    """Get list of available macro preferences."""
    return list(macro_preferences.keys())

def get_macro_preference_details(preference: str) -> Dict[str, float]:
    """
    Get detailed information about a specific macro preference.
    
    Args:
        preference: The macro preference name
        
    Returns:
        Dict with macro multipliers for the preference
    """
    pref = preference.lower()
    if pref in macro_preferences:
        return macro_preferences[pref].copy()
    else:
        print(f"Warning: Preference '{preference}' not found. Returning 'balanced'.")
        return macro_preferences.get('balanced', {
            'proteins': 1.0,
            'carbohydrates': 1.0,
            'fats': 1.0
        })

def validate_macro_preference(preference: str) -> bool:
    """Check if a macro preference is valid."""
    return preference.lower() in macro_preferences
