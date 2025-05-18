from datetime import datetime

def calculate_age(dob_input):
    dob = datetime.strptime(dob_input, "%Y-%m-%d")
    today = datetime.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    
    return age