from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from services.meal_planner_service import generate_meal_plan_api

router = APIRouter(prefix="/api", tags=["meal-planner"])

class MealRequest(BaseModel):
    height: float
    weight: float
    gender: str
    exercise_rate: str
    dob: str
    macro_preference: str
    use_deap: bool = True  # Default to using DEAP algorithm

@router.post("/meal-plan")
def create_meal_plan(data: MealRequest):
    print(f"Generating meal plan for: {data.gender}, height: {data.height}, weight: {data.weight}")
    print(f"Using DEAP algorithm: {data.use_deap}")
    plan, daily_targets = generate_meal_plan_api(
        height=data.height,
        weight=data.weight,
        gender=data.gender,
        exercise_rate=data.exercise_rate,
        dob=data.dob,
        macro_preference=data.macro_preference,
        use_deap=data.use_deap
    )
    print(f"Generated meal plan with {len(plan)} items")
    encoded_plan = jsonable_encoder(plan)
    encoded_targets = jsonable_encoder(daily_targets)
    return {"meal_plan": encoded_plan, "daily_targets": encoded_targets}
