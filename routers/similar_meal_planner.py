from fastapi import APIRouter
from pydantic import BaseModel
from services.similar_meal_service import similar_meal_service

router = APIRouter(prefix="/api", tags=["similar-meal-planner"])

class DurationRequest(BaseModel):
    days: int = 30

@router.post("/similar-meal-plan")
def similar_meal_plan(request: DurationRequest):
    print(f"Generating similar meal plans for {request.days} days")
    plans = similar_meal_service.generate_similar_meal_plans(request.days)
    print(f"Generated {len(plans)} similar meal plans")
    return {"plans": plans}
