from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from routes.meal_planner import generate_meal_plan_api
from routes.similar_meal_plan import router as similar_meal_plan_router

app = FastAPI()

class MealRequest(BaseModel):
    height: float
    weight: float
    gender: str
    exercise_rate: str
    dob: str
    macro_preference: str

@app.post("/api/meal-plan")
def create_meal_plan(data: MealRequest):
    print(data)
    plan = generate_meal_plan_api(
        height=data.height,
        weight=data.weight,
        gender=data.gender,
        exercise_rate=data.exercise_rate,
        dob=data.dob,
        macro_preference=data.macro_preference
    )
    print(plan)
    encoded_plan = jsonable_encoder(plan)
    return {"meal_plan": encoded_plan}

app.include_router(similar_meal_plan_router)