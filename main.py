from fastapi import FastAPI
from routers import meal_planner, similar_meal_planner

app = FastAPI(
    title="Meal Planner API",
    description="API for generating personalized meal plans",
    version="1.0.0"
)

# Include routers
app.include_router(meal_planner.router)
app.include_router(similar_meal_planner.router)

@app.get("/")
def read_root():
    return {"message": "Meal Planner API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# uvicorn main:app --reload --host 0.0.0.0 --port 8000