import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Load data and model globally
df = pd.read_csv("dataset/daily_food_nutrition_dataset.csv")
features = ["Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fat (g)", "Fiber (g)", "Sugars (g)", "Sodium (mg)", "Cholesterol (mg)"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn.fit(X_scaled)

firstMealDf = pd.read_csv("results/first_meal_plan.csv")
start_foods = firstMealDf["Food_Item"].dropna().tolist()

class DurationRequest(BaseModel):
    days: int = 30

def generate_meal_plan(start_food, days=30):
    meal_plan = [start_food]
    current_food = start_food
    visited = set()
    visited.add(current_food.lower())

    for _ in range(days - 1):
        idx = df[df["Food_Item"].str.lower() == current_food.lower()].index
        if len(idx) == 0:
            meal_plan.append("Not Found")
            break
        
        idx = idx[0]
        distances, indices = knn.kneighbors([X_scaled[idx]])
        
        found = False
        for neighbor_idx in indices[0][1:]:
            neighbor_food = df.iloc[neighbor_idx]['Food_Item']
            if neighbor_food.lower() not in visited:
                meal_plan.append(neighbor_food)
                current_food = neighbor_food
                visited.add(neighbor_food.lower())
                found = True
                break
        
        if not found:
            meal_plan.append(current_food)
    
    while len(meal_plan) < days:
        meal_plan.append(meal_plan[-1])
    
    return meal_plan

@router.post("/api/similar-meal-plan")
def similar_meal_plan(request: DurationRequest):
    days = request.days
    all_meal_plans = {}

    for food in start_foods:
        plan = generate_meal_plan(food, days)
        all_meal_plans[food] = plan

    meal_plans_df = pd.DataFrame(all_meal_plans).T
    meal_plans_df.columns = [f"Day {i+1}" for i in range(days)]

    return {"plans": meal_plans_df.to_dict(orient="index")}
