import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class SimilarMealPlanService:
    def __init__(self):
        # Load data and model using cleaned dataset
        self.df = pd.read_csv("dataset/daily_food_nutrition_dataset_cleaned.csv")
        self.features = ["calories", "proteins", "carbohydrates", "fats", "fibers", "sugars", "sodium", "cholesterol"]
        X = self.df[self.features]

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        self.knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
        self.knn.fit(self.X_scaled)

        # Load start foods from first meal plan
        try:
            firstMealDf = pd.read_csv("results/first_meal_plan.csv")
            self.start_foods = firstMealDf["food_item"].dropna().tolist()
        except FileNotFoundError:
            self.start_foods = []

    def generate_meal_plan(self, start_food, days=30):
        meal_plan = [start_food]
        current_food = start_food
        visited = set()
        visited.add(current_food.lower())

        for _ in range(days - 1):
            idx = self.df[self.df["food_item"].str.lower() == current_food.lower()].index
            if len(idx) == 0:
                meal_plan.append("Not Found")
                break
            
            idx = idx[0]
            distances, indices = self.knn.kneighbors([self.X_scaled[idx]])
            
            found = False
            for neighbor_idx in indices[0][1:]:
                neighbor_food = self.df.iloc[neighbor_idx]['food_item']
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

    def generate_similar_meal_plans(self, days=30, start_foods=[]):
        meal_plans = []
        
        # Generate dates starting from today
        start_date = datetime.now()
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Create meals for this day using all start foods
            meals = []
            for food in start_foods:
                # Get the food for this specific day from the meal plan
                food_plan = self.generate_meal_plan(food, days)
                if day < len(food_plan):
                    current_food = food_plan[day]
                    meals.append({
                        "foodId": current_food,
                        "status": "not_completed"
                    })
            
            # Create daily plan
            daily_plan = {
                "date": current_date.isoformat(),
                "meals": meals
            }
            
            meal_plans.append(daily_plan)
        
        return meal_plans

# Create singleton instance
similar_meal_service = SimilarMealPlanService()
