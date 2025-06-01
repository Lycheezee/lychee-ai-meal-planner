import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class SimilarMealPlanService:
    def __init__(self):
        # Load data and model
        self.df = pd.read_csv("dataset/daily_food_nutrition_dataset_with_ids.csv")
        self.features = ["Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fat (g)", "Fiber (g)", "Sugars (g)", "Sodium (mg)", "Cholesterol (mg)"]
        X = self.df[self.features]

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        self.knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
        self.knn.fit(self.X_scaled)

        # Load start foods from first meal plan
        try:
            firstMealDf = pd.read_csv("results/first_meal_plan.csv")
            self.start_foods = firstMealDf["Food_Item"].dropna().tolist()
        except FileNotFoundError:
            self.start_foods = []

    def generate_meal_plan(self, start_food, days=30):
        meal_plan = [start_food]
        current_food = start_food
        visited = set()
        visited.add(current_food.lower())

        for _ in range(days - 1):
            idx = self.df[self.df["Food_Item"].str.lower() == current_food.lower()].index
            if len(idx) == 0:
                meal_plan.append("Not Found")
                break
            
            idx = idx[0]
            distances, indices = self.knn.kneighbors([self.X_scaled[idx]])
            
            found = False
            for neighbor_idx in indices[0][1:]:
                neighbor_food = self.df.iloc[neighbor_idx]['Food_Item']
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

    def generate_similar_meal_plans(self, days=30):
        all_meal_plans = {}

        for food in self.start_foods:
            plan = self.generate_meal_plan(food, days)
            all_meal_plans[food] = plan

        meal_plans_df = pd.DataFrame(all_meal_plans).T
        meal_plans_df.columns = [f"Day {i+1}" for i in range(days)]

        return meal_plans_df.to_dict(orient="index")

# Create singleton instance
similar_meal_service = SimilarMealPlanService()
