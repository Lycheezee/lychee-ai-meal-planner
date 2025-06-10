"""
DEAP-based meal plan generator using genetic algorithms.
Integrates with the existing meal planning system.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from deap import base, creator, tools
import warnings
warnings.filterwarnings('ignore')

class DEAPMealGenerator:
    """
    Genetic Algorithm-based meal plan generator using DEAP.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DEAP meal generator.
        
        Args:
            df: Food database DataFrame
        """
        self.df = df.copy()
        self.n_items = len(df)
        
        # Map column names to match expected format
        self.nutrition_columns = {
            'calories': 'calories',
            'proteins': 'proteins', 
            'carbohydrates': 'carbohydrates',
            'fats': 'fats',
            'fibers': 'fibers',
            'sugars': 'sugars',
            'sodium': 'sodium',
            'cholesterol': 'cholesterol'
        }
        
        # Verify required columns exist
        missing_cols = [col for col in self.nutrition_columns.values() if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing nutrition columns in dataset: {missing_cols}")
          # Set up DEAP framework
        self._setup_deap()
    
    def _setup_deap(self):
        """Set up DEAP genetic algorithm framework."""
        
        # Clear any existing creators
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        # Create fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize error
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Register genetic operators with bias toward fewer selections
        self.toolbox.register("attr_bool", self._biased_selection)  # Custom selection
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.attr_bool, n=self.n_items)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._custom_mutation)  # Custom mutation
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _biased_selection(self) -> int:
        """Biased selection that favors 0 (not selected) over 1 (selected)."""
        return 1 if random.random() < 0.01 else 0
    
    def _custom_mutation(self, individual, indpb=0.05):
        """
        Custom mutation that aggressively maintains small food counts.
        """
        current_count = sum(individual)

        if current_count > 8:
            selected_indices = [i for i, val in enumerate(individual) if val == 1]
            num_to_remove = current_count - 8
            if num_to_remove > 0:
                indices_to_remove = random.sample(selected_indices, min(num_to_remove, len(selected_indices)))
                for idx in indices_to_remove:
                    individual[idx] = 0

        for i in range(len(individual)):
            if random.random() < indpb:
                current_count = sum(individual)
                
                if individual[i] == 1:
                    if current_count > 6 or random.random() < 0.8:
                        individual[i] = 0
                else:
                    if current_count < 5 and random.random() < 0.1:
                        individual[i] = 1
        
        return individual,
    
    def _evaluate_nutrition(self, individual: List[int], targets: Dict[str, float]) -> Tuple[float]:
        """
        Evaluate fitness of an individual (food selection).
        
        Args:
            individual: Binary list indicating which foods are selected
            targets: Target nutritional values
            
        Returns:
            Fitness score (lower is better)
        """
        # Count selected foods
        num_selected = sum(individual)
        
        # Extremely harsh penalties for too many foods
        if num_selected == 0:
            return (1e10,)
        elif num_selected > 12:  # Catastrophic penalty
            return (1e9 + num_selected * 100000,)
        elif num_selected > 8:   # Very high penalty
            return (1e8 + (num_selected - 8) * 50000,)
        elif num_selected < 3:   # Too few foods
            return (1e7,)
        
        # Get nutrition matrix for selected foods
        selected_foods = self.df[np.array(individual, dtype=bool)]
        
        # Calculate total nutrition
        totals = {}
        for nutrient in self.nutrition_columns.values():
            if nutrient in selected_foods.columns:
                totals[nutrient] = selected_foods[nutrient].fillna(0).sum()
            else:
                totals[nutrient] = 0
        
        # Calculate weighted error with strong penalties for overshooting
        error = 0
        critical_nutrients = ['calories', 'carbohydrates', 'fats']
        
        for nutrient, target in targets.items():
            if target > 0 and nutrient in totals:
                actual = totals[nutrient]
                percentage = (actual / target) * 100
                
                if nutrient in critical_nutrients:
                    # Very strict penalties for critical nutrients
                    if 90 <= percentage <= 110:
                        error += 0
                    elif 80 <= percentage <= 120:
                        error += abs(percentage - 100) * 10  # Acceptable range
                    else:
                        error += abs(percentage - 100) * 50
                        if percentage > 120:
                            error += (percentage - 120) * 100
                else:
                    if 80 <= percentage <= 120:
                        error += abs(percentage - 100) * 0.1
                    else:
                        error += abs(percentage - 100) * 25
        
        # Strong penalty for selecting too many foods
        if num_selected > 10:
            error += (num_selected - 10) * 500  # Heavy penalty for excess foods
        elif num_selected > 8:
            error += (num_selected - 8) * 100   # Moderate penalty
        elif num_selected < 5:
            error += (5 - num_selected) * 50    # Penalty for too few foods
        
        return (error,)
    
    def generate_meal_plan(self, daily_targets: Dict[str, float], 
                          pop_size: int = 176, n_gen: int = 282,
                          cxpb: float = 0.862, mutpb: float = 0.462) -> pd.DataFrame:
        """
        Generate optimized meal plan using genetic algorithm.
        
        Args:
            daily_targets: Target nutritional values
            pop_size: Population size
            n_gen: Number of generations
            cxpb: Crossover probability
            mutpb: Mutation probability
            
        Returns:
            DataFrame containing selected foods
        """
        print(f"\n🧬 DEAP GENETIC ALGORITHM MEAL PLANNING")
        print(f"Population: {pop_size}, Generations: {n_gen}")
        print(f"Dataset: {self.n_items} foods available")
        print(f"Targets: {daily_targets}")
        
        # Register evaluation function with current targets
        self.toolbox.register("evaluate", self._evaluate_nutrition, targets=daily_targets)
        
        # Initialize population
        population = self.toolbox.population(n=pop_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        print(f"Initial best fitness: {min(fitnesses)[0]:.4f}")
        
        # Evolution loop
        for generation in range(n_gen):
            # Select offspring
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Print progress every 10 generations
            if generation % 10 == 0:
                best_fitness = min(ind.fitness.values[0] for ind in population)
                print(f"Generation {generation:2d}: Best fitness = {best_fitness:.4f}")
          # Get best individual
        best_individual = tools.selBest(population, 1)[0]
        best_fitness = best_individual.fitness.values[0]
        
        # Post-process: ensure reasonable number of foods
        num_selected = sum(best_individual)
        if num_selected > 15:
            print(f"⚠️  Too many foods selected ({num_selected}), applying post-processing...")
            # Keep only the top foods based on their individual contribution
            selected_indices = [i for i, selected in enumerate(best_individual) if selected]
            food_scores = []
            
            for idx in selected_indices:
                food = self.df.iloc[idx]
                score = 0
                for nutrient, target in daily_targets.items():
                    if nutrient in food and target > 0:
                        contribution = food[nutrient] / target if not pd.isna(food[nutrient]) else 0
                        score += min(contribution, 1.0)  # Cap contribution at 100%
                food_scores.append((idx, score))
            
            # Keep top 10 foods
            food_scores.sort(key=lambda x: x[1], reverse=True)
            best_foods = [idx for idx, _ in food_scores[:10]]
            
            # Create new individual with only top foods
            new_individual = [0] * len(best_individual)
            for idx in best_foods:
                new_individual[idx] = 1
            
            best_individual = new_individual
            num_selected = sum(best_individual)
        
        print(f"\n🏆 OPTIMIZATION COMPLETE")
        print(f"Final best fitness: {best_fitness:.4f}")
        print(f"Foods selected: {num_selected}")
        
        # Extract selected foods
        selected_foods = self.df[np.array(best_individual, dtype=bool)].copy()
        
        if len(selected_foods) == 0:
            print("❌ No foods selected by genetic algorithm!")
            return pd.DataFrame()
        
        # Display selected foods
        print(f"\n📋 SELECTED FOODS:")
        for i, (_, food) in enumerate(selected_foods.iterrows(), 1):
            name = food.get('food_item', 'Unknown')
            calories = food.get('calories', 0)
            print(f"{i:2d}. {name}: {calories:.1f} cal")
        
        return selected_foods
    
    def evaluate_meal_plan(self, meal_plan: pd.DataFrame, daily_targets: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate how well the meal plan meets targets.
        
        Args:
            meal_plan: Selected foods DataFrame
            daily_targets: Target nutritional values
            
        Returns:
            Dictionary with nutrition totals and percentages
        """
        if meal_plan.empty:
            return {}
        
        results = {}
        critical_nutrients = ['calories', 'carbohydrates', 'fats']
        
        print(f"\n📊 DEAP MEAL PLAN EVALUATION")
        print("-" * 50)
        
        for nutrient in self.nutrition_columns.values():
            if nutrient in meal_plan.columns and nutrient in daily_targets:
                total = meal_plan[nutrient].fillna(0).sum()
                target = daily_targets[nutrient]
                percentage = (total / target * 100) if target > 0 else 0
                
                # Status determination
                if nutrient in critical_nutrients:
                    if 95 <= percentage <= 105:
                        status = "🎯 Perfect"
                    elif 90 <= percentage <= 110:
                        status = "✓ Good"
                    else:
                        status = "❌ Poor"
                else:
                    if 80 <= percentage <= 120:
                        status = "✓ Good"
                    else:
                        status = "⚠️ Outside range"
                
                results[nutrient] = {
                    'total': total,
                    'target': target,
                    'percentage': percentage,
                    'status': status
                }
                
                print(f"{nutrient.capitalize():15}: {total:8.1f} / {target:8.1f} ({percentage:6.1f}%) {status}")
        
        return results


def generate_deap_meal_plan(df: pd.DataFrame, daily_targets: Dict[str, float]) -> pd.DataFrame:
    """
    Convenience function to generate meal plan using DEAP algorithm.
    
    Args:
        df: Food database DataFrame
        daily_targets: Target nutritional values
        
    Returns:
        DataFrame containing optimized food selection
    """
    try:
        generator = DEAPMealGenerator(df)
        meal_plan = generator.generate_meal_plan(daily_targets)
        
        if not meal_plan.empty:
            generator.evaluate_meal_plan(meal_plan, daily_targets)
        
        return meal_plan
        
    except Exception as e:
        print(f"❌ DEAP meal generation failed: {e}")
        return pd.DataFrame()
