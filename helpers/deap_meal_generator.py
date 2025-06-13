"""
DEAP-based meal plan generator using genetic algorithms.
Integrates with the existing meal planning system.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from deap.algorithms import eaMuPlusLambda
from deap import base, creator, tools
import warnings
import multiprocessing
warnings.filterwarnings('ignore')

class DEAPMealGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.n_items = len(df)

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
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize error
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        self.toolbox.register("attr_bool", self._biased_selection)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.attr_bool, n=self.n_items)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._custom_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _biased_selection(self) -> int:
        return 1 if random.random() < 0.01 else 0
    
    def _custom_mutation(self, individual, indpb=0.05):
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
    
    def _custom_eaMuPlusLambda_with_convergence(self, population, toolbox, mu, lambda_, cxpb, mutpb, 
                                               ngen, stats=None, halloffame=None, verbose=False,
                                               convergence_threshold=0.001, patience=10):

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        
        # Convergence tracking
        fitness_history = []
        stagnation_count = 0
        best_fitness = float('inf')
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Vary the population
            offspring = tools.selRandom(population, lambda_)
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Apply crossover and mutation
            for i in range(1, len(offspring), 2):
                if random.random() < cxpb:
                    offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < mutpb:
                    offspring[i], = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            population[:] = toolbox.select(population + offspring, mu)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(population)

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            
            # Convergence detection
            current_best = min(ind.fitness.values[0] for ind in population)
            fitness_history.append(current_best)
            
            if verbose:
                print(f"Generation {gen:3d}: Best fitness = {current_best:8.4f}")
            
            # Check for improvement
            if current_best < best_fitness - convergence_threshold:
                best_fitness = current_best
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Check convergence based on variance in recent generations
            if len(fitness_history) >= patience:
                recent_variance = np.var(fitness_history[-patience:])
                
                if recent_variance < convergence_threshold:
                    if verbose:
                        print(f"\n🛑 CONVERGENCE DETECTED at generation {gen}!")
                        print(f"   Recent {patience} generations variance: {recent_variance:.6f}")
                        print(f"   Best fitness: {best_fitness:.4f}")
                    break
            
            # Early stopping if no improvement
            if stagnation_count >= patience:
                if verbose:
                    print(f"\n🛑 EARLY STOPPING at generation {gen}!")
                    print(f"   No improvement for {patience} generations")
                    print(f"   Best fitness: {best_fitness:.4f}")
                break
                
            if verbose and gen % 10 == 0:
                print(logbook.stream)

        return population, logbook
    
    def _evaluate_nutrition(self, individual: List[int], targets: Dict[str, float]) -> Tuple[float]:
        num_selected = sum(individual)

        if num_selected == 0:
            return (1e10,)
        elif num_selected > 12:
            return (1e9 + num_selected * 100000,)
        elif num_selected > 8:
            return (1e8 + (num_selected - 8) * 50000,)
        elif num_selected < 3:
            return (1e7,)

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
                      pop_size: int = 99, n_gen: int = 98,
                      cxpb: float = 0.893, mutpb: float = 0.591,
                      enable_convergence: bool = True,
                      convergence_threshold: float = 0.1,
                      patience: int = 5) -> pd.DataFrame:

        print(f"\n🧬 DEAP MEAL PLANNER")
        print(f"Population: {pop_size}, Max Generations: {n_gen}")
        print(f"Convergence: {'Enabled' if enable_convergence else 'Disabled'}")
        print(f"Targets: {daily_targets}")

        self.toolbox.register("evaluate", self._evaluate_nutrition, targets=daily_targets)

        mu = pop_size
        lambda_ = int(pop_size * 1.5)

        population = self.toolbox.population(n=mu)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("std", np.std)

        if enable_convergence:
            # Use custom algorithm with convergence detection
            population, logbook = self._custom_eaMuPlusLambda_with_convergence(
                population, self.toolbox,
                mu=mu, lambda_=lambda_,
                cxpb=cxpb, mutpb=mutpb,
                ngen=n_gen,
                stats=stats,
                halloffame=hof,
                verbose=True,
                convergence_threshold=convergence_threshold,
                patience=patience
            )
        else:
            # Use standard DEAP algorithm
            population, logbook = eaMuPlusLambda(
                population, self.toolbox,
                mu=mu, lambda_=lambda_,
                cxpb=cxpb, mutpb=mutpb,
                ngen=n_gen,
                stats=stats,
                halloffame=hof,
                verbose=True
            )

        best_individual = hof[0]
        best_fitness = best_individual.fitness.values[0]
        generations_run = len(logbook)
        
        print(f"\n🏆 OPTIMIZATION COMPLETE")
        print(f"   Generations run: {generations_run}/{n_gen}")
        print(f"   Best fitness: {best_fitness:.4f}")
        print(f"   Foods selected: {sum(best_individual)}")

        selected_foods = self.df[np.array(best_individual, dtype=bool)].copy()
        selected_foods['selected'] = True
        return selected_foods
    
    def evaluate_meal_plan(self, meal_plan: pd.DataFrame, daily_targets: Dict[str, float]) -> Dict[str, float]:
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
                
                # Calculate numerical score
                if nutrient in critical_nutrients:
                    if 95 <= percentage <= 105:
                        status = "🎯 Perfect"
                        score = 100
                    elif 90 <= percentage <= 110:
                        status = "✓ Good"
                        score = 80
                    else:
                        status = "❌ Poor"
                        score = 20
                else:
                    if 80 <= percentage <= 120:
                        status = "✓ Good"
                        score = 80
                    else:
                        status = "⚠️ Outside range"
                        score = 40
                
                results[nutrient] = {
                    'total': total,
                    'target': target,
                    'percentage': percentage,
                    'status': status,
                    'score': score  # Add this line
                }
                
                print(f"{nutrient.capitalize():15}: {total:8.1f} / {target:8.1f} ({percentage:6.1f}%) {status} - score: {score}")
        
        return results


def generate_deap_meal_plan(df: pd.DataFrame, daily_targets: Dict[str, float]) -> pd.DataFrame:
    try:
        generator = DEAPMealGenerator(df)
        meal_plan = generator.generate_meal_plan(daily_targets)
        
        if not meal_plan.empty:
            generator.evaluate_meal_plan(meal_plan, daily_targets)
        
        return meal_plan
        
    except Exception as e:
        print(f"❌ DEAP meal generation failed: {e}")
        return pd.DataFrame()
