import pandas as pd
import numpy as np
from math import ceil,floor
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the dataset
data = pd.read_csv("Supadupa_data.csv")

# Verify expected columns
expected_columns = [
    "Main food description", "Ultimate Category", "WWEIA Category description",
    "Energy (kcal)", "Protein (g)", "Total Fat (g)", "Carbohydrate (g)",
    "Calcium (mg)", "Iron (mg)", "Folic acid (mcg)"]
if list(data.columns) != expected_columns:
    raise ValueError(f"Expected columns {expected_columns}, but found {data.columns}")

# Nutritional targets (same as before)
nutrient_targets = {
    "Carbohydrate (g)": 175,
    "Protein (g)": 71,
    "Total Fat (g)": (45, 78),
    "Folic acid (mcg)": (500, 600),
    "Calcium (mg)": (900, 1000),
    "Iron (mg)": (24, 40),
}
# Note: Dataset lacks Vitamin D, so it's excluded

# Nutrient weights
nutrient_weights = {
    "Carbohydrate (g)": 0.2,
    "Protein (g)": 0.25,
    "Total Fat (g)": 0.15,
    "Folic acid (mcg)": 0.25,
    "Calcium (mg)": 0.15,
    "Iron (mg)": 0.2,
}

# Get unique categories
categories = data["Ultimate Category"].values
unique_categories = np.unique(categories)

# Calculate maximum units per food item
def calculate_max_units(food_data, targets, max_calories):
    max_units = []
    for _, row in food_data.iterrows():
        nutrient_limits = [
            targets[nutrient] / row[nutrient]
            for nutrient in targets
            if isinstance(targets[nutrient], (int, float)) and row[nutrient] > 0.01  # Avoid division by near-zero
        ]
        calorie_limit = max_calories / row["Energy (kcal)"] if row["Energy (kcal)"] > 0 else float("inf")
        if not nutrient_limits:  # Handle foods with no valid nutrient contributions
            max_unit = 1  # Default to 1 unit if no nutrient constraints
        else:
            max_unit = int(np.floor(min(nutrient_limits + [calorie_limit, 3])))
        max_unit = max(max_unit, 1)  # Ensure at least 1 unit
        max_units.append(max_unit)
    return max_units

# Initialize sparse chromosome as list of (index, units) tuples with category diversity
def initialize_sparse_chromosome(num_foods, max_units, max_foods=10, food_data=None, max_per_category=3):
    
    # Initialize empty chromosome
    chromosome = []
    selected_indices = []
    category_counts = {cat: 0 for cat in unique_categories}
    # Select max_foods items, respecting category limits
    while len(chromosome) < max_foods:
        # Candidates: foods not yet selected and whose category isn't over limit
        valid_indices = [
            i for i in range(num_foods)
            if i not in selected_indices and
            category_counts[categories[i]] < max_per_category
        ]
        if not valid_indices:
            break  # No more valid items
        idx = np.random.choice(valid_indices)
        units = round(np.random.uniform(1, max_units[idx] + 1),1)
        chromosome.append((int(idx), units))
        selected_indices.append(idx)
        category_counts[categories[idx]] += 1
    
    formatted = []
    for idx, units in chromosome:
        food_name = food_data.iloc[idx]["Main food description"]
        formatted.append(f"{food_name}: {units} units")

    # print("\nChromosome: ", formatted)
    # print(len(chromosome))
    return chromosome

import numpy as np

def select_parents(population, fitness_scores, population_size, scheme="fps", tournament_size=3):
    """
    Select parents from the population based on the specified scheme.
    
    Args:
        population: List of chromosomes (lists of (index, units) tuples).
        fitness_scores: List of fitness scores for each chromosome.
        population_size: Number of parents to select.
        scheme: Selection scheme ("fitness_proportional", "tournament", "rank").
        tournament_size: Number of individuals in each tournament (for tournament selection).
    
    Returns:
        List of selected parent chromosomes (copies).
    """
    if scheme == "fps":
        fitness_sum = sum(fitness_scores)
        if fitness_sum == 0:
            probabilities = np.ones(population_size) / population_size
        else:
            probabilities = [f / fitness_sum for f in fitness_scores]
        parent_indices = np.random.choice(range(population_size), size=population_size, p=probabilities)
    
    elif scheme == "tournament":
        parent_indices = []
        for _ in range(population_size):
            tournament_indices = np.random.choice(range(population_size), size=tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner = tournament_indices[np.argmax(tournament_fitness)]
            parent_indices.append(winner)
    
    elif scheme == "rank":
        ranks = np.argsort(np.argsort(fitness_scores)) + 1  # Ranks: 1 (worst) to population_size (best)
        rank_sum = sum(ranks)
        probabilities = [r / rank_sum for r in ranks]
        parent_indices = np.random.choice(range(population_size), size=population_size, p=probabilities)
    
    else:
        raise ValueError(f"Unknown parent selection scheme: {scheme}")
    
    return [population[i].copy() for i in parent_indices]

def select_survivors(population, offspring, fitness_scores, population_size, scheme="elitist", tournament_size=3, elite_size=1):
    """
    Select survivors for the next generation from population and offspring.
    
    Args:
        population: List of current population chromosomes.
        offspring: List of offspring chromosomes.
        fitness_scores: List of fitness scores for combined population + offspring.
        population_size: Number of survivors to select.
        scheme: Selection scheme ("elitist", "tournament", "age").
        tournament_size: Number of individuals in each tournament (for tournament selection).
        elite_size: Number of best individuals to preserve (for age-based selection).
    
    Returns:
        List of selected survivor chromosomes (copies).
    """
    combined = population + offspring
    if scheme == "elitist":
        indices = np.argsort(fitness_scores)[::-1][:population_size]
        return [combined[i].copy() for i in indices]
    
    elif scheme == "tournament":
        survivor_indices = []
        for _ in range(population_size):
            tournament_indices = np.random.choice(range(len(combined)), size=tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner = tournament_indices[np.argmax(tournament_fitness)]
            survivor_indices.append(winner)
        return [combined[i].copy() for i in survivor_indices]
    
    elif scheme == "age":
        # Preserve elite_size best individuals
        elite_indices = np.argsort(fitness_scores)[::-1][:elite_size]
        elite = [combined[i].copy() for i in elite_indices]
        # Fill with offspring (newest individuals)
        remaining = offspring[:population_size - elite_size]
        return elite + [chrom.copy() for chrom in remaining]
    
    else:
        raise ValueError(f"Unknown survivor selection scheme: {scheme}")

def plot_fitness_history(generations,best_fitness_history, avg_fitness_history,file_path="fitness_progress.png"):
    """
    Plot best and average fitness over generations.
    
    Args:
        best_fitness_history: List of best fitness scores per generation.
        avg_fitness_history: List of average fitness scores per generation.
        generations: Number of generations.
    """
    plt.figure(figsize=(10, 6))
    generations_range = range(1, generations + 1)
    plt.plot(generations_range, best_fitness_history, label='Best Fitness', color='blue', linewidth=2)
    plt.plot(generations_range, avg_fitness_history, label='Average Fitness', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Best and Average Fitness Over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.show()
    print(f"Fitness progress graph saved to {file_path}")

# Fitness function with sparsity and diversity penalties
def calculate_fitness(chromosome, food_data, targets, max_calories, max_foods=10, sparsity_penalty=0.5, diversity_penalty=1.0):
    nutrient_totals = {nutrient: 0 for nutrient in targets}
    total_calories = 0
    category_counts = defaultdict(int)
    
    # Calculate nutrient and calorie totals
    for idx, units in chromosome:
        row = food_data.iloc[idx]
        for nutrient in targets:
            nutrient_totals[nutrient] += units * row[nutrient]
        total_calories += units * row["Energy (kcal)"]
        category = row["Ultimate Category"]
        category_counts[category] += 1
    
    # Base fitness score
    fitness = 100.0    
    
    for nutrient, target in targets.items():
        actual = nutrient_totals[nutrient]
        # Handle range targets (e.g., Total Fat (45, 78))
        if isinstance(target, tuple):
            target_mid = (target[0] + target[1]) / 2
            target_range = target[1] - target[0]
            deviation = abs(actual - target_mid) / target_mid
        else:
            deviation = abs(actual - target) / target
        
        # Penalty capped at 25 points per nutrient
        penalty = min(deviation * 25, 25)
        weight = nutrient_weights.get(nutrient, 0.3)
        fitness -= weight * penalty
    
    # Calorie penalty
    calorie_deviation = abs(total_calories - max_calories) / max_calories
    if calorie_deviation <= 0.1:
        # Moderate penalty for deviations within 10%
        calorie_penalty = min(calorie_deviation * 25, 25)
    else:
        # Heavier penalty for deviations > 10%
        calorie_penalty = min(calorie_deviation * 50, 50)
    calorie_weight = 0.4  # Emphasize calorie adherence
    fitness -= calorie_weight * calorie_penalty
    
    # Constraint penalties
    if len(chromosome) > max_foods:
        fitness -= 10 * (len(chromosome) - max_foods)  # Penalize excess foods
    
    max_per_category = 3
    for category, count in category_counts.items():
        if count > max_per_category:
            fitness -= 10 * (count - max_per_category)  # Penalize category overages
    
    # Diversity penalty (if applicable)
    if diversity_penalty > 0:
        unique_foods = len(chromosome)
        diversity_score = max(0, 10 - unique_foods)  # Encourage up to 10 foods
        fitness -= diversity_penalty * diversity_score
    
    # Ensure fitness is non-negative
    fitness = max(fitness, 0)
    
    return fitness

# SBX crossover for tuple-based chromosomes
def sbx_crossover(parent1, parent2, max_units, max_foods=10, food_data=None, max_per_category=3, eta_c=15):
    # Get categories
    categories = food_data["Ultimate Category"].values
    
    # Combine indices from both parents
    indices1 = [idx for idx, _ in parent1]
    indices2 = [idx for idx, _ in parent2]
    all_indices = list(set(indices1 + indices2))
    
    # Initialize offspring
    offspring1, offspring2 = [], []
    category_counts1 = {cat: 0 for cat in np.unique(categories)}
    category_counts2 = {cat: 0 for cat in np.unique(categories)}
    
    # Randomly select up to max_foods indices
    np.random.shuffle(all_indices)
    selected_indices = all_indices[:min(len(all_indices), max_foods)]
    
    # Perform SBX on units for shared indices
    for idx in selected_indices:
        if idx in indices1 and idx in indices2 and category_counts1[categories[idx]] < max_per_category and category_counts2[categories[idx]] < max_per_category:
            units1 = next(u for i, u in parent1 if i == idx)
            units2 = next(u for i, u in parent2 if i == idx)
            if np.random.random() < 0.5 and abs(units1 - units2) > 1e-14:
                x1, x2 = min(units1, units2), max(units1, units2)
                beta = 1.0 + (2.0 * (x1 - 0) / (x2 - x1))
                alpha = 2.0 - beta ** (-(eta_c + 1.0))
                rand = np.random.random()
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                u1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
                u2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))
                u1 = max(0, min(ceil(u1), max_units[idx]))
                u2 = max(0, min(ceil(u2), max_units[idx]))
                if u1 > 0:
                    offspring1.append((idx, u1))
                    category_counts1[categories[idx]] += 1
                if u2 > 0:
                    offspring2.append((idx, u2))
                    category_counts2[categories[idx]] += 1
        elif idx in indices1 and category_counts1[categories[idx]] < max_per_category and np.random.random() < 0.5:
            units = next(u for i, u in parent1 if i == idx)
            offspring1.append((idx, units))
            category_counts1[categories[idx]] += 1
        elif idx in indices2 and category_counts2[categories[idx]] < max_per_category and np.random.random() < 0.5:
            units = next(u for i, u in parent2 if i == idx)
            offspring2.append((idx, units))
            category_counts2[categories[idx]] += 1
    
    # Fill remaining slots with new items if needed
    for offspring, counts in [(offspring1, category_counts1), (offspring2, category_counts2)]:
        while len(offspring) < max_foods:
            valid_indices = [
                i for i in range(len(food_data))
                if i not in [idx for idx, _ in offspring] and counts[categories[i]] < max_per_category
            ]
            if not valid_indices:
                break
            idx = np.random.choice(valid_indices)
            units = round(np.random.uniform(1, max_units[idx] + 1),1)
            offspring.append((idx, units))
            counts[categories[idx]] += 1
    
    # Enforce max_foods
    offspring1 = offspring1[:max_foods]
    offspring2 = offspring2[:max_foods]
    
    return offspring1, offspring2

# Mutation for tuple-based chromosomes
def mutate(chromosome, max_units, max_foods=10, food_data=None, max_per_category=3, mutation_rate=0.1):
    categories = food_data["Ultimate Category"].values
    chromosome = chromosome.copy()
    
    # Mutate units of existing items
    for i, (idx, units) in enumerate(chromosome):
        if np.random.random() < mutation_rate:
            new_units = round(np.random.uniform(1, max_units[idx] + 1),1)
            if new_units == 0:
                chromosome.pop(i)
            else:
                chromosome[i] = (idx, new_units)
     
    # Add new food item if under max_foods
    category_counts = {cat: 0 for cat in np.unique(categories)}
    for idx, _ in chromosome:
        category_counts[categories[idx]] += 1
    
    if len(chromosome) < max_foods and np.random.random() < mutation_rate:
        valid_indices = [
            i for i in range(len(food_data))
            if i not in [idx for idx, _ in chromosome] and category_counts[categories[i]] < max_per_category
        ]
        if valid_indices:
            idx = np.random.choice(valid_indices)
            units = round(np.random.uniform(1, max_units[idx] + 1),1)
            chromosome.append((int(idx), units))
    
    # Remove excess items if over max_foods
    if len(chromosome) > max_foods:
        chromosome = np.random.choice(chromosome, size=max_foods, replace=False).tolist()
    
    return chromosome

# Genetic Algorithm
def genetic_algorithm(food_data, targets, max_calories, max_foods=10, max_per_category=3, population_size=200, generations=10, parent_selection_scheme="fps",survivor_selection_scheme="elitist",mutation_rate=0.1, eta_c=15):
    tournament_size=3;
    elite_size=1;
    decay_factor = 0.2
    num_foods = len(food_data)
    max_units = calculate_max_units(food_data, targets, max_calories)
    best_fitness_history = []
    avg_fitness_history = []
    # Initialize population
    population = [
        initialize_sparse_chromosome(num_foods, max_units, max_foods, food_data, max_per_category)
        for _ in range(population_size)
    ]
    
    
    for generation in range(generations):
        # Calculate fitness
        mutation_rate = mutation_rate * np.exp(-decay_factor * (generation / generations))
        fitness_scores = [
            calculate_fitness(chrom, food_data, targets, max_calories, max_foods, diversity_penalty=1.0)
            for chrom in population
        ]
        
        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        # # Select parents
        # fitness_sum = sum(fitness_scores)

        # if fitness_sum == 0:
        #     probabilities = np.ones(population_size) / population_size
        # else:
        #     probabilities = [f / fitness_sum for f in fitness_scores]
    
        # parent_indices = np.random.choice(range(population_size), size=population_size, p=probabilities)
        # parents = [population[i].copy() for i in parent_indices]
        
        parents = select_parents(
            population, 
            fitness_scores, 
            population_size, 
            scheme=parent_selection_scheme, 
            tournament_size=5
        )

        # Generate offspring
        offspring = []
        for i in range(0, population_size, 2):
            if i + 1 < population_size:
                child1, child2 = sbx_crossover(
                    parents[i], parents[i+1], max_units, max_foods, food_data, max_per_category, eta_c
                )
                
                offspring.extend([child1, child2])
            else:
                offspring.append(parents[i])

        
        # Apply mutation
        offspring = [
            mutate(chrom, max_units, max_foods, food_data, max_per_category, mutation_rate)
            for chrom in offspring
        ]
        
        

        # Select survivors
        combined_fitness = [
            calculate_fitness(chrom, food_data, targets, max_calories, max_foods, diversity_penalty=1.0)
            for chrom in population + offspring
        ]
        population = select_survivors(
            population, 
            offspring, 
            combined_fitness, 
            population_size, 
            scheme=survivor_selection_scheme, 
            tournament_size=tournament_size,
            elite_size=elite_size
        )
        
        # Log progress
        best_fitness = max(fitness_scores)
        # print("\nBEST FITNESS THIS GEN: ", best_fitness)
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness:.2f} Avg Fitness = {avg_fitness:.2f}")
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_chrom = population[best_idx]
            # print(best_chrom)
    
    # Return best chromosome
    best_idx = np.argmax([
        calculate_fitness(chrom, food_data, targets, max_calories, max_foods, diversity_penalty=1.0)
        for chrom in population
    ])
    return population[best_idx] ,best_fitness_history,avg_fitness_history

# Interpret and display dietary plan
def interpret_diet_plan(chromosome, food_data, targets, max_calories, max_foods):
    total_nutrients = {nutrient: 0 for nutrient in targets}
    total_calories = 0
    plan = []
    categories = []
    
    for idx, units in chromosome:
        row = food_data.iloc[idx]
        total_calories += units * row["Energy (kcal)"]
        categories.append(row["Ultimate Category"])
        for nutrient in targets:
            nutrient_value = row[nutrient] if nutrient in row else 0
            total_nutrients[nutrient] += units * nutrient_value
        plan.append((row["Main food description"], units, row["Energy (kcal)"] * units, row["Ultimate Category"]))
    
    return plan, total_nutrients, total_calories
    # print("\nOptimized Dietary Plan:")
    # print("----------------------")
    # for food, units, cals, category in plan:
    #     print(f"{food}: {units} units ({cals:.1f} kcal) [Category: {category}]")
    # print(f"\nTotal Calories: {total_calories:.1f} kcal (Max: {max_calories})")
    # print(f"Unique Categories: {len(set(categories))} ({set(categories)})")
    # print("\nNutrient Breakdown:")
    # for nutrient, value in total_nutrients.items():
    #     target = targets[nutrient]
    #     if isinstance(target, tuple):
    #         print(f"{nutrient}: {value:.1f} (Target: {target[0]}-{target[1]})")
    #     else:
    #         print(f"{nutrient}: {value:.1f} (Target: {target})")
    # print(f"\nFitness Score: {calculate_fitness(chromosome, food_data, targets, max_calories,max_foods):.2f}")

# Save top 3 plans to memory


def save_top_3_plans(plans, nutrient_targets):     
    for rank, (plan, _, _) in enumerate(plans, start=1):
        meal_data = []
        nutrient_totals = {key: 0 for key in nutrient_targets}
        total_calories = 0.0
        
        for item, units, cals, category in plan:
            # Extract individual nutrient values for this item
            row = data[data["Main food description"] == item].iloc[0]
            item_nutrients = {nutrient: units * row[nutrient] for nutrient in nutrient_targets}
            
            # Add the item data to the meal plan
            meal_data.append({
                "Rank": rank,
                "Food Item": item,
                "Units": units,
                "Calories": cals,
                "Category": category,
                **item_nutrients
            })
            total_calories += cals
            # Update nutrient totals
            for nutrient, value in item_nutrients.items():
                nutrient_totals[nutrient] += value
        
        # Create total row for this rank
        total_row = {
            "Rank": f"Total for Plan {rank}",
            "Food Item": "All Items in Plan",
            "Units": "-",
            "Calories": f"{total_calories:.1f}",
            "Category": "-"
        }
        for nutrient, target in nutrient_targets.items():
            total_row[nutrient] = f"{nutrient_totals.get(nutrient, 0):.1f}"
        meal_data.append(total_row)
    
        # Convert to DataFrame and save each plan separately
        df = pd.DataFrame(meal_data)
        file_name = f"Top_Meal_Plan_{rank}.xlsx"
        df.to_excel(file_name, index=False)
        print(f"Plan {rank} with totals saved to {file_name}.")

    
    # print("Top 3 meal plans with nutrient totals and targets saved to memory.")
    #df.to_excel("Top_3_Meal_Plans_with_Totals.xlsx", index=False)


# Decaying mutation rate function





# Main execution
def main():
    weight = 80
    max_calories = weight * 24
    max_foods = 10
    max_per_category = 2
    parent_selection_scheme = "rank"
    survivor_selection_scheme = "elitist"
    generations = 1000
    population_size = 200
    initial_mutation_rate = 0.5
    eta_c = 15
    
    # Run genetic algorithm
    best_plan, best_fitness_history, avg_fitness_history = genetic_algorithm(
        data, nutrient_targets, max_calories, max_foods, max_per_category, population_size, generations, parent_selection_scheme, survivor_selection_scheme, initial_mutation_rate, eta_c
    )
    
    # Extract final plans from the best chromosome
    plans = []
    for i in range(3):
        plan, nutrients, calories = interpret_diet_plan(best_plan, data, nutrient_targets, max_calories, max_foods)
        plans.append((plan, nutrients, calories))
    
    
    #plot and save fitness progress
    plot_fitness_history(generations, best_fitness_history, avg_fitness_history)
    
    # Save the top 3 plans to separate Excel files
    save_top_3_plans(plans, nutrient_targets)
    print("Top 3 meal plans with totals saved as separate Excel files.")

if __name__ == "__main__":
    main()
