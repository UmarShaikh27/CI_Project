import pandas as pd
import numpy as np
from math import ceil

# Load the dataset
data = pd.read_csv("supa_data.csv")

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
    "Protein (g)": 0.3,
    "Total Fat (g)": 0.2,
    "Folic acid (mcg)": 0.2,
    "Calcium (mg)": 0.15,
    "Iron (mg)": 0.15,
}

# Get unique categories
categories = data["Ultimate Category"].values
unique_categories = np.unique(categories)

# Calculate maximum units per food item
def calculate_max_units(food_data, targets, max_calories):
    max_units = []
    for _, row in food_data.iterrows():
        nutrient_limits = []
        for nutrient, target in targets.items():
            nutrient_value = row[nutrient] if nutrient in row else 0
            if nutrient_value == 0:
                nutrient_limits.append(float('inf'))
            else:
                if isinstance(target, tuple):
                    target_value = target[1]
                else:
                    target_value = target
                nutrient_limits.append(target_value / nutrient_value)
        if row["Energy (kcal)"] == 0:
            calorie_limit = float('inf')
        else:
            calorie_limit = max_calories / row["Energy (kcal)"]
        max_units.append(min(nutrient_limits + [calorie_limit]))
    return np.array([ceil(min(max(0, u), 3)) for u in max_units])

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
        units = np.random.randint(1, max_units[idx] + 1)
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

def select_parents(population, fitness_scores, population_size, scheme="fitness_proportional", tournament_size=3):
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
    if scheme == "fitness_proportional":
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

# Fitness function with sparsity and diversity penalties
def calculate_fitness(chromosome, food_data, targets, max_calories, max_foods=2, sparsity_penalty=0.5, diversity_penalty=1.0):
    fitness = 100.0
    total_calories = 0
    nutrient_totals = {nutrient: 0 for nutrient in targets}
    
    # Track categories for diversity
    categories = []
    num_foods_selected = len(chromosome)
    
    # Calculate nutrient totals
    for idx, units in chromosome:
        row = food_data.iloc[idx]
        total_calories += units * row["Energy (kcal)"]
        categories.append(row["Ultimate Category"])
        for nutrient in targets:
            nutrient_value = row[nutrient] if nutrient in row else 0
            nutrient_totals[nutrient] += units * nutrient_value
    
    # Penalize if calories exceed max_calories
    if total_calories > max_calories:
        # print("\ntotal calories: -->", total_calories)
        # print("max calories: -->", max_calories)
        return 0
    
    # Penalize excessive food items
    if num_foods_selected > max_foods:
        fitness -= sparsity_penalty * (num_foods_selected - max_foods)
    
    # Penalize low category diversity (fewer unique categories than desired)
    unique_categories = len(set(categories))
    min_categories = min(3, num_foods_selected)  # Expect at least 3 categories or num foods
    if unique_categories < min_categories:
        fitness -= diversity_penalty * (min_categories - unique_categories)
    
    # Calculate nutrient deviations
    for nutrient, target in targets.items():
        actual = nutrient_totals[nutrient]
        if isinstance(target, tuple):
            min_target, max_target = target
            if actual < min_target:
                deviation = (min_target - actual) / min_target
            elif actual > max_target:
                deviation = (actual - max_target) / max_target
            else:
                deviation = 0
        else:
            deviation = abs(actual - target) / target if target != 0 else 0
        fitness -= deviation * nutrient_weights[nutrient] * 25
    
    return max(0, fitness)

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
            units = np.random.randint(1, max_units[idx] + 1)
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
            new_units = np.random.randint(0, max_units[idx] + 1)
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
            units = np.random.randint(1, max_units[idx] + 1)
            chromosome.append((int(idx), units))
    
    # Remove excess items if over max_foods
    if len(chromosome) > max_foods:
        chromosome = np.random.choice(chromosome, size=max_foods, replace=False).tolist()
    
    return chromosome

# Genetic Algorithm
def genetic_algorithm(food_data, targets, max_calories, max_foods=10, max_per_category=3, population_size=100, generations=80, mutation_rate=0.1, eta_c=15,parent_selection_scheme="tournament",):
    num_foods = len(food_data)
    max_units = calculate_max_units(food_data, targets, max_calories)
    
    # Initialize population
    population = [
        initialize_sparse_chromosome(num_foods, max_units, max_foods, food_data, max_per_category)
        for _ in range(population_size)
    ]

    # return
    
    
    for generation in range(generations):
        # Calculate fitness
        fitness_scores = [
            calculate_fitness(chrom, food_data, targets, max_calories, max_foods, diversity_penalty=1.0)
            for chrom in population
        ]
        
       
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
        
        

        # Combine and select top N
        combined = population + offspring
        fitness_scores = [
            calculate_fitness(chrom, food_data, targets, max_calories, max_foods, diversity_penalty=1.0)
            for chrom in combined
        ]
        indices = np.argsort(fitness_scores)[::-1][:population_size]
        population = [combined[i].copy() for i in indices]
        
        # Log progress
        best_fitness = max(fitness_scores)
        # print("\nBEST FITNESS THIS GEN: ", best_fitness)
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")
    
    # Return best chromosome
    best_idx = np.argmax([
        calculate_fitness(chrom, food_data, targets, max_calories, max_foods, diversity_penalty=1.0)
        for chrom in population
    ])
    return population[best_idx]

# Interpret and display dietary plan
def interpret_diet_plan(chromosome, food_data, targets, max_calories):
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
    
    print("\nOptimized Dietary Plan:")
    print("----------------------")
    for food, units, cals, category in plan:
        print(f"{food}: {units} units ({cals:.1f} kcal) [Category: {category}]")
    print(f"\nTotal Calories: {total_calories:.1f} kcal (Max: {max_calories})")
    print(f"Unique Categories: {len(set(categories))} ({set(categories)})")
    print("\nNutrient Breakdown:")
    for nutrient, value in total_nutrients.items():
        target = targets[nutrient]
        if isinstance(target, tuple):
            print(f"{nutrient}: {value:.1f} (Target: {target[0]}-{target[1]})")
        else:
            print(f"{nutrient}: {value:.1f} (Target: {target})")
    print(f"\nFitness Score: {calculate_fitness(chromosome, food_data, targets, max_calories):.2f}")

# Main execution
def main():
    # weight = float(input("Enter weight in kg: "))
    weight = 60
    max_calories = weight * 24
    max_foods = 5
    max_per_category = 3
    
    # Run genetic algorithm
    best_plan = genetic_algorithm(data, nutrient_targets, max_calories, max_foods, max_per_category)
    
    # Interpret and display results
    interpret_diet_plan(best_plan, data, nutrient_targets, max_calories)

if __name__ == "__main__":
    main()