import pandas as pd
import numpy as np
from math import ceil

# Load the new dataset (CSV instead of Excel)
data = pd.read_csv("supa_data.csv")

# Define column names based on the new dataset structure
columns = [
    "Food_code", "Main_food_description", "WWEIA_Category_number", "WWEIA_Category_description",
    "Energy_kcal", "Protein_g", "Carbohydrate_g", "Total_Fat_g", "Folic_acid_mcg",
    "Calcium_mg", "Iron_mg"
]

# Verify column alignment
if data.shape[1] != len(columns):
    raise ValueError(f"Expected {len(columns)} columns, but found {data.shape[1]} in the CSV file")
data.columns = columns

print(data.head())

# Nutritional targets for women with gestational diabetes
nutrient_targets = {
    "Carbohydrate_g": 175,
    "Protein_g": 71,
    "Total_Fat_g": (45, 78),  # Range
    "Folic_acid_mcg": (500, 600),  # Range
    "Calcium_mg": (900, 1000),  # Range
    "Iron_mg": (24, 40),  # Range
    "Vitamin_D_mcg": (0, 0)  # Placeholder, as Vitamin D is not in dataset
}

# Weights for nutrients in fitness function
nutrient_weights = {
    "Carbohydrate_g": 0.2,
    "Protein_g": 0.3,  # Higher weight for protein
    "Total_Fat_g": 0.2,
    "Folic_acid_mcg": 0.2,  # Higher weight for folic acid
    "Calcium_mg": 0.15,
    "Iron_mg": 0.15,
    "Vitamin_D_mcg": 0.0  # No weight since not in dataset
}

# Function to calculate maximum units per food item
def calculate_max_units(food_data, targets, max_calories):
    max_units = []
    for _, row in food_data.iterrows():
        # Calculate max units based on nutrient constraints
        nutrient_limits = []
        for nutrient, target in targets.items():
            nutrient_value = row[nutrient] if nutrient in row else 0  # Handle Vitamin D
            if nutrient_value == 0:  # Avoid division by zero
                nutrient_limits.append(float('inf'))
            else:
                if isinstance(target, tuple):
                    target_value = target[1]  # Use upper bound for ranges
                else:
                    target_value = target
                nutrient_limits.append(target_value / nutrient_value)
        # Include calorie constraint
        if row["Energy_kcal"] == 0:
            calorie_limit = float('inf')
        else:
            calorie_limit = max_calories / row["Energy_kcal"]
        max_units.append(min(nutrient_limits + [calorie_limit]))
    return np.array([ceil(min(max(0, u), 100)) for u in max_units])  # Cap at 100 units

# Fitness function
def calculate_fitness(chromosome, food_data, targets, max_calories):
    fitness = 100.0
    total_calories = 0
    nutrient_totals = {nutrient: 0 for nutrient in targets}

    for i, units in enumerate(chromosome):
        if units > 0:
            row = food_data.iloc[i]
            total_calories += units * row["Energy_kcal"]
            for nutrient in targets:
                nutrient_value = row[nutrient] if nutrient in row else 0  # Handle Vitamin D
                nutrient_totals[nutrient] += units * nutrient_value

    # Penalize if calories exceed max_calories
    if total_calories > max_calories:
        return 0

    # Calculate deviations and deduct from fitness
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
        fitness -= deviation * nutrient_weights[nutrient] * 25  # Max 25 points per nutrient

    return max(0, fitness)  # Ensure non-negative fitness

# SBX Crossover
def sbx_crossover(parent1, parent2, max_units, eta_c=15):
    offspring1, offspring2 = parent1.copy(), parent2.copy()
    for i in range(len(parent1)):
        if np.random.random() < 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-14:
                x1, x2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                beta = 1.0 + (2.0 * (x1 - 0) / (x2 - x1))
                alpha = 2.0 - beta ** (-(eta_c + 1.0))
                rand = np.random.random()
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                offspring1[i] = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
                offspring2[i] = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))
                # Ensure offspring are within bounds
                offspring1[i] = max(0, min(ceil(offspring1[i]), max_units[i]))
                offspring2[i] = max(0, min(ceil(offspring2[i]), max_units[i]))
    return offspring1, offspring2

# Mutation
def mutate(chromosome, max_units, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if np.random.random() < mutation_rate:
            chromosome[i] = np.random.randint(0, max_units[i] + 1)
    return chromosome

# Genetic Algorithm
def genetic_algorithm(food_data, targets, max_calories, population_size=100, generations=3, mutation_rate=0.1, eta_c=15):
    num_foods = len(food_data)
    max_units = calculate_max_units(food_data, targets, max_calories)
    
    # Initialize population2
    population = [np.random.randint(0, max_units + 1, size=num_foods) for _ in range(population_size)]
    
    for generation in range(generations):
        # Calculate fitness for each chromosome
        fitness_scores = [calculate_fitness(chrom, food_data, targets, max_calories) for chrom in population]
        print('Generation: ',generations,'Scores: ', fitness_scores)
        
        # Select parents using Fitness Proportional Selection
        fitness_sum = sum(fitness_scores)
        if fitness_sum == 0:
            probabilities = np.ones(population_size) / population_size
        else:
            probabilities = [f / fitness_sum for f in fitness_scores]
        parent_indices = np.random.choice(range(population_size), size=population_size, p=probabilities)
        parents = [population[i].copy() for i in parent_indices]
        
        # Generate offspring via crossover
        offspring = []
        for i in range(0, population_size, 2):
            if i + 1 < population_size:
                child1, child2 = sbx_crossover(parents[i], parents[i+1], max_units, eta_c)
                offspring.extend([child1, child2])
            else:
                offspring.append(parents[i])
        
        # Apply mutation
        offspring = [mutate(chrom.copy(), max_units, mutation_rate) for chrom in offspring]
        
        # Combine parents and offspring, select top N
        combined = population + offspring
        fitness_scores = [calculate_fitness(chrom, food_data, targets, max_calories) for chrom in combined]
        indices = np.argsort(fitness_scores)[::-1][:population_size]
        population = [combined[i].copy() for i in indices]
        
        # Check convergence (optional: stop if best fitness doesn't improve)
        best_fitness = max(fitness_scores)
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")
    
    # Return best chromosome
    best_idx = np.argmax([calculate_fitness(chrom, food_data, targets, max_calories) for chrom in population])
    return population[best_idx]

# Function to interpret and display the dietary plan
def interpret_diet_plan(chromosome, food_data, targets, max_calories):
    total_nutrients = {nutrient: 0 for nutrient in targets}
    total_calories = 0
    plan = []
    
    for i, units in enumerate(chromosome):
        if units > 0:
            row = food_data.iloc[i]
            total_calories += units * row["Energy_kcal"]
            for nutrient in targets:
                nutrient_value = row[nutrient] if nutrient in row else 0  # Handle Vitamin D
                total_nutrients[nutrient] += units * nutrient_value
            plan.append((row["Main_food_description"], units, row["Energy_kcal"] * units))
    
    print("\nOptimized Dietary Plan:")
    print("----------------------")
    for food, units, cals in plan:
        print(f"{food}: {units} units ({cals:.1f} kcal)")
    print(f"\nTotal Calories: {total_calories:.1f} kcal (Max: {max_calories})")
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
    weight=40
    max_calories = weight * 24

    category_buckets = df.groupby('Ultimate Category').apply(lambda x: x.index.tolist()).to_dict()
    print(category_buckets)
    # Display the number of items in each category
    category_counts = {category: len(items) for category, items in category_buckets.items()}
    print(category_counts)  
    return

    # Run genetic algorithm
    best_plan = genetic_algorithm(data, nutrient_targets, max_calories)
    
    # Interpret and display results
    interpret_diet_plan(best_plan, data, nutrient_targets, max_calories)

if __name__ == "__main__":
    main()