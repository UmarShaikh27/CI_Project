# Gestational Diabetes Diet Planner

An intelligent dietary planning system that uses genetic algorithms to generate optimized meal plans for women with gestational diabetes. This project combines computational intelligence with healthcare to create personalized, nutritionally balanced meal plans that help manage blood glucose levels during pregnancy.

## Features

- **Personalized Meal Planning**: Generates individualized meal plans based on:
  - Caloric requirements
  - Essential nutrient targets
  - Dietary preferences and restrictions
  - Food category diversity

- **Advanced Genetic Algorithm Implementation**:
  - Multiple parent selection schemes (FPS, Tournament, Rank-based)
  - Sophisticated survivor selection (Elitist, Age-based)
  - Simulated Binary Crossover (SBX) for real-parameter optimization
  - Adaptive mutation rates with decay
  - Multi-objective fitness evaluation

- **Nutritional Optimization**:
  - Optimizes for key nutrients:
    - Carbohydrates: 175g target
    - Protein: 71g target
    - Total Fat: 45-78g range
    - Folic Acid: 500-600 mcg range
    - Calcium: 900-1000 mg range
    - Iron: 24-40 mg range
  - Caloric intake based on individual weight
  - Category-based food diversity

- **Visualization and Output**:
  - Progress tracking with fitness graphs
  - Detailed Excel reports for each meal plan
  - Comprehensive nutritional analysis
  - Category distribution visualization

## Requirements

```
pandas
numpy
matplotlib
```

## Installation

1. Clone this repository:
```powershell
git clone [your-repo-url]
```

2. Install required packages:
```powershell
pip install pandas numpy matplotlib
```

3. Ensure you have the dataset file `super_data.csv` in your project directory

## Usage

1. Run the main script:
```powershell
python supercode.py
```

2. The program will generate:
- Three optimized meal plans saved as Excel files (`Top_Meal_Plan_1.xlsx`, `Top_Meal_Plan_2.xlsx`, `Top_Meal_Plan_3.xlsx`)
- A fitness progress graph (`fitness_progress.png`)

## Configuration

You can modify the following parameters in `main()`:

```python
weight = 80  # User's weight in kg
max_foods = 10  # Maximum foods in a meal plan
max_per_category = 2  # Maximum items from each food category
generations = 1000  # Number of genetic algorithm generations
population_size = 200  # Size of the population in each generation
```

## Algorithm Details

### Genetic Algorithm Components

1. **Chromosome Representation**
   - Sparse representation using (food_index, units) tuples
   - Variable-length chromosomes with maximum size constraint

2. **Selection Mechanisms**
   - Parent Selection:
     - Fitness Proportional Selection (FPS)
     - Tournament Selection
     - Rank-based Selection
   - Survivor Selection:
     - Elitist Selection
     - Age-based Selection

3. **Genetic Operators**
   - SBX Crossover with control parameter η
   - Adaptive Mutation with decay
   - Smart initialization with category constraints

4. **Fitness Evaluation**
   - Multi-objective fitness function
   - Nutrient target adherence
   - Caloric requirement compliance
   - Category diversity promotion
   - Constraint satisfaction penalties

## Healthcare Applications

This system assists healthcare providers and patients by:

- Generating nutritionally balanced meal plans
- Ensuring adherence to gestational diabetes dietary guidelines
- Promoting dietary diversity
- Maintaining essential nutrient intake
- Supporting maternal and fetal health
- Reducing the risk of pregnancy complications

## Output Files

1. **Excel Reports** (`Top_Meal_Plan_[1-3].xlsx`):
   - Detailed food items and quantities
   - Nutritional content per item
   - Total nutritional values
   - Category distribution
   - Caloric information

2. **Fitness Graph** (`fitness_progress.png`):
   - Evolution of best and average fitness
   - Convergence visualization
   - Generation-wise progress



## Acknowledgments

- Dataset source: ] A. R. S. United States Department of Agriculture, “Food and nutrient database for dietary studies 2021-2023,” 2021, available at [USDA Food Composition Database https://www.ars.usda.gov/ARS.usda.gov/Services/default.htmcatid=Food
- Nutritional guidelines based on medical research for gestational diabetes
- Computational intelligence concepts from modern genetic algorithm literature
