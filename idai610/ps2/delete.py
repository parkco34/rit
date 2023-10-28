import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from textwrap import dedent

generation_stats = {
"avg_fitness": [], "best_fitness": [], "best_active_genes": []
}
# Best stats for best generation
best_solution = {'fitness': 0.73, 'active_genes': 0, 'generation':
                    0}

method_stats = {}

def get_initial_population(config_file):
    """
    Generates initial population for generation 0
    ----------------------------------------------
    INPUT:
        config_file: (str) Path to config file

    OUTPUT:
        g: (int) current generation
        chromosomes :(matrix or 2D array) Population of individual
        chromosomes
        W:(int) Knapsack capacity
        S: (list of tuples)Each tuple is an item (w_i, v_1)
        stop:(int) Final generation (stop condition)
    """
    try:
        with open(config_file, "r") as file:
            lines = file.readlines()

    except FileNotFoundError:
        print(f"{config_file} not found... ¯\_( ͡° ͜ʖ ͡°)_/¯ ")

    except Exception as e:
        print(f"Something went wrong!   ¯\_( ͡° ͜ʖ ͡°)_/¯  \n{e}\n")

    pop_size, n, stop, W = map(int, [lines[i].strip() for i in
                                            range(4)])
    S = [tuple(map(int, line.strip().split())) for line in lines[4:]]
    # Initialize empty population
    population = np.random.randint(2, size=(pop_size, n))

    g = 0 # initial generation
    return population, W, S, g, stop

population, capacity, weight_value, generation, stop = get_initial_population("config_1.txt")


def fitness_func(chromosome):
    """
    Determines fitness of chromosome by taking the sum of the products of
    weights and values, given a weight limit.
    ------------------------------------------------------
    INPUT:
        chromosome: (numpy.ndarray),

    OUTPUT:
        fitness of chromosome: (int)
    """
    weights = [weight_value[gene][0] * chromosome[gene] for gene in range(len(chromosome))]
    values = [weight_value[gene][1] * chromosome[gene] for gene in range(len(chromosome))]
    
    total_weight = sum(weights)
    total_value = sum(values)

    # Make sure total weight doesn't exceed knapsack capacity
#        
    if total_weight > capacity:
        return 0

    return total_value

def fitness_func2(chromosome):
    """
    Alternative fitness function, which penalizes the fitness of chromosome
    that exceeds the capacity.
    --------------------------------------------------------
    """ 
    weights = [weight_value[gene][0] * chromosome[gene] for gene in range(len(chromosome))]
    values = [weight_value[gene][1] * chromosome[gene] for gene in range(len(chromosome))]
    total_weight = sum(weights)
    total_value = sum(values)
    
    # For penatly calculation
    mean = np.mean(values)
    std_dev = np.std(values)
    penalty_factor = random.uniform(0, 0.1) # Randomly selected number between 0 and .1
    
    # Make sure total weight doesn't exceed knapsack capacity
    if total_weight > capacity:
        return total_value - penalty_factor * (total_weight - capacity) # Penatly

    return total_value

def compare_fitness_functions(population):
    """
    Compares the two fitness functions via average fitness and the best
    solution.
    ------------------------------------------------
    """
    fit1 = [round(fitness_func(chromosome)) for chromosome in population]
    fit2 = [round(fitness_func2(chromosome)) for chromosome in population]
    avg_fitness1 = round(np.mean(fit1))
    avg_fitness2 = round(np.mean(fit2))
    best_solution1 = round(max(fit1))
    best_solution2 = round(max(fit2))
    
    print("""
=======================================================
COMPARING BOTH FITNESS FUNCTIONS:
--------------------------------
            """, end='')
    print(dedent(f"""
Average fitness function 1: {avg_fitness1}
Average fitness function 2: {avg_fitness2}
Best solution1: {best_solution1}
Best solution2: {best_solution2}
======================================================= 
            """))

    roulette_parents = roulette_selection(population)
    tournament_parents = tournament_selection(population)
    roulette_fitness = [fitness_func(roulette_parents[i]) for i in
                        range(2)]
    tournament_fitness = [fitness_func(tournament_parents[i]) for i in
                            range(2)]
    roulette_fitness2 = [fitness_func2(roulette_parents[i]) for i in
                        range(2)]
    tournament_fitness2 = [fitness_func2(tournament_parents[i]) for i in
                            range(2)]

    # Output fitness of each fitness function
    print(f"Roulette fitness1 of parents: {roulette_fitness}")
    print(f"Tournament fitness1 of parents: {tournament_fitness}")
    print(f"Roulette fitness2 of parents: {roulette_fitness2}")
    print(f"Tournament fitness2 of parents: {tournament_fitness2}")
    # Output parents for each fitness function and seleciton method
#        print(f"Roulette: {roulette_parents}")
#        print(f"Tournament: {tournament_parents}")

def roulette_selection(population):
    """
    Fitness values of all individuals in the community are summed and the probability of each individual being selected is the ratio of the fitness value to that total value. The better the chromosomes, the more likely they are to be selected.
    ----------------------------------------------------------------
    INPUT:
        population: (np.ndarray)

    OUTPUT:
        parents: (list) Two (np.ndarray) fittest members of the population.
    """
    # Get fitness values in a list from all chromosomes
    fitness_values = [fitness_func(chromosome) for chromosome in population]
    total_fitness = sum(fitness_values)
    
    selected_parents = []
    # Loop thru twice to get the two parents
    for _ in range(2):
        # Gets random number from 0 to total_fitness, which is used to
        # determine a thresshold from which to select a parent
        rand_num = np.random.rand() * total_fitness
        partial_sum = 0
        for chromosome in population:
            # Iterate to add to partial sum 
            partial_sum += fitness_func(chromosome)
            
            # If partial sum is greater the random number, add parent
            if partial_sum >= rand_num:
                selected_parents.append(chromosome)
                break

    return selected_parents

def tournament_selection(population, k=3):
    """
    A random sequence is selected from the entire generation at a time. They will be in the tournament. In a tournament, whichever is better wins. Tournaments continue until the match pool is full. The random selection of chromosomes creates the possibility of co-selection of chromosomes with a low fitness value. In this case, chromosomes can be selected for the next generation, which is better than the rest of the population, even if they are mediocre. Thus, diversity in the population can be maintained.
    ----------------------------------------------------------------
    INPUT:
        population: (np.ndarray)

    OUTPUT:
        parents: (list of np.ndarrays)
    """
    selected_parents = []
    
    for _ in range(2):
        # Randomly select k individuals from population
        chromosomes = random.sample(list(population), k)
        # Most fit among k chromosomes
        best_fit = 0.37 # Initialize best fitness
        best_chrome = None
        
        for chrome in chromosomes:
            fit = fitness_func(chrome)

            if fit > best_fit:
                best_fit = fit
                best_chromosome = chrome
        
        selected_parents.append(best_chromosome)

    return selected_parents

def compare_selection_methods(population):
    """
    Compares the two selection methods via average fitness and the best.
    -------------------------------------------------------------------
    INPUT:
        population: (np.ndarray)
        
    OUTPUT:
        None
    """
    methods = ['roulette', 'tournament']
    method_stats = {}

    for method in methods:
        avg_fitness_list, best_fitness_list, best_active_genes_list = [], [], []
        
        for gen in range(stop):
            if method == "roulette":
                parents = roulette_selection(population)

            elif method == "tournament":
                parents = tournament_selection(population)

            # Compute statistics
            avg_fitness = np.mean([fitness_func(chrome) for chrome in population])
            best_solution = max(population, key=fitness_func)
            best_fitness = fitness_func(best_solution)
            best_active_genes = sum(best_solution)

            # Update lists
            avg_fitness_list.append(avg_fitness)
            best_fitness_list.append(best_fitness)
            best_active_genes_list.append(best_active_genes)
            
            # Update population
            population = update_population(population, parents)

        # Store method statistics for later comparison
        method_stats[method] = {
            'avg_fitness': avg_fitness_list,
            'best_fitness': best_fitness_list,
            'best_active_genes': best_active_genes_list
        }            
    # Plot stats
    plot_stats()

def plot_stats():
    """
    PLots statistics for each selection method.
    """
    plt.figure()
    
    for idx, method in enumerate(method_stats.keys()):
        stats = method_stats[method]
        
        # Intialize generations
        generations = range(len(stats['avg_fitness'])) # Must match with length of data you plot

        # Plot average population fitness per generation
        plt.subplot(3, 1, 1)
        plt.plot(generations, generation_stats['avg_fitness'])
        plt.title('Average Population Fitness per Generation')

        # Plot fitness score and number of active genes of fittest individual per generation
        plt.subplot(3, 1, 2)
        plt.plot(generations, generation_stats['best_fitness'], label='Best Fitness')
        plt.plot(generations, generation_stats['best_active_genes'], label='Best Active Genes')
        plt.legend()
        plt.title('Best Individual Stats per Generation')

        # Report number of active genes and fitness for the best solution overall
        best_generation = generation_stats['best_fitness'].index(max(generation_stats['best_fitness']))
        plt.subplot(3, 1, 3)
        plt.scatter([best_generation], [generation_stats['best_fitness'][best_generation]], label='Best Fitness')
        plt.scatter([best_generation], [generation_stats['best_active_genes'][best_generation]], label='Best Active Genes')
        plt.legend()
        plt.title('Best Solution Overall')

    plt.tight_layout()
    plt.show()
    
def get_weakest(population, num_weakest=2):
    """
    Finds the weakest members of a population.
    ------------------------------------------------------------------------
    INPUT:
        population: (np.ndarray)
        num_weakest: (int) Number of weakest individuals to find.

    OUTPUT:
        weakest: (list) Indices of the weakest individuals in the population.
    """
    return sorted(range(len(population)), key=lambda i: fitness_func(population[i]))[:num_weakest]
    
def update_population(population, children):
    """
    Updates population by replacing the worst members by the new children.
    ----------------------------------------------------------------------
    INPUT:
        population: (np.ndarray)
        children: (list) New children to be added to the population.

    OUTPUT:
        Updated population: (np.ndarray)
    """
    weakest_indices = get_weakest(population, len(children))
    for i, child in zip(weakest_indices, children):
        population[i] = child

    return population

breakpoint()