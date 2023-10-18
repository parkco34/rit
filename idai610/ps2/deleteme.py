#!/usr/bin/env python
# Import required libraries
from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt

class SelectionStrategy(Enum):
    # Constants
    ROULETTE = "roulette"
    TOURNAMENT = "tournament"
    COMPARE = "compare"
    INTEGRATE = "integrate"
    EXPLORE = "explore"

# Modify the MySack class
class MySack():
    """
    Genetic Algorithm for the 0-1 Knapsack problem
    """
    def __init__(self, config_file):
        self.config_file = config_file
        (self.population,
        self.capacity,
        self.weight_value,
        self.generation,
        self.stop) = self.get_initial_population()

        # Add a variable to store the best solution throughout the experiment
        self.best_solution = {
            'chromosome': None,
            'fitness': -1,
            'active_genes': 0,
            'generation': -1
        }

        # Add a variable to store statistics
        self.stats = {
            'avg_fitness': [],
            'best_fitness': [],
            'best_active_genes': []
        }

    def get_initial_population(self):
        """
        Generates initial population for generation 0
        ----------------------------------------------
        INPUT:
            _

        OUTPUT:
            g: (int) current generation
            chromosomes :(matrix or 2D array) Population of individual
            chromosomes
            W:(int) Knapsack capacity
            S: (list of tuples)Each tuple is an item (w_i, v_1)
            stop:(int) Final generation (stop condition)
        """
        try:
            with open(self.config_file, "r") as file:
                lines = file.readlines()

        except FileNotFoundError:
            print(f"{self.config_file} not found... ¯\_( ͡° ͜ʖ ͡°)_/¯ ")

        except Exception as e:
            print(f"Something went wrong!   ¯\_( ͡° ͜ʖ ͡°)_/¯  \n{e}\n")

        pop_size, n, stop, W = map(int, [lines[i].strip() for i in
                                                range(4)])
        S = [tuple(map(int, line.strip().split())) for line in lines[4:]]
        # Initialize empty population
        population = np.random.randint(2, size=(pop_size, n))

        g = 0 # initial generation
        return population, W, S, g, stop

    def get_weight(self, chromosome):
        """
        Gets total weight of chromosome.
        ---------------------------------
        INPUT:
            chromosome: (np.ndarray)

        OUTPUT:
            total weight: (int)
        """
        return sum(self.weight_value[gene][0] for gene in range(len(chromosome)))

    def get_value(self, chromosome):
        """
        Gets total value of chromosome.
        ---------------------------------
        INPUT:
            chromosome: (np.ndarray)

        OUTPUT:
            total value: (int)
        """
        return sum(self.weight_value[gene][1] for gene in range(len(chromosome)))

    def fitness_func(self, chromosome, alt=False):
        """
        Determines fitness of chromosome by taking the sum of the products of
        weights and values, given a weight limit.
        ------------------------------------------------------
        INPUT:
            chromosome: (numpy.ndarray),
            alt: (bool) When True, fitness function penalizes chromosomes whose
            weights exceed the weight capacity

        OUTPUT:
            fitness of chromosome: (int)
        """
        total_weight = self.get_weight(chromosome)
        total_value = self.get_value(chromosome)
        penalty_factor = 0.5

        if alt:
            # Alternative fitness function
            if total_weight > self.capacity:
                return total_value - penalty_factor * (total_weight - self.capacity)

        else:
            if total_weight > self.capacity:
                return total_value

            return total_value

    def roulette_selection(self, initial_population):
        """
        Roulette Wheel Selection
        Samples from probability distribution of population size, where:
            p_s = fitness_func / sum of all fitness_funcs by obtaining a list
            of fitness function results for each chromosome, returning a
            list of most fit chromosomes
        -----------------------------------------------------------
        INPUT:
            initial_population: (np.ndarray)

        OUTPUT:
            two parents: (list) the two chromosomes for the two most fit parents
        """
        list_of_fitness = [self.fitness_func(chromosome) for chromosome in
                           initial_population] # Not sorted, if it matters
        total = sum(list_of_fitness)

        p_s = [] # List of Probabilities of solutions
        for fitness in list_of_fitness:
            # If total_weight is over the weight limit, select two random
            # parents, without weights

            if total == 0:
                return random.choices(initial_population, k=2)

            else:
                p_s.append(fitness / total)

        parent1, parent2 = random.choices(initial_population, weights=p_s, k=2)
        return [parent1, parent2]

    def tournament_selection(self, initial_population):
        """
        Tournaments run among a few chromosomes, chosen at random from
        population.
        --------------------------------------------------------------
        INPUT:
            initial_population: (np.ndarray)

        OUTPUT:
            two parents: (list) the two chromosomes for the two most fit
            parents
        """
        # Randomly selects chromosomes from population
        try:
            competitors = random.sample(list(initial_population), 2)

        except Exception as e:
            print(f"Something went wrong: {e}")

        fit_vals = [self.fitness_func(chromosome) for chromosome in competitors]
        # Sort in descending order
        sorted_indices = np.argsort(fit_vals)[::-1]
        # Select most fit parents
        parent1 = competitors[sorted_indices[0]]
        parent2 = competitors[sorted_indices[1]]

        return [parent1, parent2]

    def selection(self, initial_population, roulette=True):
        """
        Selection method for selecting parents.
        ----------------------------------------
        INPUT:
            initial_population: (np.ndarray)
            roulette: (bool) True if using roulette, tournament is used
            otherwise.
                default: True

        OUTPUT:
            parents: (list of np.ndarrays)
        """
        if roulette:
            return self.roulette_selection(initial_population)

        else:
            return self.tournament_selection(initial_population)

    def crossover(self, parents):
        """
        Single-point crossover possible at any chromosome index,
        where the crossover rate C_x, wich functions as a cutoff - if a random
        number is above it you perform crossover, otherwise you don't.
        ------------------------------------------
        INPUT:
            parents: (list of np.ndarrays)

        OUTPUT:
            children: (list of np.ndarrays)
        """
        # Generate random number between 0 and 1
        rand_number = random.uniform(0,1)

        # Define crossover rate C_x
        C_x = 0.473
        if rand_number > C_x:
            # Point crossover using a random integer for the split
            point = random.randint(1, len(parents[0])-1)
            # For the children; stacks arrays in sequence, horizontally
            # (Column-wise)
            child1 = np.hstack((parents[0][:point], parents[1][point:]))
            child2 = np.hstack((parents[1][:point], parents[0][point:]))
            return [child1, child2]

        else:
            return [parents[0], parents[1]]

    def mutation(self, chromosome):

        """
        Determines how often offspring have random mutations to their
        representation.  Once generated, offspring's bits are flipped with
        probability = M_r (mutation rate)
        -------------------------------------------------------------------
        INPUT:
            chromosome: (np.ndarray)

        OUTPUT:
            chromosome: (np.ndarray)
        """
        # Randomly select a mutation rate between 0.05 and 0.2 with an increment of 0.05
        Mr = random.choice([0.05, 0.1, 0.15, 0.2])

        # Iterate through genes
        for i in range(len(chromosome)):
            rand_num = random.uniform(0, 1)  # Generate random number

            # If the number is less than Mr, flip the bit!
            if rand_num < Mr:
                chromosome[i] = 1 - chromosome[i]

        return chromosome

    def average_fitness(self, population):
        """
        Calculates average fitness for a population
        --------------------------------------------------------
        INPUT:
            population:  (np.ndarray)

        OUTPUTS:
            total_fitness: (float) Average fitness for a given population
        """
        total_fitness = sum([self.fitness_func(chromosome) for chromosome in
                             population])

        return total_fitness / len(population)

    def the_fittest(self, population):
        """
        Determines the fittest individual for a given population
        --------------------------------------------------------
        INPUT:
            population: (np.ndarray),

        OUTPUT:
            max_chromosome, max_fitness, active_genes: (tuple: (np.ndarray),
            (np.int64), (np.int64))
        """
        fittest_values = [self.fitness_func(chromosome) for chromosome in
                          population]
        max_fitness = max(fittest_values)
        max_index = fittest_values.index(max_fitness)
        active_genes = sum(population[max_index])

        return population[max_index], max_fitness, active_genes

    def fitness_plot(self,
                    avg_fit_data,
                    the_fittest_data,
                    best_generation,
                     best_active_genes,
                     strategy=SelectionStrategy.ROULETTE
                    ):
        """
        Plots and compares the two different selection methods,
        displaying average, most fit and active genes
        ------------------------------------------------------
        INPUT:
            avg_fit_data: (list of floats)
            the_fittest_data: (list of tuples)
            roulette: (bool) True by default, depending on which Selection
            method you choose to use

        OUTPUT:
            Plots graphs: (None)

        """
        title = ""
        plt.figure(figsize=(12, 6))  # Set the figure size

        # Plot for Average Fitness Data
        plt.subplot(3, 1, 1)
        plt.plot(avg_fit_data, marker='o', linestyle='-')
        plt.xlabel(f'Generation: (Best Generation: {best_generation})')
        plt.ylabel('Average Fitness')
        plt.grid(True)

        plt.title(r"$\bf{" + f"{title}" + r"}$" + f"\nAverage Fitness per \
                  Generation: {strategy}")

        # Prevents empty list from being passed through
        if the_fittest_data is not None:
            fittest, active_genes = zip(*the_fittest_data)

        else:
            print("Warning: the_fittest_data is None")
            # Handle the case when the_fittest_data is None, maybe set fittest and active_genes to empty lists
            fittest = []
            active_genes = []

        # Plot for The Fittest Data
        plt.subplot(3, 1, 2)
        fittest, active_genes = zip(*the_fittest_data)
        plt.plot(fittest, label="The Fittest", marker='x', linestyle='-')
        plt.xlabel(f'Generation: (Best Generation: {best_generation})')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)

        # Title based on the selection method
        plt.title(r"$\bf{" + f"{title}" r"}$" + "\nThe Fittest per \
                  Generation: {strategy}")

        # Plot for Active Genes Data
        plt.subplot(3, 1, 3)
        plt.plot(active_genes, label=
                 f"Active Genes: (Best Count: {best_active_genes})",
                 marker='s',
                 linestyle='--')

        plt.xlabel(f'Generation: (Best Generation: {best_generation})')
        plt.ylabel(f'Active Genes')
        plt.legend()
        plt.grid(True)

        # Title based on the selection method
        plt.title(f"" + r"$\bf{" + f"{title}" r"}$" + "\nActive Genes per \
                  Generation: {strategy}")

        plt.tight_layout()
        plt.show()

    def check_diversity_loss(self, threshold=0.1):
        """
        My own stopping criteria.
        Stops algorithm if diversity is below a certain threshold.
        -----------------------------------------------------------------
        INPUT:
            threshold: (float) default: 0.1

        OUTPUT:
            stop_flag: (bool) True if diversity is below threshold, False
            otherwise
        """
        unique_chromosome = np.unique(self.population, axis=0)
        diversity_ratio = len(unique_chromosome) / len(self.population)

        if diversity_ratio < threshold:
            return True

        return False

    def integrate_crossover_mutation(self, population, stopping_criteria):
        """
        Running trials with each selection operator, reporting results on
        crossover and mutation.
            - Include comparative results for my implemented stopping criteria.
        --> Use plots method
        ---------------------------------------------------------
        INPUT:
            population: (np.ndarray)
            stopping_criteria: (int)
        """
        # Initialize data storage for plots
        avg_fit_data_roulette = []
        avg_fit_data_tournament = []

        # Run trials with Roulette selection
        for gen in range(stopping_criteria):
            parents_roulette = self.selection(population)
            children_roulette = self.crossover(parents_roulette)
            mutants_roulette = [self.mutation(kid) for kid in children_roulette]

            # Replace least fit individuals
            sorted_indices = np.argsort([self.fitness_func(ind) for ind in population])
            population[sorted_indices[:len(mutants_roulette)]] = mutants_roulette

            # Collect data for plotting
            avg_fitness_roulette = self.average_fitness(population)
            avg_fit_data_roulette.append(avg_fitness_roulette)

        # Reset population for next trial
        population = self.population

        # Run trials with Tournament selection
        for gen in range(stopping_criteria):
            parents_tournament = self.selection(population, roulette=False)
            children_tournament = self.crossover(parents_tournament)
            mutants_tournament = [self.mutation(kid) for kid in children_tournament]

            # Replace least fit individuals
            sorted_indices = np.argsort([self.fitness_func(ind) for ind in population])
            population[sorted_indices[:len(mutants_tournament)]] = mutants_tournament

            # Collect data for plotting
            avg_fitness_tournament = self.average_fitness(population)
            avg_fit_data_tournament.append(avg_fitness_tournament)

        # Use your existing plot method to visualize the results
        self.fitness_plot(avg_fit_data_roulette, None, None, None, True)
        self.fitness_plot(avg_fit_data_tournament, None, None, None, False)

    def explore_pop_sizes(self, population):
        """
        Using different population sizes,
        1) Explore at least 10 population sizes,
        reporting on the total_value, weights and number of included items
        (active genes) at the stopping criterion.
        2) Run 30 trials and report mean weight, std dev.
        -----------------------------------------------------------
        INPUT:
            population: (np.ndarray)

        """
        weight_value = self.weight_value
        # Generates a list of 10 random integers between 13 and 73
        pop_sizez = np.random.randint(13, 74, size=10).array_tolist()

        for pop_size in pop_sizes:
            print(f"Running trials for population size: {pop_size}")

            # Initialize population based on the new size
            population = np.random.randint(2, size=(pop_size, len(weight_value)))

            final_values = []
            final_weights = []
            final_no_items = []

            self.run_trials(num_trials=30)

            # Get perfromance
            best_chromosome = self.the_fittest(population)[0]
            best_fitness = self.fitness_func(best_chromosome)
            best_weight = self.get_weight(best_chromosome)
            no_items = self.get_active_items(best_chromosome)

            final_values.append(best_fitness)
            final_weights.append(best_weight)
            final_no_items.append(no_items)

            print("Values: {} ± {}".format(np.mean(final_values), np.std(final_values)))
            print("Weight: {}".format(np.max(final_weights)))
            print("No of items: {}".format(np.mean(final_no_items)))

    def apply_strategy(self, strategy):
        """
        Applie given strategy to produce a new population.
        """
        new_population = []

        if strategy == SelectionStrategy.ROULETTE:
            new_population = self.apply_roulette_strategy()

        elif strategy == SelectionStrategy.TOURNAMENT:
            new_population = self.apply_tournament_strategy()

        elif strategy == SelectionStrategy.COMPARE:
            new_population = self.apply_compare_strategy()
        
        elif strategy == SelectionStrategy.INTEGRATE:
            new_population = self.apply_integrate_strategy()

        elif strategy == SelectionStrategy.EXPLORE:
            new_population = self.apply_explore_strategy()

        return new_population

    def apply_roulette_strategy(self):
        """
        Selection method of Roulette to obtain progeny.
        """
        new_population = []
        for _ in range(len(self.population) // 2):  # Looping half the number of times because each loop produces two children
            parents = self.selection(self.population)
            children = self.crossover(parents)
            mutants = [self.mutation(kid) for kid in children]
            new_population.extend(mutants)
        return new_population

    def apply_tournament_strategy(self):
        """
        Tournament Selection.
        """
        new_population = []
        for _ in range(len(self.population) // 2):  # Looping half the number of times because each loop produces two children
            parents = self.selection(self.population, roulette=False)
            children = self.crossover(parents)
            mutants = [self.mutation(kid) for kid in children]
            new_population.extend(mutants)
        return new_population

    def apply_compare_strategy(self):
        """
        Compares the Roulette and Tournament selection methods by creating two
        new populations using each method.
        """
        # Population generated using Roulette
        parents_roulette = self.selection(self.population)
        children_roulette = self.crossover(parents_roulette)
        mutants_roulette = [self.mutation(kid) for kid in children_roulette]

        # Population generated using Tournament
        parents_tournament = self.selection(self.population, roulette=False)
        children_tournament = self.crossover(parents_tournament)
        mutants_tournament = [self.mutation(kid) for kid in children_tournament]

        return mutants_roulette, mutants_tournament

    def apply_explore_strategy(self):
        """
        Explores different population sizes to analyze performance.
        """
        # Define a list of population sizes to explore
        pop_sizes = [20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]

        # Dictionary to store performance metrics for each population size
        performance_metrics = {}

        for size in pop_sizes:
            # Generate a new random population based on the new size
            new_population = np.random.randint(2, size=(size, self.population.shape[1]))

            # Initialize metrics storage for this population size
            avg_fitness_list = []
            max_fitness_list = []
            active_genes_list = []

            # Run genetic algorithm for a certain number of generations
            for gen in range(self.stop):
                parents = self.selection(new_population)
                children = self.crossover(parents)
                mutants = [self.mutation(kid) for kid in children]

                # Replace least fit individuals
                sorted_indices = np.argsort([self.fitness_func(ind) for ind in new_population])
                new_population[sorted_indices[:len(mutants)]] = mutants

                # Collect performance metrics
                avg_fitness = self.average_fitness(new_population)
                _, max_fitness, active_genes = self.the_fittest(new_population)

                avg_fitness_list.append(avg_fitness)
                max_fitness_list.append(max_fitness)
                active_genes_list.append(active_genes)

            # Store the metrics for this population size
            performance_metrics[size] = {
                'Average Fitness': avg_fitness_list,
                'Max Fitness': max_fitness_list,
                'Active Genes': active_genes_list
            }

        return performance_metrics

    def update_population(self, new_population):
        """
        Updates current population with the new one.
        """
        combined_population = np.vstack([self.population, new_population])
        sorted_indices = np.argsort([self.fitness_func(ind) for ind in combined_population])[::-1]
        self.population = combined_population[sorted_indices[:len(self.population)]]

    def collect_statistics(self):
        """
        Collects and storess the statistics.
        """
        avg_fitness = self.average_fitness(self.population)

    # Add a new method to update the best solution
    def update_best_solution(self, chromosome, fitness, active_genes, generation):
        if fitness > self.best_solution['fitness']:
            self.best_solution = {
                'chromosome': chromosome,
                'fitness': fitness,
                'active_genes': active_genes,
                'generation': generation
            }

    def run_genetic(self, strategy=SelectionStrategy.ROULETTE):
        """
        Runs the genetic algorithm for each generation, creating a new
        population with each iteration.
        """
        # Initialize variables to keep track of best solution and stats
        avg_fit_data = []  # List to store average fitness per generation
        the_fittest_data = []  # List to store best fitness and active genes per generation

        # Handle the 'EXPLORE' strategy separately
        if strategy == SelectionStrategy.EXPLORE:
            performance_metrics = self.apply_explore_strategy()
        else:
            # Loop through each generation
            for gen in range(self.stop):
                # Check for diversity loss and break if necessary (Extra Credit)
                if self.check_diversity_loss():
                    print("Stopping due to loss of diversity.")
                    break

                # Apply either the ROULETTE or TOURNAMENT strategy
                if strategy in (SelectionStrategy.ROULETTE, SelectionStrategy.TOURNAMENT):
                    new_population = self.apply_strategy(strategy)
                    self.update_population(new_population)

                    # Collect statistics for this generation
                    avg_fitness = self.average_fitness(self.population)
                    avg_fit_data.append(avg_fitness)  # Store average fitness

                    chromosome, max_fitness, active_genes = self.the_fittest(self.population)
                    breakpoint()
                    the_fittest_data.append((max_fitness, active_genes))  # Store the fittest data

                    # Update the best solution if a better one is found
                    self.update_best_solution(chromosome, max_fitness, active_genes, gen)

            # Plot the data (Uncomment the line if you want to see the plot)
            self.fitness_plot(
                avg_fit_data, 
                the_fittest_data, 
                self.best_solution['generation'], 
                self.best_solution['active_genes'], 
                strategy
            )


if __name__ == "__main__":
    ga = MySack("config_1.txt")
    ga.run_genetic()

