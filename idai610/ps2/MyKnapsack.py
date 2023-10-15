#!/usr/bin/env python
import numpy as np
import random
import matplotlib.pyplot as plt
from textwrap import dedent

seed = 1470
print(f"\t\t...\t...\t...Random seed used: {seed}...\t...\t...")
np.random.seed(seed)


class MyKnapsack(object):
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

        total_weight, total_value = 0, 0
        penalty_factor = 0.5

        for i in range(len(chromosome)):
            total_weight += self.weight_value[i][0] * chromosome[i]
            total_value += self.weight_value[i][1] * chromosome[i]

        if alt:
            # Apply penalty if weight limit is exceeded
            if total_weight > self.capacity:
                return total_value - penalty_factor * (total_weight - self.capacity)

        else:
            # Alternative method for dealing weight problems
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

    def tournament_selection(self, initial_population, tournament_size=2):
        """
        Tournaments run among a few chromosomes, chosen at random from
        population.
        --------------------------------------------------------------
        INPUT:
            initial_population: (np.ndarray)
            tournament_size: (int) number of competitors in tournament
                (default: 2)

        OUTPUT:
            two parents: (list) the two chromosomes for the two most fit
            parents
        """
        # Randomly selects chromosomes from population
        try:
            competitors = random.sample(list(initial_population), tournament_size)

        except ValueError:
            print(
dedent(f"""Check the size of input population.\nIt must not exceed the
  tournament size of {tournament_size}.""")
            )
        fit_vals = [self.fitness_func(chromosome) for chromosome in competitors]
        # Sort in descending order
        sorted_indices = np.argsort(fit_vals)[::-1]
        # Select most fit parents
        parent1 = competitors[sorted_indices[0]]
        parent2 = competitors[sorted_indices[1]]

        return [parent1, parent2]

    def selection(self, population, roulette=True):
        """
        Uses Roulette selection by default
        -----------------------------------
        INPUT:
            population: (np.ndarray) current population,
            routlette: (bool) Whether to use Roulette or Tournament selection
            (Default: True)

        OUTPUT:
            returns selected parents: (list of two np.ndarrays)
        """
        if roulette:
            return self.roulette_selection(population)

        else:
            return self.tournament_selection(population)

    def crossover(self, parents):
        """
        INPUT:
            parents: (list of np.ndarrays)

        OUTPUT:
            children: (tuple)
        """
        # Point crossover using a random integer for the split
        point = random.randint(1, len(parents[0])-1)
        # For the children; stacks arrays in sequence, horizontally
        # (Column-wise)
        child1 = np.hstack((parents[0][:point], parents[1][point:]))
        child2 = np.hstack((parents[1][:point], parents[0][point:]))

        return child1, child2

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

    def the_fittest(self, population, compare_fit_funcs=False):
        """
        Determines the fittest individual for a given population
        --------------------------------------------------------
        INPUT:
            population: (np.ndarray),
            compare_fit_funcs: (bool) If True, both fitness functions are
            compared.  (default: False)

        OUTPUT:
            max_chromosome, max_fitness, active_genes: (tuple: (np.ndarray),
            (np.int64), (np.int64))
        """
        if compare_fit_funcs:
            # compare fitness functions FUNCTION goes here
            pass

        fittest_values = [self.fitness_func(chromosome) for chromosome in
                          population]
        max_fitness = max(fittest_values)
        max_index = fittest_values.index(max_fitness)
        active_genes = sum(population[max_index])

        return population[max_index], max_fitness, active_genes

    def fitness_plot(self, 
                       avg_fitness_data, 
                       the_fittest_data,
                       best_generation,
                       best_active_genes, 
                       roulette=True):
        """
        Plots and compares the two different selection methods,
        displaying average, most fit and active genes
        ------------------------------------------------------
        INPUT:
            avg_fitness_data: (list of floats)
            the_fittest_data: (list of tuples)
            roulette: (bool) True by default, depending on which Selection
            method you choose to use

        OUTPUT:
            Plots graphs: (None)

        """
        plt.figure(figsize=(12, 6))  # Set the figure size

        # Plot for Average Fitness Data
        plt.subplot(3, 1, 1)
        plt.plot(avg_fitness_data, marker='o', linestyle='-')
        plt.xlabel(f'Generation: (Best Generation: {best_generation})')
        plt.ylabel('Average Fitness')
        plt.grid(True)

        if roulette:
            plt.title("Average Fitness per Generation: Roulette")
        else:
            plt.title("Average Fitness per Generation: Tournament")

        # Plot for The Fittest Data
        plt.subplot(3, 1, 2)
        fittest, active_genes = zip(*the_fittest_data)
        plt.plot(fittest, label="The Fittest", marker='x', linestyle='-')
        plt.xlabel(f'Generation: (Best Generation: {best_generation})')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)

        # Title based on the selection method
        if roulette:
            plt.title("The Fittest per Generation: Roulette")
        else:
            plt.title("The Fittest per Generation: Tournament")

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
        if roulette:
            plt.title("Active Genes per Generation: Roulette")
        else:
            plt.title("Active Genes per Generation: Tournament")

        plt.tight_layout()
        plt.show()

    def run_genetic(self, roulette=True, selection_only=False):
        """
        Runs the genetic algorithm, obtaining the best solution,
        most fit individual and active genese for each generation.
        ---------------------------------------------------------
        INPUT:
            roulette: (bool) Use roulette or tournament selection
                (default: True for Roulette Selection)
            selection_only: (bool) Whether to compare performances for the two
            selection methods only (default: False)

        OUTPUT:
            None
        """
        # Initialized average fitness data and fittest individual data
        avg_fitness_data = []
        the_fittest_data = []

        best_solution = None
        best_fitness = 0
        best_active_genes = 0
        best_generation = self.generation

        # For each generation, run genetic algorithm
        for gen in range(self.stop):
            # Population initialization for updates
            new_population = []

            # Run genetic algorithm with Roulette Selection as default
            if roulette:
                parents = self.selection(self.population)

            else:
                parents = self.selection(self.population, False)

            # For QUESTION 2: Selection only for performance comparison
            if not selection_only:
                children = self.crossover(parents)
                mutants = [self.mutation(kid) for kid in children]

                # Update population
                new_population.extend(mutants)

            else:
                new_population.extend(parents)
                # Using subset of existing population to the new population
                remaining_population_size = \
                len(self.population) - len(new_population)
                remaining_population = random.sample(list(self.population),
                                                      remaining_population_size)
                new_population.extend(remaining_population)

            avg_fitness = self.average_fitness(self.population) # (float)
            # Gets numpy array of best chromosome, most_fit individual (int)
            # and the number of active genes in chromosome (int)
            fittest_chromosome, most_fit, active_genes = self.the_fittest(
                self.population
            )

            avg_fitness_data.append(avg_fitness) # List of floats
            the_fittest_data.append((most_fit, active_genes)) # List of Tuples

            # Update best solution based on whether 
            if most_fit > best_fitness:
                best_solution = fittest_chromosome
                best_fitness = most_fit
                best_generation = gen
                best_active_genes = active_genes

            # New population
            self.population = np.array(new_population)

        # Plotting
        self.fitness_plot(
            avg_fitness_data, 
            the_fittest_data,
            best_generation,
            best_active_genes, 
            roulette
        )

        # Output results to console
        if roulette:
            print(f"++++++++++++++++ ROULETTE ++++++++++++++++++++++++++++++++")
            print(f"Best fitness overall: {best_fitness}")
            print(f"Best solution overall: {best_solution}")
            print(f"In generation: {best_generation}")
            print(f"Number of active genes for most fit: {best_active_genes}")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        else:
            print(f"++++++++++++++++ TOURNAMENT ++++++++++++++++++++++++++++++++")
            print(f"Best fitness overall: {best_fitness}")
            print(f"Best solution overall: {best_solution}")
            print(f"In generation: {best_generation}")
            print(f"Number of active genes for most fit: {best_active_genes}")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# ================================ Selection performance comparison ================================


if __name__ == "__main__":

    for i in range(1,3):
        sack = MyKnapsack(f"config_{i}.txt")
#        sack.run_genetic()
#        sack.run_genetic(roulette=False)
        # Question 2:
        sack.run_genetic(selection_only=True)
        sack.run_genetic(roulette=False, selection_only=True)


