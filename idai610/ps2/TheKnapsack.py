#!/usr/bin/env python

"""
+++++++++++++++++++++++++++++++++ KNAPSACK PROBLEM +++++++++++++++++++++++++++++++++

"""
import numpy as np
import matplotlib.pyplot as plt
import random
from textwrap import dedent


class TheKnapsack(object):
    """
    Genetic Algorithm for the 0-1 Knapsack problem
    """
    def __init__(self, config_file, enable_crossover=True):
        self.config_file = config_file
        self.enable_crossover = enable_crossover

        # Initialize stats
        self.generation_stats = {
            "avg_fitness": [], "best_fit": [], "best_active_genes": []
        }
        self.best_solution = {'fitness': -1, 'active_genes': 0, 'generation': -1}

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

    def get_overview_of_data(self):
        """
        Plots overview of data via weights and values for comparison.
        """
        weights = [self.weight_value[gene][0] for gene in
                   range(len(self.weight_value))]
        values = [self.weight_value[gene][1] for gene in
                  range(len(self.weight_value))]
        print(f"Mean (weights): {np.mean(weights)}")
        print(f"Mean (values): {np.mean(values)}")
        print(f"Std Dev (weights): {np.std(weights)}")
        print(f"Std Dev (values): {np.std(values)}")
        print(f"Max weight: {max(weights)}")
        print(f"Max value: {max(values)}")
        
        print(f"Weights: {weights}\nValues: {values}\n")
        plt.scatter(weights, values)
        plt.xlabel("Weights")
        plt.ylabel("values")
        plt.title("Weights vs. Values for given population")
        plt.show()

    def fitness_func(self, chromosome):
        """
        Determines fitness of chromosome by taking the sum of the products of
        weights and values, given a weight limit.
        ------------------------------------------------------
        INPUT:
            chromosome: (numpy.ndarray),

        OUTPUT:
            fitness of chromosome: (int)
        """
        weights = [self.weight_value[gene][0] for gene in
                   range(len(self.weight_value))]
        values = [self.weight_value[gene][1] for gene in
                  range(len(self.weight_value))]
        total_weight = sum(weights)
        total_value = sum(values)
        
        # Make sure total weight doesn't exceed knapsack capacity
        if total_weight > self.capacity:
            return total_value

        return total_value

    def fitness_func2(self, chromosome):
        """
        Alternative fitness function, which penalizes the fitness of chromosome
        that exceeds the capacity.
        --------------------------------------------------------
        """ 
        weights = [self.weight_value[gene][0] for gene in
                   range(len(self.weight_value))]
        values = [self.weight_value[gene][1] for gene in
                  range(len(self.weight_value))]
        total_weight = sum(weights)
        total_value = sum(values)
        
        # For penatly calculation
        mean = np.mean(values)
        std_dev = np.std(values)
        penalty_factor = random.uniform(0, 0.1) # Randomly selected number between 0 and .1
        
        # Make sure total weight doesn't exceed knapsack capacity
        if total_weight > self.capacity:
            return total_value - penalty_factor * (total_weight - self.capacity) # Penatly

        return total_value

    def compare_fitness_functions(self):
        """
        Compares the two fitness functions via average fitness and the best
        solution.
        ------------------------------------------------
        """
        fit1 = [round(self.fitness_func(chromosome)) for chromosome in self.population]
        fit2 = [round(self.fitness_func2(chromosome)) for chromosome in self.population]
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

        roulette_parents = self.roulette_selection(self.population)
        tournament_parents = self.tournament_selection(self.population)
        roulette_fitness = [self.fitness_func(roulette_parents[i]) for i in
                            range(2)]
        tournament_fitness = [self.fitness_func(tournament_parents[i]) for i in
                             range(2)]
        roulette_fitness2 = [self.fitness_func2(roulette_parents[i]) for i in
                            range(2)]
        tournament_fitness2 = [self.fitness_func2(tournament_parents[i]) for i in
                             range(2)]
   
        # Output fitness of each fitness function
        print(f"Roulette fitness1 of parents: {roulette_fitness}")
        print(f"Tournament fitness1 of parents: {tournament_fitness}")
        print(f"Roulette fitness2 of parents: {roulette_fitness2}")
        print(f"Tournament fitness2 of parents: {tournament_fitness2}")
        # Output parents for each fitness function and seleciton method
#        print(f"Roulette: {roulette_parents}")
#        print(f"Tournament: {tournament_parents}")
    
    def roulette_selection(self, population):
        """
        Fitness values of all individuals in the community are summed and the probability of each individual being selected is the ratio of the fitness value to that total value. The better the chromosomes, the more likely they are to be selected.
        ----------------------------------------------------------------
        INPUT:
            population: (np.ndarray)

        OUTPUT:

        """
        # Get fitness values in a list from all chromosomes
        fitness_values = [self.fitness_func(chromosome) for chromosome in population]
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
                partial_sum += self.fitness_func(chromosome)
                
                # If partial sum is greater the random number, add parent
                if partial_sum >= rand_num:
                    selected_parents.append(chromosome)
                    break

        return selected_parents

    def tournament_selection(self, population, k=3):
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
            best_fit = -1 # Initialize best fitness
            best_chrome = None
            
            for chrome in chromosomes:
                fit = self.fitness_func(chrome)

                if fit > best_fit:
                    best_fit = fit
                    best_chromosome = chrome
            
            selected_parents.append(best_chromosome)

        return selected_parents

    def compare_selections(self):
        """
        Compares selection methods and produces a plot with:
            a) Average population fitness
        """
        pass

    def plot_statistics(self):
        # Initialize number of generations
        generations = range(self.num_generations)

        plt.figure()

        # Plot average population fitness per generation
        plt.subplot(3, 1, 1)
        plt.plot(generations, self.generation_stats['avg_fitness'])
        plt.title('Average Population Fitness per Generation')

        # Plot fitness score and number of active genes of fittest individual per generation
        plt.subplot(3, 1, 2)
        plt.plot(generations, self.generation_stats['best_fitness'], label='Best Fitness')
        plt.plot(generations, self.generation_stats['best_active_genes'], label='Best Active Genes')
        plt.legend()
        plt.title('Best Individual Stats per Generation')

        # Report number of active genes and fitness for the best solution overall
        best_generation = self.generation_stats['best_fitness'].index(max(self.generation_stats['best_fitness']))
        plt.subplot(3, 1, 3)
        plt.scatter([best_generation], [self.generation_stats['best_fitness'][best_generation]], label='Best Fitness')
        plt.scatter([best_generation], [self.generation_stats['best_active_genes'][best_generation]], label='Best Active Genes')
        plt.legend()
        plt.title('Best Solution Overall')

        plt.tight_layout()
        plt.show()

    def run(self, selection="roulette"):
        """
        Runs overall genetic algorithm.
        -------------------------------------------
        INPUT:
            selection_only (optional): (bool) Whether to run crossover and
            mutation or not: default False

        OUTPUT:
            None
        """
        population = self.population

        for gen in range(self.stop):
            # Selection
            if selection == "roulette":
                self.roulette_selection(population)

            elif seleciton == "tournament":
                self.tournament_selection(population)

            # Crossover and mutation enabled or disabled
            if self.enable_crossover:
                pass

            # Compute and log statistics
            avg_fitness = np.mean([self.fitness_func(chrome) for chrome in
                                  population])
            best_solution = max(population, key=self.fitness_func)
            best_fitness = self.fitness_func(best_solution)
            best_active_genes = sum(best_solution)

           # Log for this generation
            self.generation_stats['avg_fitness'].append(avg_fitness)
            self.generation_stats['best_fitness'].append(best_fitness)
            self.generation_stats['best_active_genes'].append(best_active_genes)


if __name__ == "__main__":
    ga = TheKnapsack("config_2.txt")
    ga.run()





