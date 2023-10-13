#!/usr/bin/env python
import numpy as np
import random
import matplotlib.pyplot as plt

seed = 1470
print(f"Random seed used: {seed}")
np.random.seed(seed)


class MySack(object):
    """
    Genetic Algorithm for the 0-1 Knapsack problem
    """
    def __init__(self, config_file):
        (self.population, 
        self.capacity, 
        self.weight_value, 
        self.generation,
        self.stop) = self.get_initial_population(config_file)

    def get_initial_population(self, config_file):
        """
        Generates initial population for generation 0
        ----------------------------------------------
        INPUT:
            configuration file: (str) 

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

    def fitness_func(self, chromosome):
        """
        Determines fitness of chromosome by taking the sum of the products of
        weights and values, given a weight limit.
        ------------------------------------------------------
        INPUT:
            chromosome: (numpy.ndarray)

        OUTPUT:
            fitness of chromosome: (int)
        """
        total_weight, total_value = 0, 0

        for i in range(len(chromosome)):
            total_weight += self.weight_value[i][0] * chromosome[i]
            total_value += self.weight_value[i][1] * chromosome[i]

            # Esnure weight limit isn't exceeded
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

        OUTPUT:
            two parents: (tuple) the two chromosomes for the two most fit parents           
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

        return parent1, parent2

    def tournament_selection(self, initial_population, tournament_size=3):
        """
        Tournaments run among a few chromosomes, chosen at random from
        population.
        --------------------------------------------------------------
        OUTPUT:
            two parents: (tuple) the two chromosomes for the two most fit
            parents
        """
        # Randomly selects chromosomes from population
        competitors = random.sample(list(initial_population), tournament_size)
        fit_vals = [self.fitness_func(chromosome) for chromosome in competitors]
        # Sort in descending order
        sorted_indices = np.argsort(fit_vals)[::-1] 
        # Select most fit parents
        parent1 = competitors[sorted_indices[0]]
        parent2 = competitors[sorted_indices[1]]

        return parent1, parent2

    def selection(self, population, roulette=True):
        """
        Uses Roulette selection by default
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
        # Mutation rate
        Mr = 1 / len(chromosome)

        # Iterate thru genes
        for i in range(len(chromosome)):
            rand_num = random.uniform(0, 1) # Generate random number

            # If number is less than Mr, flip!
            if rand_num < Mr:
                chromosome[i] = 1 - chromosome[i]
        
        return chromosome

    def plot_selection(self, avg_fitness_data, the_fittest_data):
        """
        Plots and compares the two different selection methods

        """
        plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(avg_fitness_data)
        plt.title("Average Fitness per Generation")

        plt.subplot(2, 1, 2)
        fittest_fit, active_genes = zip(*the_fittest_data)
        plt.plot(fittest_fit, label="The Fittest")
        plt.plot(active_genes, label="Active Genes")
        plt.legend()
        plt.title("The Fittest per Generation")

        plt.tight_layout()
        plt.show()

    def average_fitness(self, population):
        """
        Calculates average fitness for a population
        """
        total_fitness = sum([self.fitness_func(chromosome) for chromosome in
                             population])
        return total_fitness / len(population)

    def the_fittest(self, population):
        """
        Determines the fittest individual for a given population
        """
        fitness_values = [self.fitness_func(chromosome) for chromosome in
                          population]
        max_fitness = max(fitness_values)
        max_index = fitness_values.index(max_fitness)

        return population[max_index], max_fitness, sum(population[max_index])

        def fitness_plot(self):
            """
            FIx this shit!
            """
            # create figure
            plt.figure(figsize=(12, 6))

            # first subplot
            plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
            sns.lineplot(x=range(len(fitness_score1)), y=fitness_score1)
            plt.title(dedent(
            f"""fitness score for config_{config} file: roulette selection"""
            ))
            plt.xlabel("generation")
            plt.ylabel("fintess score")

            # create 2nd subplot
            plt.subplot(1, 2, 2) # 1 row, 2 columns, 2ns plot
            sns.lineplot(x=range(len(fitness_score2)), y=fitness_score2)
            plt.title(dedent(
            f"""fitness score for config_{config} file: tournament selection"""
            ))
            plt.xlabel("generation")
            plt.ylabel("fitness score")

            # show plot
            plt.tight_layout()
            plt.show()

            # sleep for a ew seconds
            sleep(3)

    def run_genetic(self):
        # Initialized average fitness data and fittest individual data
        avg_fitness_data = []
        the_fittest_data = []

        best_solution = None
        best_fitness = 0
        best_generation = 0

        # For each generation, run genetic algorithm
        for gen in range(self.stop):
            roulette_parents = list(self.selection(self.population))
            tournament_parents = list(self.selection(self.population, False))
            roulette_children = self.crossover(roulette_parents)
            tournament_children = self.crossover(tournament_parents)
            roulette_mutants = [self.mutation(kid) for kid in roulette_children]
            tournament_mutants = [self.mutation(kid) for kid in
                                  tournament_children]

#            breakpoint()
            # Selection performance comparison
            avg_fitness = self.average_fitness(self.population)
            fittest_idx, fittest_fit, active_genes = self.the_fittest(
                self.population
            )

            avg_fitness_data.append(avg_fitness)
            the_fittest_data.append((fittest_fit, active_genes))

            # Update best solution
            if fittest_fit > best_fitness:
                best_solution = fittest_idx
                best_fitness = fittest_fit
                best_generation = gen

        # Plotting
        self.plot_selection(avg_fitness_data, the_fittest_data)
        # Output results to console
        print(f"Best solution overall: {best_solution}")
        print(f"Best fitness overall: {best_fitness}")
        print(f"In generation: {best_generation}")

 

if __name__ == "__main__":

    sack = MySack("config_2.txt")
    sack.run_genetic()



