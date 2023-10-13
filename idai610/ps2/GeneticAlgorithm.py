#!/usr/bin/env python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from textwrap import dedent
from time import sleep

# Log Random seed in order to replicate results
seed = 1470
print(f"Random seed used: {seed}")
np.random.seed(seed)

MAX_STAGNANT_GENERATIONS=10
CONVERGENCE_THRESHOLD=0.01

class GeneticAlgorithm(object):
    """
    Genetic Algorithm for the 0-1 Knapsack problem
    ------------------------------------------------
    INPUT:
        population: 
        limit:
        value_weight:
        generation:
        stop: 
    """
    def __init__(self, config_file):
        # Need to change this...
        self.population, self.limit, self.weight_value, self.generation, self.stop = self.initial_population(config_file)

    def initial_population(self, config_file):
        """
        Populates variables from config and initiates P at gen 0.
        --------------------------------------------------------
        RETURNS:
            g: current generation
            chromosomes (matrix or 2D array): Population of individual
            chromosomes
            W (int): Knapsack capacity
            S (list of tuples): Each tuple is an item (w_i, v_1)
            stop (int): Final generation (stop condition)
        """

        # Read file
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
        Determines the fitness for a chromosome of length n via
        taking the summation of the products of values and weights,
        given a total weight constraint
        -----------------------------------------------------------
        INPUT:
            chromosome: (numpy.ndarray)

        OUTPUT:
            Fitness of chromosome: (int)
        """
        total_weight = 0
        total_value =  0
        # If the weight limit is exceeded, return current value
        for i in range(len(chromosome)):
            total_value += self.weight_value[i][1] * chromosome[i]
            total_weight += self.weight_value[i][0] * chromosome[i]

            # Ensure weight limit isn't exceeded
            if total_weight >= self.limit:
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
        print(f"Parent1: {parent1}")
        print(f"Parent2: {parent2}")

        return parent1, parent2

    def crossover(self, parents):
        """
        INPUT:
            parents: (list of lists)

        OUTPUT:
            children: (tuple)
        """
        point = random.randint(1, len(parents[0])-1)
        # For the children; stacks arrays in sequence, horizontally
        # (Column-wise)
        child1 = np.hstack((parents[0][:point], parents[1][point:]))
        child2 = np.hstack((parents[1][:point], parents[0][point:]))

        return child1, child2

    def mutation(self, chromosome, Mr=0.1):
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
        # Iterate thru genes
        for i in range(len(chromosome)):
            rand_num = random.uniform(0, 1) # Generate random number

            # If number is less than Mr, flip!
            if rand_num < Mr:
                chromosome[i] = 1 - chromosome[i]
        
        return chromosome

    def elitism(self, population, children):
        """
        Making up the next generation, includes top-scoring parents from the
        previous generation by combining the 
        ------------------------------------------------------------------
        INPUT:
            population: (np.ndarray) Current population
            children: (np.ndarray) 

        OUTPUT:
            new_generation: (np.ndarray)
        """
        combined = np.vstack((population, children))
        # Sotrs population, using fitness function in descending order
#        sorted_population = combined[np.argsort([-self.fitness_func(chrome) for chrome in combined])]
        sorted_population = sorted(combined, key=lambda chromosome: -self.fitness_func(chromosome))
        new_generation = np.array(sorted_population[:len(population)])

        return new_generation

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

    def generate_generation(self, older_population, roulette=True):
        """
        Generates a new generation
        ------------------------------------
        iNPUT:
            initial_population: 
        """
        new_generation = []
        # Generate new population until it's same size as original
        while len(new_generation) < len(older_population):
            # Roulette or Tournament selection?
            if roulette:
                parent1, parent2 = self.roulette_selection(older_population)

            else:
                parent1, parent2 = self.tournament_selection(older_population)
            
            child1, child2 = self.crossover([parent1, parent2])
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            # Add to new population
            new_generation.extend([child1, child2])

        return new_generation


def main():
    
    ga = GeneticAlgorithm(f"config_{2}.txt")
    
    fitness_score1 = []
    fitness_score2 = []
    stagnant_gen_count = 0 # counter for stagnant generations

    # Stoping criteria: Last generation reached
    for gen in range(ga.stop):
        # Roulette Selection
        elite_population1 = ga.generate_generation(initial_population)
        # Tournament Selection
        elite_population2 = ga.generate_generation(initial_population, False)

        best_solution1 = max(elite_population1, key=ga.fitness_func)
        best_solution2 = max(elite_population2, key=ga.fitness_func)
        best_fit1 = ga.fitness_func(best_solution1)
        best_fit2 = ga.fitness_func(best_solution2)
        fitness_score1.append(best_fit1)
        fitness_score2.append(best_fit2)

        # my stopping criteria: extra credit! (▀̿ĺ̯▀̿ ̿)
        # check for convergence or stagnation
        if len(fitness_score) > 1:
            if (abs(fitness_score[-1] - fitness_score[-2]) <
            CONVERGENCE_THRESHOLD):
                stagnant_gen_count += 1

            else:
                stagnant_gen_count = 0 # reset counter if improvments

        if stagnant_gen_count >= MAX_STAGNANT_GENERATIONS:
            print("stopping early due to fitness score convergence.")
            break

    # Delete this I think
#    solution = max(initial_population, key=ga.fitness_func)
#    best_fitness = ga.fitness_func(solution)
#    total_weight = sum(ga.weight_value[i][0] * solution[i] for i in
#               range(len(solution)))
    # printing values for debugging
    print(dedent(f"""
                 solution1: {solution1}, best_fitness1: {best_fitness1},
                 fitness_score1: {fitness_score1}, weight1: {weight1}
                 """))
    print(dedent(f"""
                 solution2: {solution2}, best_fitness2: {best_fitness2},
                 fitness_score2: {fitness_score2}, weight2: {weight2}
                 """))


if __name__ == "__main__":
    main()

