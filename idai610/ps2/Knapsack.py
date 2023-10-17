#!/usr/bin/env python
import numpy as np
import random
import matplotlib.pyplot as plt
from textwrap import dedent

seed = 1470
print(f"\t\t...\t...\t...Random seed used: {seed}...\t...\t...")
np.random.seed(seed)


class Knapsack(object):
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
                     q2=False,
                     q3=False, 
                     q4=False,
                    roulette=True):
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

        if q2:
            title = "Selection Only"

        elif q3:
            title = "Integrate crossover/mutation"

        elif q4:
            title = "Exploring Population Sizes"

        if roulette:
            plt.title(f""+ r"$\bf{" + f"{title}" + r"}$" + "\nAverage Fitness per Generation: Roulette")
        else:
            plt.title(f"" + r"$\bf{"+ f"{title}" + r"}$" + "\nAverage Fitness per Generation: Tournament")

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
        if roulette:
            plt.title(f"" + r"$\bf{" + f"{title}" r"}$" + "\nThe Fittest per Generation: Roulette")

        else:
            plt.title(f"" + r"$\bf{" + f"{title}" r"}$" + "\nThe Fittest per Generation: Tournament")

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
            plt.title(f"" + r"$\bf{" + f"{title}" r"}$" + "\nActive Genes per Generation: Roulette")

        else:
            plt.title(f"" + r"$\bf{" + f"{title}" r"}$" + "\nActive Genes per Generation: Tournament")

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

        def get_weight(self, chromosome):
            """
            Gets weight of this chromosome.
            -----------------------------------
            INPUT:
                chromosome: (np.ndarray)

            OUTPUT:
                total weight for current chromosome: (int)
            """
            return 

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

    def run_genetic(self,
                    roulette=True,
                    q2=False, 
                    q3=False, 
                    q4=False
                   ):
        """
        Runs the genetic algorithm for each generation, creating a new
        population with each iteration.
        """
        stop = self.stop
        # Initialize populations: The 2nd one is for when comparing
        population = self.population
        population2 = self.population

        best_fitness = 0
        best_active_genes = 0
        best_generation = None
        avg_fit_data = []
        the_fittest_data = []

        for gen in range(stop):
            new_population = []
            new_population2 = []

            # Ensure the population size stays same
            if len(new_population) < len(population):

                if not roulette:
                    # Tournament selection
                    parents = self.selection(population, roulette=False)

                    children = self.crossover(parents)
                    mutants = [self.mutation(kid) for kid in children]
                    new_population.extend(mutants)

                    remaining_population_size = len(population) - \
                    len(new_population)
                    remaining_population = random.sample(list(population),
                                                         remaining_population_size)
                    new_population.extend(remaining_population)


                elif q2:
                    # Compare selection methods
                    parents1 = self.selection(population)
                    parents2 = self.selection(population, roulette=False)
                    
                    new_population.extend(parents1)
                    new_population2.extend(parents2)

                    remaining_population_size1 = len(population) - \
                    len(new_population)
                    remaining_population_size2 = len(population) - \
                    len(new_population2)

                    remaining_population1 = random.sample(list(population),
                    remaining_population_size1)
                    remaining_population2 = random.sample(list(population2),
                    remaining_population_size2)

                    new_population.extend(remaining_population1)
                    new_population2.extend(remaining_population2)
                    """
                    !!!!1 ===> Generation is 0 everytime! ¯\_( ͡° ͜ʖ ͡°)_/¯ 
                    """


                elif q3:
                    # Integrate crossover/mutation
                    self.integrate_crossover_mutation(population, stop)

                elif q4:
                    # Exploring population sizes
                    self.explore_pop_sizes(population)

                else:
                    # Roulette selection
                    parents = self.selection(population)
                    children = self.crossover(parents)
                    mutants = [self.mutation(kid) for kid in children]

                    # Update population
                    new_population.extend(mutants)

            else:
                print(f"""Length of new  popultion exceeds the length of the
                      initial population""")

    # Sort the old population by fitness and replace the least fit individuals
            sorted_indices = np.argsort([self.fitness_func(ind) for ind in population])

            # Getting statistical info
            avg_fitness = self.average_fitness(population)
            fittest_chromosome, most_fit, active_genes = \
            self.the_fittest(population)
            # Append data to lists
            avg_fit_data.append(avg_fitness)
            the_fittest_data.append((most_fit, active_genes))

            # Update best solution
            if most_fit > best_fitness:
                best_solution = fittest_chromosome
                best_fitness = most_fit
                best_generation = gen
                best_active_genes = active_genes

            if q2:
                sorted_indices2 = np.argsort([self.fitness_func(ind) for ind in population])
                population[sorted_indices2[:len(new_population2)]] = \
                new_population2
               
            # Sort the old population by fitness and replace the least fit individuals
            sorted_indices = np.argsort([self.fitness_func(ind) for ind in population])
            population[sorted_indices[:len(new_population)]] = new_population

        # END LOOP

        # Plots
        var = True

        if q2:
            

            for i in range(1, 3):
                self.fitness_plot(
                    avg_fit_data,
                    the_fittest_data,
                    best_generation,
                    best_active_genes,
                    q2,
                    q3,
                    q4,
                    roulette=var
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

                var = False if var else True # Roulette -> Tournament

if __name__ == "__main__":
    ga = Knapsack(f"config_1.txt")
#    ga.run_genetic()
#    ga.run_genetic(roulette=False)
    ga.run_genetic(q2=True)
#    ga.run_genetic(q3=True)
#    ga.run_genetic(q4=True)
"""
Need to refactor this entire thing so user doesn't have to run one at a time...
"""

