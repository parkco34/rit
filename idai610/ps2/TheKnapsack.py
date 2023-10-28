#!/usr/bin/env python

"""
+++++++++++++++++++++++++++++++++ KNAPSACK PROBLEM +++++++++++++++++++++++++++++++++

"""
import numpy as np
import matplotlib.pyplot as plt
import random
from textwrap import dedent


class MySack(object):
    """
    Genetic Algorithm for the 0-1 Knapsack problem
    """
    def __init__(self, config_file, enable_crossover=True):
        self.config_file = config_file
        self.enable_crossover = enable_crossover
        self.method_stats = {}
        self.fitness_func_stats = {}

        # Best stats for each generation
        self.generation_stats = {
            "avg_fitness": [], "best_fitness": [], "best_active_genes": []
        }
        # Best stats for best generation
        self.best_solution = {'fitness': 0.73, 'active_genes': 0, 'generation':
                             0}

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

        self.pop_size, n, stop, W = map(int, [lines[i].strip() for i in
                                                range(4)])
        S = [tuple(map(int, line.strip().split())) for line in lines[4:]]
        # Initialize empty population
        population = np.random.randint(2, size=(self.pop_size, n))

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
        print("========================================================================")
        print(f"Mean (weights): {np.mean(weights)}")
        print(f"Mean (values): {np.mean(values)}")
        print(f"Std Dev (weights): {np.std(weights)}")
        print(f"Std Dev (values): {np.std(values)}")
        print(f"Max weight: {max(weights)}")
        print(f"Max value: {max(values)}")
        
        print(f"Weights: {weights}\nValues: {values}\n")
        print("========================================================================")
        plt.scatter(weights, values)
        plt.xlabel("Weights")
        plt.ylabel("values")
        plt.title("Weights vs. Values for given population")
        plt.show()

    def output_stats(self):
        """
        Outputs statistics to console.
        """
        # Output data from best_solution dictionary
        print(f"{int(7) * '======' } Best Solutions {int(7) * '======' }:")
        for i,j in self.best_solution.items():
            print(dedent(f"""
        {i}: {j}
                  """), end='')
        print(f"{int(13) * '========'}")

        print(f"{int(7) * '======' } Generation Stats  {int(7) * '======' }:")
        for i,j in self.generation_stats.items():
            print(dedent(f"""
        {i}: {j}
                  """), end='')
        print(f"{int(13) * '========'}")

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
        weights = [self.weight_value[gene][0] * chromosome[gene] for gene in range(len(chromosome))]
        values = [self.weight_value[gene][1] * chromosome[gene] for gene in range(len(chromosome))]
        
        total_weight = sum(weights)
        total_value = sum(values)

        # Make sure total weight doesn't exceed knapsack capacity
        # breakpoint()
        if total_weight > self.capacity:
            return 0

        return total_value

    def fitness_func2(self, chromosome):
        """
        Alternative fitness function, which penalizes the fitness of chromosome
        that exceeds the capacity.
        --------------------------------------------------------
        """ 
        weights = [self.weight_value[gene][0] * chromosome[gene] for gene in range(len(chromosome))]
        values = [self.weight_value[gene][1] * chromosome[gene] for gene in range(len(chromosome))]
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

    def compare_fitness_functions(self, population):
        """
        Compares the two fitness functions via average fitness and the best
        solution.
        ------------------------------------------------
        """
        fit1 = [round(self.fitness_func(chromosome)) for chromosome in population]
        fit2 = [round(self.fitness_func2(chromosome)) for chromosome in population]
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

        roulette_parents = self.roulette_selection(population)
        tournament_parents = self.tournament_selection(population)
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
        Fitness values of all individuals in the community are summed and the probability of each individual being 
        selected is the ratio of the fitness value to that total value. The better the chromosomes, the more likely they are to be selected.
        ----------------------------------------------------------------
        INPUT:
            population: (np.ndarray)

        OUTPUT:
            parents: (list) Two (np.ndarray) fittest members of the population.
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
        A random sequence is selected from the entire generation at a time. 
        They will be in the tournament. In a tournament, whichever is better wins. 
        Tournaments continue until the match pool is full. The random selection of 
        chromosomes creates the possibility of co-selection of chromosomes with a low fitness value. 
        In this case, chromosomes can be selected for the next generation, which is better than the rest of the population, even if they are mediocre. 
        Thus, diversity in the population can be maintained.
        ----------------------------------------------------------------
        INPUT:
            population: (np.ndarray)

        OUTPUT:
            parents: (list of np.ndarrays)
        """
        selected_parents = []
        count = 0 # Avoids infinite loop
        
        while not all(selected_parents) and count < 7:
            # Randomly select k individuals from population
            chromosomes = random.sample(list(population), k)
            # Most fit among k chromosomes
            best_fit = float('-inf')
            
            for chrome in chromosomes:
                fit = self.fitness_func(chrome)

                if fit > best_fit:
                    best_fit = fit
                    best_chromosome = chrome
                    
                    # If best chromosome is not None, add it
                    if best_chromosome is not None:
                        selected_parents.append(best_chromosome)
                        
                    count += 1
                    
        return selected_parents

    def compare_selection_methods(self, population):
        """
        Compares the two selection methods via average fitness and the best.
        -------------------------------------------------------------------
        INPUT:
            population: (np.ndarray)
            
        OUTPUT:
            None
        """
        methods = ['roulette', 'tournament']

        for method in methods:
            avg_fitness_list, best_fitness_list, best_active_genes_list = [], [], []
            
            for gen in range(self.stop):
                if method == "roulette":
                    parents = self.roulette_selection(population)

                elif method == "tournament":
                    parents = self.tournament_selection(population)

                # Compute statistics
                avg_fitness = np.mean([self.fitness_func(chrome) for chrome in population])
                best_solution = max(population, key=self.fitness_func)
                best_fitness = self.fitness_func(best_solution)
                best_active_genes = sum(best_solution)

                # Update lists
                avg_fitness_list.append(avg_fitness)
                best_fitness_list.append(best_fitness)
                best_active_genes_list.append(best_active_genes)
                
                # Update population
                population = self.update_population(population, parents)

            # Store method statistics for later comparison
            # method_stats first officiallu being used
            self.method_stats[method] = {
                'avg_fitness': avg_fitness_list,
                'best_fitness': best_fitness_list,
                'best_active_genes': best_active_genes_list
            }            
            
        print(f"Method stats: {self.method_stats}")

    def plot_stats(self):
        """
        Plots statistics for each selection method.
        """
        plt.figure()

        if self.method_stats:
            for idx, method in enumerate(self.method_stats.keys()):
                stats = self.method_stats[method]
                generations = range(len(stats['avg_fitness']))

                # Plot average population fitness per generation
                plt.subplot(3, 1, 1)
                plt.plot(generations, stats['avg_fitness'], label=f"{method} Avg Fitness")
                plt.title('Average Population Fitness per Generation')
                plt.legend()

                # Plot fitness score and number of active genes of fittest individual per generation
                plt.subplot(3, 1, 2)
                plt.plot(generations, stats['best_fitness'], label=f"{method} Best Fitness")
                plt.plot(generations, stats['best_active_genes'], label=f"{method} Best Active Genes")
                plt.title('Best Individual Stats per Generation')
                plt.legend()

                # Report number of active genes and fitness for the best solution overall
                best_generation = stats['best_fitness'].index(max(stats['best_fitness']))
                plt.subplot(3, 1, 3)
                plt.scatter([best_generation], [stats['best_fitness'][best_generation]], label=f"{method} Best Fitness")
                plt.scatter([best_generation], [stats['best_active_genes'][best_generation]], label=f"{method} Best Active Genes")
                plt.title('Best Solution Overall')
                plt.legend()
                
            else: 
                # Plot average population fitness per generation
                plt.subplot(3, 1, 1)
                plt.plot(range(len(self.generation_stats['avg_fitness'])), self.generation_stats['avg_fitness'])
                plt.title('Average Population Fitness per Generation')

                # Plot fitness score and number of active genes of fittest individual per generation
                plt.subplot(3, 1, 2)
                plt.plot(range(len(self.generation_stats['best_fitness'])), self.generation_stats['best_fitness'], label='Best Fitness')
                plt.plot(range(len(self.generation_stats['best_active_genes'])), self.generation_stats['best_active_genes'], label='Best Active Genes')
                plt.legend()
                plt.title('Best Individual Stats per Generation')

                # Report number of active genes and fitness for the best solution overall
                plt.subplot(3, 1, 3)
                plt.scatter([self.best_solution['generation']], [self.best_solution['fitness']], label='Best Fitness')
                plt.scatter([self.best_solution['generation']], [self.best_solution['active_genes']], label='Best Active Genes')
                plt.legend()
                plt.title('Best Solution Overall')
                
        # if hasattr(self, "fitness_func_stats"):
        #     plt.figure()
            
        #     for idx, func_name in enumerate(self.fitness_func_stats.keys()):
        #         stats = self.fitness_func_stats[func_name]
        #         generations = range(len(stats['avg_fitness']))
                
        #         plt.subplot(3, 1, 1)
        #         plt.plot(generations, stats['avg_fitness'], label=f"{func_name} Avg Fitness")
        #         plt.legend()
                
        #         plt.subplot(3, 1, 2)
        #         plt.plot(generations, stats['best_fitness'], label=f"{func_name} Best Fitness")
        #         plt.legend()

        #         plt.subplot(3, 1, 3)
        #         best_generation = stats['best_fitness'].index(max(stats['best_fitness']))
        #         plt.scatter([best_generation], [stats['best_fitness'][best_generation]], label=f"{func_name} Best Fitness")
        #         plt.legend()
                
            plt.tight_layout()
            plt.show()

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

    def mutation(self, children):
        """
        Determines how often offspring have random mutations to their
        representation. Once generated, offspring's bits are flipped with
        probability = M_r (mutation rate)
        -------------------------------------------------------------------
        INPUT:
            children: (list of np.ndarray) Two chromosomes

        OUTPUT:
            mutants: (list of np.ndarray)
        """
        # Randomly select a mutation rate between 0.05 and 0.2 with an increment of 0.05
        mutation_rate = random.choice([0.05, 0.1, 0.15, 0.2])

        for chromosome in children:
            for gene in range(len(chromosome)):
                rand_num = random.uniform(0, 1)  # Generate random number for each gene

                # If the number is less than Mr, flip the bit!
                if rand_num < mutation_rate:
                    chromosome[gene] = 1 - chromosome[gene]

        return children

    def get_weakest(self, population, num_weakest=2):
        """
        Finds the weakest members of a population.
        ------------------------------------------------------------------------
        INPUT:
            population: (np.ndarray)
            num_weakest: (int) Number of weakest individuals to find.

        OUTPUT:
            weakest: (list) Indices of the weakest individuals in the population.
        """
        return sorted(range(len(population)), key=lambda i: self.fitness_func(population[i]))[:num_weakest]

    def update_population(self, population, children):
        """
        Updates population by replacing the worst members by the new children.
        ----------------------------------------------------------------------
        INPUT:
            population: (np.ndarray)
            children: (list) New children to be added to the population.

        OUTPUT:
            Updated population: (np.ndarray)
        """
        weakest_indices = self.get_weakest(population, len(children))
        for i, child in zip(weakest_indices, children):
            population[i] = child

        return population

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
        parents, mutants = [], []
       
        if selection in ["roulette", "tournament"]:
            # Run like normal
            for gen in range(self.stop):
                # This is so ugly
                if selection == "tournament":
                    parents = self.tournament_selection(population) 
                
                elif selection == "roulette":
                    parents = self.roulette_selection(population)

                # Crossover and mutation enabled or disabled
                if self.enable_crossover:
                    children = self.crossover(parents)
                    mutants = self.mutation(children)
                    
                    # Compute and log statistics
                    avg_fitness = np.mean([self.fitness_func(chrome) for chrome in
                                        population])
                    best_solution = max(population, key=self.fitness_func)
                    best_fitness = self.fitness_func(best_solution)
                    best_active_genes = sum(best_solution)
                    
                    # Update best solution when better one is found
                    if best_fitness > self.best_solution["fitness"]:
                        self.best_solution.update({"fitness": best_fitness,
                                                "active_genes": best_active_genes,
                                                "generation": gen})

                    # Stats for this generation
                    self.generation_stats['avg_fitness'].append(avg_fitness)
                    self.generation_stats['best_fitness'].append(best_fitness)
                    self.generation_stats['best_active_genes'].append(best_active_genes)
                    
                    # Updates population by replacing the weak peepz with mutants
                    population = self.update_population(population, mutants)
                    
        else:
            # Run selection only
            self.compare_selection_methods(population) # Has the loop already in it. 
                    
        # Run compare_selection_methods if selection_only is None
        self.plot_stats()


if __name__ == "__main__":
    for config in ["config_1.txt", "config_2.txt"]: 
        ga = MySack(config)
        print("\t\t\tSELECTION ONLY")
        ga.run(selection=None)
        print("\t\t\tSELECTION + CROSSOVER + MUTATION")
        ga.run()


