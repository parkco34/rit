#!/usr/bin/env python


class SelectionStrategy(Enum):
    # Constants
    ROULETTE = "roulette"
    TOURNAMENT = "tournament"
    COMPARE = "compare"
    INTEGRATE = "integrate"
    EXPLORE = "explore"


class TheKnapsack(object):
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

    def selection(self, initial_population,
                  strategy=SelectionStrategy.ROULETTE):

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
        pass


     def run_genetic(self):
         """
         Runs overall genetic algorithm.
         """

         for gen in range(self.stop):


