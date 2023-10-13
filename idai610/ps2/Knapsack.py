#!/usr/bin/env python
# Genetic Algorithm to solve the knapsack problem
import numpy as np
"""
n = total # of items available in the set (n = |S|)
pop_size = size of the population (# of individuals, which are also potential
candidate solutions)
Pg = population in a particular generation `g` that's a matrix of size 
    pop_size X n, where each row represents an individual.
    - Binary genes, where 1 = inclusion of item, 0 = exclusion of item
v = value, w = weight: stored in separate arrays or pairs of Tuples

"""

# Set seed, manually
seed_value = 1470
np.random.seed(seed_value)
# Log seed value
with open("random_seed_log.txt", "a") as log_file:
    log_file.write(f"Seed value: {seed_value}")

# EXample code:
def get_initial_population(config):
    """
    Populates variables from config and initiates P at gen 0.
    Parameters: 
        config (str): path to config file
    RETURNS:
        g (int): current generation
        P (matrix or 2D array): Population of individuals
        W (int): Knapsack capacity
        S (list of tuples): Each tuple is an item (w_i, v_1)
        stop (int): Final generation (stop condition)
    """
    # Populate the problem variables
    with open(config, "r") as file:
        lines = file.readlines()

    pop_size, n, stop, W = map(int, [lines[i].strip() for i in range(4)])
    S = [tuple(map(int, line.strip().split())) for line in lines[4:]]
    
    # Initialize population at generation 0
    g = 0
    P = np.random.randint(2, size = (pop_size, n))
    breakpoint()

    return P, W, S, g, stop

P, W, S, g, stop = get_initial_population("config_1.txt")
print(f"P = {P}\nW = {W}\ng = {g}\nstop = {stop}\n")


def fitness():
    pass


