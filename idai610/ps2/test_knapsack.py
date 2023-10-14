#!/usr/bin/env python
import numpy as np
import random

# TEST DATA
CHROMOSOME = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0,
       1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
       0, 1, 0, 1, 0, 0])
ITEMS = [(7, 360), (0, 83), (30, 59), (22, 130), (80, 431), (94, 67), (11, 230), (81, 52), (70, 93), (64, 125), (59, 670), (18, 892), (0, 600), (36, 38), (3, 48), (8, 147), (15, 78), (42, 256), (9, 63), (0, 17), (42, 120), (47, 164), (52, 432), (32, 35), (26, 92), (48, 110), (55, 22), (6, 42), (29, 50), (84, 323), (2, 514), (4, 28), (18, 87), (56, 73), (7, 78), (29, 15), (93, 26), (44, 78), (71, 210), (3, 36), (86, 85), (66, 189), (31, 274), (65, 43), (0, 33), (79, 10), (20, 19), (65, 389), (52, 276), (13, 312)]

def test_knapsack_fitness(chromosome):
    total_weight, total_value = 0, 0

    for i in range(len(chromosome)):
        total_weight += self.weight_value[i][0] * chromosome[i]
        total_value += self.weight_value[i][1] * chromosome[i]

        # Esnure weight limit isn't exceeded
        if total_weight > self.capacity:
            return total_value

    return total_value


if __name__ == "__main__":
    assert(
        test_knapsack_fitness(CHROMOSOME) == 3856
    )
