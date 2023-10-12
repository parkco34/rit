#!/usr/bin/env python
from random import choice
import matplotlib.pyplot as plt
from textwrap import dedent
from math import sqrt

class Location(object):
    """ Tells the SIMULATION will involve at most two dimensions
    and there's no built-in assumption in the class about the set of directions
    in which a drunk might move"""
    def __init__(self, x, y):
        """ x and y are numbers """
        self._x, self._y = x, y

    def move(self, delta_x, delta_y):
        """ delta_x and y are numbers """
        return Location(self._x + delta_x, self._y + delta_y)

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def dist_from(self, other):
        ox, oy = other._x, other._y
        x_dist, y_dist = self._x - ox, self._y - oy
        return sqrt(x_dist**2 + y_dist**2)

    def __str__(self):
        return f"<{self._x}, {self._y}>"


class Field(object):
    """ Maintains a mapping of drunks to locations. 
    It places no constraints on onlcations, so the Field is of unbouned size...
    Allows drunks to be added into a Field at random locations.
    It says nothing about the patterns in which drunks move, nor does it
    prohibit multiple drunks from occupying the same locaetion or moving
    through spaces occupied other drunks"""
    def __init__(self):
        self._drunks = {}

    def add_drunk(self, drunk, loc):
        if drunk in self._drunks:
            raise ValueError("Duplicate drunk")

        else:
            self._drunks[drunk] = loc

    def move_drunk(self, drunk):
        if drunk not in self._drunks:
            raise ValueError("Drunk not in field")

        x_dist, y_dist = drunk.take_step()
        current_location = self._drunks[drunk]
        self._drunks[drunk] = current_location.move(x_dist, y_dist)

    def get_loc(self, drunk):
        if drunk not in self._drunks:
            raise ValueError("Drunk not in field")
        
        return self._drunks[drunk]


class Drunk(object):
    """ Defines ways in which a drunk wonders thru a field """
    def __init__(self, name=None):
        """ Assume name is a str """
        self._name = name

    def __str__(self):
        if self != None:
            return self._name

        return "Anonymous"


class Usual_drunk(Drunk):
    """ Defines ways in which a drunk wonders thru a field """
    def take_step(self):
        # Restriction that each step is of lenth one and is parallel to either
        # x- or y-axis
        step_choices = [(0,1), (0,-1), (1,0), (-1,0)]
        # Returns randmoly chosen member of sequence passed, each step is
        # equally likely and not influenced by previous stept
        return choice(step_choices)


def walk(field, drunk, num_steps):
    """ Returns the distance between final location and location at the start """
    start = field.get_loc(drunk)
    for step in range(num_steps):
        field.move_drunk(drunk)
    return start.dist_from(field.get_loc(drunk))

def sim_walks(num_steps, num_trials, drunk_class):
    """ Assumes num_steps an int >= 0, num_trials an int > 0,
        drunk_class a subclass of Drunk.
        Simulates num_trials walks of num_steps step each.
        Returns a list of the final distances for each trial
    """
    Homer = drunk_class()
    origin = Location(0, 0)
    distances = []

    for trial in range(num_trials):
        field = Field()
        field.add_drunk(Homer, origin)
        distances.append(round(walk(field, Homer, num_trials), 1))
    return distances

def drunk_test(walk_lengths, num_trials, drunk_class):
    """ Asssumes walk_lengths a sequence of int >= 0
    num_trials and int > 0, drunk_class a subclass of Drunk
    For each number of steps in walk_lengths, runs sim_walks with
    num_trials walks and prints results"""
    for num_steps in walk_lengths:
        distances = sim_walks(num_steps, num_trials, drunk_class)
        print(drunk_class.__name__, "Walk of", num_steps, 
              "steps: Mean =", f"{sum(distances) / len(distances): .3f}", 
              "Max =", f"{max(distances)}, Min = {min(distances)}"
        )

    # Plot walks
    plot_walks(distances, num_steps)

def plot_walks(distance, num_steps): # Doesn't work yet!
    plt.figure(figsize=(10,6))
    plt.plot(distance, sqrt(num_steps), color="blue")
    plt.title("Mean distance from Origin")
    plt.xlabel("Number of Steps")
    plt.ylabel("Distance from Origin")
    plt.show()


def main():
    drunk_test((10, 100, 1000, 10000), 100, Usual_drunk)


if __name__ == "__main__":
    main()
