#!/usr/bin/env python
# Brute force  method to solve knapsack 0/1 problem

class Item(object):
    def __init__(self, n, v, w):
        self._name = n
        self._value = v
        self._weight = w

    def get_name(self):
        return self._name

    def get_value(self):
        return self._value

    def get_weight(self):
        return self._weight

    def __str__(self):
        return f"<{self._name}, {self._value}, {self._weight}>"


def value(item):
    return item.get_value()

def weight_inverse(item):
    return 1.0 / item.get_weight()

def density(item):
    return item.get_value() / item.get_weight()

def greedy(items, max_weight, key_function):
    """Assumes items is a list, max_weight >= 0,
    key_function maps elements of items to numbers"""
    items_copy = sorted(items, key=key_function, reverse=True)
    result = []
    total_value, total_weight = 0.0, 0.0

    for i in range(len(items_copy)):
        if (total_weight + items_copy[i].get_weight()) <= max_weight:
            result.append(items_copy[i])
            total_weight += items_copy[i].get_weight()
            total_value += items_copy[i].get_value()

    return (result, total_value)

def build_items():
    names = ["sword", "laptop", "book", "drugs", "alcohol", "celsius"]
    values = [4573, 2300, 250, 1000, 230, 26]
    weights = [40, 8, 4, 0.37, 6, 13]
    Items = []

    for i in range(len(values)):
        Items.append(Item(names[i], values[i], weights[i]))
    return Items

def test_greedy(items, max_weight, key_function):
    taken, val = greedy(items, max_weight, key_function)
    print("Total value of items taken: ", val)
    for item in taken:
        print(" ", item)

def test_greedys(max_weight=20):
    items = build_items()
    print("Use greedy by value to fill knapsack of size: ", max_weight)
    test_greedy(items, max_weight, weight_inverse)
    print("\nUse greedy by weight to fill knapsack size: ", max_weight)
    test_greedy(items, max_weight, density)

def choose_best(pset, max_weight, get_val, get_weight):
    """ INPUT: 
            pset: 
            max_weight: 
            get_val:
            get_weight:

        OUTPUT: (tuple) 
    """
    best_val = 0.0
    best_set = None
    
    for items in pset:
        items_val= 0.0
        items_weight = 0.0

        for item in items:
            items_val += get_val(item)
            items_weight += get_weight(item)

            if items_weight <= max_weight and items_val > best_val:
                best_val = items_val
                best_set = items

    return (best_set, best_val)

def get_binary_rep(n, num_digits):
    """ Assumes n and num_digits are nonnegative ints
    Returns a str of length num_digits that's a binary represnetation of n """
    result = ''
    while n > 0:
        result = str(n % 2) + result
        n //= 2

        if len(result) > num_digits:
            raise ValueError("NOT ENOUGH DIGITS")
#
    # Insert zeros to beginning to properly represent binary number
    for i in range(num_digits - len(result)):
        result = "0" + result

    return result

def gen_powerset(L):
    """ Assumes L is a list
    Returns a list of lists containing all possible combinations of the
    elements of L ..."""
    powerset = []
    for i in range(0, 2**len(L)):
        bin_str = get_binary_rep(i, len(L)) # Why is this None??
        subset = []

        for j in range(len(L)):
            if bin_str[j] == "1":
                subset.append(L[j])

            powerset.append(subset)

    return powerset

def test_best(max_weight=20):
    items = build_items()
    pset = gen_powerset(items)
    taken, val = choose_best(pset, max_weight, Item.get_value, Item.get_weight)
    print("Total value of items taken: ", val)

    for item in taken:
        print(item)

""" 𝜃(n*2^n), where n = length of the items and the
list of lists from gen_powerset is 2^n in length, where the
longest list in it is of length n.
choose_best outer loop will be executed 𝛳(2^n) times and the 
number of times inner loop is executed is bounded by n """

def main():
    print("Test Greedy algorithms: \n")
    test_greedys()
    print("Test Best: \n")
    test_best()

if __name__ == "__main__":
    main()
