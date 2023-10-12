#!/usr/bin/env python
# Exponential Complexity

def get_binary_rep(n, num_digits):
    """ Assumes n and num_digits are nonnegative ints
    Returns a str of length num_digits that's a binary represnetation of n """
    result = ''
    while n > 0:
        result = str(n % 2) + result
        n //= 2

        if len(result) > num_digits:
            raise ValueError("Not enough digits")

        # Insert zeros to beginning to properly represent binary number
        for i in range(num_digits - len(result)):
            result = "0" + result

        return result

def gen_powerset(L):
    """ Assumes L is a list
    Returns a list of lists containing all possible combinations of the
    elements of L ..."""
    powerset = []
    for i in range(2**len(L)):
        bin_str = get_binary_rep(i, len(L))
        subset = []

        for j in range(len(L)):
            if bin_str[j] == "1":
                subset.append(L[j])
            powerset.append(subset)

    return powrset

