#!/usr/bin/env python
from math import pow, e

input = [1.25, 2.44, .78,  .12]

def softmax(input):
    x_max = max(input)
    e_x = [exp(i-x_max) for i in input]
    sum_ex = sum(e_x)

    return [round(i / sum_ex, 2) for i in e_x]

def exp(n):
    return pow(e, n)

print(softmax(input))
