#!/usr/bin/env python
from math import log, exp

def index_difficulty(A, W):
    return log(2*A/W, 2)

print(f"Index index_difficulty: {index_difficulty}")

