#!/usr/bin/env python
import numpy as np
import time

# Memoization: O(n)
# Top down; Recursive
def fib_memo(n, memo=None):
    start = time.time()
    if memo == None:
        memo = {}

    if n == 0 or n == 1:
        end = time.time()
        print(f"fib_memo1: {end-start}")
        return 1

    try:
        end = time.time()
        print(f"fib_memo2: {end-start}")
        return memo[n]

    except KeyError:
        result = fib_memo(n-1, memo) + fib_memo(n-2, memo)
        memo[n] = result

        end = time.time()
        print(f"fib_memo3: {end-start}")
        return result

# Tabular: O(n)
# Bottom up; enumerative
def fib_tab(n):
    start = time.time()

    tab = [1] * (n+1) # Only the first two values matter
    for i in range(2, n+1):
        tab[i] = tab[i-1] + tab[i-2] # List of all ones

    end = time.time()
    print(f"fib_tab: {end - start}")
    return tab[n]

def main():
    print(f"7th fibonacci number: {fib_tab(7)}")
    print(f"7th fibonacci number: {fib_memo(7)}")

    a = np.array([1, 2, 3])
    print(a)

if __name__ == "__main__":
    main()

