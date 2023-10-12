# Algorithmic Complexity (pg. 240)
import time

def f(i):
    """ Assume i is an int and i    >= 0"""
    answer = 1
    while i >= 1:
        answer *= i
        i -= 1
        print(i)
    return answer

def linear_search(L, x):
    # Worst-case running time is linear in size of L
    for e in L:
        if e == x:
            return True
    return False

def fact(n): # Number of steps: 5n + 2
    # Worst-case running time
    """ Assumes possitive n as an int
    Returns n!
    ----------------------------------
    Number of steps:
    5n (1 for while loop condition, 2 for first assignment, 2 for second assignment) + 2 ( 1 step for initial assignment, 1 step for return statment)
    """
    answer = 1
    while n > 1:
        answer *= n
        n -= 1
                    
    return answer
    
# Asymptotic notation
def func(x):
    """ Assume x is int > 0
    ------------------------
    Time: 1000 + x + 2*x**2
    """
    ans = 0
    
    # Loop takes constant time
    for i in range(1000): # 1000 steps
        ans += 1 # 2 steps
    print("Number of additions os far", ans)
    
    # Loop takes time x
    for i in range(x): # x steps
        ans += 1
    print("Number of additions so far", ans)
    
    # Nested loops take time x**2
    # x**2 steps
    for i in range(x): # x
        for j in range(x): # x --> The Quadratic Component
            ans += 1 # These assignments justify the 2 in 2x**2
            ans += 1
    print("Number of additions so far", ans)
    
    return ans

def int_to_str(i):
    """ Assumes i is a nonnegative int
    Returns a decimal string representation of i"""
    # O(log(i)) Complexity
    digits = '0123456789'
    if i == 0:
        return '0'
    
    result = ''
    while i > 0:
        result = digits[i % 10] + result
        i //= 10 # Determines the asymptotic complexity
        
    return result

def add_digits1(n):
    """ Assumes n is nonnegative int
    Returns sum of digits in n"""
    # O(log(n)) complexity
    string_rep = int_to_str(n)
    val = 0
    for c in string_rep:
        val += int(c)
        
    return val

def factorial(x):
    """ ASssumes that x is positive int
    Returns X!"""
    # factorial(x), factorial(x-1), factorial(x-2), ...
    # O(x) SPACE COMPLEXITY
    if x == 1:
        return 1
    
    else:
        return x * factorial(x - 1)
  
# Polynomial Complexity  
def is_subset(L1, L2):
    """ Assumes L1 and L2 are lists.
    Returns True if each element in L1 is also in L2
    and False otherwise"""
    # Each time Outer loop is executed len(L1) times
    for e1 in L1:
        matched = False
        # Each time inner loop is reached, it's executed len(L2) times
        for e2 in L2:
            if e1 == e2:
                matched = True
                break
            
        if not matched:
            return False
    # Inner loop will be reached in len(L1) * len(L2) times
    return True

def intersect(L1, L2):
    """ Assumes L1 and L2 are lists
    Returns a list without duplicates that's the intersection of L1 and L2"""
    # Build a list containing common elements
    tmp = []
    for e1 in L1:
        for e2 in L2:
            if e1 == e2:
                tmp.append(e1)
                break
    # Build a list witout duplicates
    
    result = []
    for e in tmp:
        if e not in result:
            result.append(e)
            
    return result

# Exponential Complexity
def get_binary_rep(n, num_digits):
    """ Assumes n and num_digits are non-negative ints
    Returns a str of length num_digits that's binary
    representation of n"""
    result = ""
    while n > 0:
        result = str(n % 2) + result
        n //= 2
        
    if len(result) > num_digits:
        raise ValueError("not enough digits")
    
    for i in range(num_digits - len(result)):
        result = '0' + result
        
    return result

def gen_powerset(L):
    """ Assumes L is a list
        Returns a list of lists that contain all possible 
        combinations of elements of L.
        If L is [1, 2] it will return a list with elements
        [], [1], [2], and [1, 2]"""
    powerset = []
    for i in range(0, 2**len(L)):
        bin_str = get_binary_rep(i, len(L))
        subset = []
        
        for j in range(len(L)):
            if bin_str[j] == '1':
                subset.append(L[j])
        powerset.append(subset)

    return powerset

# Finger exercise ===================================================
# Determine the Aysmptotic Complexity for the following functions:
def g(L, e):
    """L a list of ints, e is an int"""
    for i in range(100):
        for el in L:
            if el == e:
                return True
    return False

# Answer:

def h(L, e):
    """L a list of ints, e is an int"""
    for i in range(e):
        for el in L:
            if el == e:
                return False
    return True

# Answer:n
# ===================================================================

def main():
    func(10)
    func(1000)
    # func(1000000) # Takes FOREVER, don't do it!
    

if __name__ == "__main__":
    main()