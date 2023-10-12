def fact_iter(n):
    """ Assumes n an int > 0
    (Non recursion)
    returns n!"""
    print("fact_iter called")
    result = 1
    print(result)
    for i in range(1, n+1):
        result *= i
        print(result)
    return result

def fact_rec(n):
    # Recursion
    print("fact_rec called")
    if n == 1:
        return n
    
    else:
        result = n * fact_rec(n-1)
        print(result)
        return result
    

def main():
    n = 1
    result = fact_iter(n)
    print(result, "\n")
    
    n = 4
    result = fact_iter(n)
    print(result)
    
    # n = 1
    # result = fact_rec(n)
    # print(result, " \n")
    
    # n = 4
    # result = fact_rec(n)
    # print(result)
    
    
if __name__ == "__main__":
    main()