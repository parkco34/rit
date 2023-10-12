def fact_iter(n):
    """Assumes an integer n > 0
    returns n!"""
    result = 1
    for i in range(1, n+1):
        result *= i
    
    return result

def fact_rec(n):
    """Assumes n and int > 0
    returns n!"""
    if n == 1:
        return n
    else:
        return n*fact_rec(n-1)
    
def fib(n):
    """Assumes n int > 0
    returns Fn = Fn-1 + Fn-2"""
    if n ==0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)
    
def test_fib(n):
    for i in range(n+1):
        print("fib of", i, "=", fib(i))
    
def main():
    n = 1
    result = fact_iter(n)
    print(result, "\n")
    
    n = 4
    result = fact_iter(n)
    print(result)
    

main()
     
    