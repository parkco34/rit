def find_root(x, power, epsilon):
    # Find interval containing answer
    if x < 0 and power % 2 == 0:
        return None # Negative number has no even-powered roots
    
    low  = min(-1, x)
    high = max(1, x)
    # Use bisection search
    ans = (high +  low) / 2
    
    while abs(ans**power - x) >= epsilon: # Compute the absolute value of the difference
        # , where the guess is at least epsilon
        # If the sqaure of middle value is less than the number, set low = ans
        # Else, set high = ans
        if ans**power < x:
            low = ans # Low boundary
        
        else:
            high = ans # High boundary
        
        ans = (high + low) / 2 # Midpoint
        
    return ans

if __name__ == "__main__":
    _EPSILON = 0.001
    root1 = find_root(25, 2, _EPSILON)
    root2 = find_root(-8, 3, _EPSILON)
    root3 = find_root(16, 4, _EPSILON)

    if (type(root1) == type(root2) == type(root3)):
        print(sum([root1, root2, root3]))
        
    else:
        print("Data types not compatible")