# Find approximation to square root of x using bisection search
def bisection_sqrt(number, epsilon):
    # number: number to find square root of
    # epsilon: how close we want to get to the answer
    if number < 0:
        print("Does not exist")
        
    else:
        low = 0
        high = max(1, number)
        ans = (high + low)/2.0
        
        while abs(ans**2 - number) >= epsilon:
            if ans**2 < number:
                low = ans
            else:
                high = ans
                
            ans = (high + low)/2.0
            
        return ans

    
def main():
    num = int(input("Enter a number: "))
    epsilon = float(input("Enter epsilon: "))
    print("Square root of", num, "is", bisection_sqrt(num, epsilon))
    
if __name__ == "__main__":
    main()