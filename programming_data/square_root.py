from datetime import datetime
tick = datetime.now()

x = int(input("Enter an integer: "))
epsilon = 0.01
# step = epsilon**2
step = epsilon**3
num_guess = 0
ans = 0.0

# while abs(ans**2 - x) >= epsilon and ans <= x:
while abs(ans**2 - x) >= epsilon and ans*ans <= x:
    ans += step
    num_guess += 1
    
    if num_guess < 10:
        print(num_guess, ans)
        
print()
print('num_guess =', num_guess)

if abs(ans**2 - x) >= epsilon:
    print('Failed on square root of', x)
else:
    print(ans, 'is close to square root of', x)
    
tock = datetime.now()
print("Time taken to execute the program is ", tock - tick)