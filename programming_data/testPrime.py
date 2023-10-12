from datetime import datetime

tick = datetime.now()
x = int(input("Enter an integer greater than 2: "))
smallest_divisor = None
print(type(smallest_divisor))
print(smallest_divisor)

for guess in range(2, x):
    print(guess)
    if x % guess == 0:
        smallest_divisor = guess
        largest_divisor = int((x/smallest_divisor))
        break
    
if smallest_divisor != None:
    print("The smallest divisor of ", x, "is ", smallest_divisor)
    print("The largest divisor of ", x, "is ", largest_divisor)
else:
    print(x, "is a prime number")
    
tock = datetime.now()
print()
print("Time taken to execute the program is ", tock - tick)