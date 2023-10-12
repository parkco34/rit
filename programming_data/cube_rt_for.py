from datetime import datetime
tick = datetime.now()
x  = int(input("Enter an integer: "))

smallest_divisor = None
for guess in range(2, x):
    if x % guess == 0:
        smallest_divisor = guess
        break
    
if smallest_divisor != None:
    print("Smallest divisor of ", x, "is ", smallest_divisor)
else:
    print( x, "is a prime number")
    
    
tock = datetime.now()
print()
print("Time taken to execute the program is ", tock - tick)