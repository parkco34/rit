from datetime import datetime
tick = datetime.now()
x = int(input("Enter an integer: "))

ans = 0
while ans**3 < abs(x):
    ans = ans + 1
    print(ans)
    print("Value of decrementing functiona abs(x) is ", abs(x) - ans**3)
    
if ans**3 != abs(x):
    print(x, "is not a perfect cube")
else:
    if x < 0:
        ans = -ans
    print("Cube root of ", x, "is ", ans)
    
tock = datetime.now()
print()
print("Time taken to execute the program is ", tock - tick)