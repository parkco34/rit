# Cory Parker
# 5/18/23
# ISCH-620

def divide(integer1, integer2):
    return integer1 / integer2 if integer2 != 0 else "Cannot divide by zero, ma dude"

def decimal_to_binary(decimal_number):
    "Function to convert a decimal number to binary for both positive and negative numbers"
    binary = ""
    count = 1  # Counter for the number of bits
    # Two's complement
    if decimal_number < 0:
        is_neg = True
        decimal_number = -decimal_number # Make positive
    
    else:
        is_neg = False
    
    while decimal_number > 0:
        # Adds the remainder to the binary number
        binary = str(decimal_number % 2) + binary
        decimal_number //= 2
        count += 1
            
    if is_neg:
        decimal_number = -decimal_number
        binary = str(decimal_number % 2) + binary
        decimal_number //= 2
        
        # Take the complement
        binary = ["1" if bit == "0" else "0" for bit in binary]
        # Add 1
        for bit in range(len(binary) - 1, -1, -1):
            if binary[bit] == "1":
                binary[bit] = "0"
                
            else:
                binary[bit] = "1"
                break
            
        binary = ''.join(binary)  # Convert list back to string
               
    # In case user enters zero
    if binary == "":
        binary = "0000"
        
    elif count <= 4:
        binary = "0" * (4 - len(binary)) + binary
        
    elif count <= 8:
        binary = "0" * (8 - len(binary)) + binary

    # Convert list back to string
    binary = ''.join(binary)
    return binary

def get_mod(integer1, integer2):
    if integer2 != 0:
        remainder = integer1 % integer2
        quotient = integer1 // integer2
        return quotient, remainder
    
    else:
        return "cannot divide by zero, ma dude"

def main():
    print("Welcome to GCCIS ISCH-620 Graduate Intro to Programming with Data: Programming Assignment 1")
    print("User will provide two integres and the program will perform the following operations: ")
    selection = ""
    
    while selection != "q":
        print("====================================================================================================")
        selection = input("""Select one of the following: 
        a. Division
        b. Convert a base 10 number to binary
        c. Modulo arithmetic (remainder and quotient)
        q. Quit\n""").lower()
        print("====================================================================================================")

        try:
            if selection == "a":
                integer1 = int(input("Enter integer 1: "))
                integer2 = int(input("Enter integer 2: "))
                print(f"Result: {divide(integer1, integer2)}") 
            
            elif selection == "b":
                decimal_number = int(input("Enter a decimal number: "))
                print(f"{decimal_to_binary(decimal_number)}")
                
            elif selection == "c":
                integer1 = int(input("Enter integer 1: "))
                integer2 = int(input("Enter integer 2: "))
                quotient, remainder = get_mod(integer1, integer2)
                if integer2 == 0:
                    get_mod(integer1, integer2)
                else:
                    print(f"Result: {integer1} mod {integer2} = {remainder} and {integer1} / {integer2} = {quotient}")
                
            elif selection == "q":
                print("Goodbye!")
                break
            
            else:
                print("Invalid selection. Please try again.")
                
        except ValueError:
            print("Invalid input. Please try again.")
            
        except ZeroDivisionError:
            print("Cannot divide by zero, ma dude")

if __name__ == "__main__":
    main()