from program1 import divide, decimal_to_binary, get_mod

# Test divide function
assert divide(37, 1) == 37
assert divide(37, 37) == 1
assert divide(10, 5) == 2
assert divide(10, 0) == "Cannot divide by zero, ma dude"
assert divide(0, 10) == 0

# Test decimal_to_binary function
assert decimal_to_binary(0) == "0000"
assert decimal_to_binary(6) == "0110"
assert decimal_to_binary(-6) == "1010"
assert decimal_to_binary(73) == "01001001"
assert decimal_to_binary(-73) == "10110111"

# Test get_mod function
assert get_mod(37, 1) == (37, 0)
assert get_mod(54, 23) == (2, 8)
assert get_mod(23, 54) == (0, 23)
assert get_mod(0, 54) == (0, 0)
assert get_mod(54, 0) == "cannot divide by zero, ma dude"