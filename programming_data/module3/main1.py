# Cory Parker
# Assignment 3
from testfile1 import *

def main():
    test = read_file("testfile.txt")
    
    # Iterate thru test cases
    for case in test:
        print(f"Running: {test}")
        # int
        try:
            to_int(case)
            
        except ValueError:
            pass
        
        # float
        try:
            to_float(case)
            
        except ValueError:
            pass
        
        # list
        try:
            to_list(case)
            
        except IndexError:
            pass
        
        # dict
        try:
            to_dict(case)
            
        except KeyError:
            pass
            
        
            

if __name__ == '__main__':
    main()