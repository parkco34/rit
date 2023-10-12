# =====================================================================================
# testfile.py is a module that contains functions that are used to test the validity of the input file
import os

# Check file existence and read file if exists
def read_file(file_name):
    if not os.path.isfile(file_name):
        print(f"Test file {file_name} not found.")
        return []
    with open(file_name, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]
           
# Conver to int
def to_int(test):
    print(int(test))
    
# Convert ot float
def to_float(test):
    print(float(test))
    
# Convert to list
def to_list(test):
    if test.startswith('[') and test.endswith(']'):
        proper = list(map(int, [element.strip() for element in test[1:-1].split(',')]))
        print(proper)
        
        # Index error
        try:
            print(proper[73])
            
        except IndexError:
            pass
        
def dict_test(test):
    # Oversimplified approach to deteriming if things inside curly brackets are assumed elements that would be in a python dictionary ᕕ( ͡° ͜ʖ ͡°)ᕗ
    #  ̿̿ ̿̿ ̿̿ ̿'̿'\̵͇̿̿\з= ( ▀ ͜͞ʖ▀) =ε/̵͇̿̿/’̿’̿ ̿ ̿̿ ̿̿ 
    if test.startswith('{') and test.endswith('}'):
        # Create each key:value pair in new dict, using dictionary comprehension
        return {pair.split(':')[0].strip().replace('"', ''): pair.split(':')[1].strip().replace('"', '') for pair in test[1:-1].split(',')}
        
# COnvert to dictionary
def to_dict(test):
    print(dict_test(test))
    
    # key error
    try:
        print(dict_test(test)["key73"])
        
    except KeyError:
        pass
    
        

