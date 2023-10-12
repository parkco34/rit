from datetime import datetime
import os
from format_output import output_format, output_to_file

def file_validation():
    # Validates user-defined path to file
    while True:
        path = input("Enter the path to the file: ")
        if os.path.isfile(path):
            print(f"File path '{path}' is valid")
            return path
        
        else:
            print("Invalid path")
            continue

def get_desitnation_address(line):
    # Takes the cleaned string and returns the destination address
    # Returns a string
    return line[:12]

def get_source_address(line):
    # Takes the cleaned string and returns the source address
    return line[12:24]

def get_ethertype(line):
    # Takes the cleaned string and returns the ethertype
    return line[24:28]
    
def main():
    path = file_validation() # Get user-defined path to file
    data = [] # List to store all data
      
    # Read the file
    try:
        with open(path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("|"):
                    line = line.replace("|", "").replace(" ", "") # Strip line of all whitespaces and vetrical bars
                    print()
                    print(line)
                    
                    # Append correspoding data to lists
                    destination = get_desitnation_address(line)
                    source = get_source_address(line)
                    ethertype = get_ethertype(line)
                    data.append([destination, source, ethertype])
            
    except FileNotFoundError as e:
        print(f"Failed to read file: {e}")
        
    except Exception as e:
        print(f"Excpeption: {e}")

    output_format(data) # Output formatted data
    output_to_file(data) # Output formatted data to file

if __name__ == "__main__":
    tick = datetime.now()
    main()    
    tock = datetime.now()
    print("Time taken: ", tock - tick)