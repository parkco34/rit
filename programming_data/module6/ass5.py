#!/usr/bin/env python
# Cory Parker
# pp1548@rit.edu
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

# Read data file
def read_file(file_path):
    try:
        with open(file_path, "r") as file:
            data = file.read()

    except Exception as e:
        print(f"OOPS! {str(e)}")
        exit()
    # Return a list of floating point numbers
    return [float(num) for num in data.split()]

# Calculate mode
def get_mode(data):
    # Tuples of arrays: unique data elements, number of occurences
    values, counts = np.unique(data, return_counts=True)
    # Dictionary of the data values and their counts (unique)
    value_count = dict(zip(values, counts))

    # Find the maximum count
    max_count = max(value_count.values())
    # Get a list of all values with the maximum count
    modes = [value for value, count in value_count.items() if count == max_count]

    # If there is a single mode, return it; otherwise, return None
    if len(modes) == 1:
        return modes[0]
    else:
        return None

def get_std_dev(data):
    """ Calculates the standard deviation of a dataset """
    pass

def main(): 
    data = read_file("a5_dataset.txt")
#    plt.hist(data, bins=10)
#    plt.show()
#    data = read_file("delete.txt")

    # Output Mode:
    print(f"Mode: {get_mode(data)}")

    # Histogram specs
    counts, bin_edges = np.histogram(data, bins=30, density=True) 
    proper_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.title("Assignment 5 Dataset")
    plt.xlabel("Values")
    plt.ylabel("Frequencies")
    plt.plot(proper_bins, counts)
    # Plot histogram
    plt.show()

    # Statistics
    variance = np.var(data)
    mean = np.mean(data)
    median = np.median(data)
    std_dev = get_std_dev(data)
#    std_dev = sqrt(variance) # Standard Deviation, technically
    
    # Standard Deviations percentages
    # try-except block here!
    perc_within_1_std_dev = np.sum((data >= mean - std_dev) & (data <= mean + std_dev)) / len(data)
    perc_within_2_std_dev = np.sum((data >= mean - 2*std_dev) & (data <= mean + 2*std_dev)) / len(data)
    perc_within_3_std_dev = np.sum((data >= mean - 3*std_dev) & (data <= mean + 3*std_dev)) / len(data)
    
    print(f"variance: {variance}")
    print(f"mean: {mean}")
    print(f"median: {median}")
    print(f"standard deviation: {std_dev}")
    print(f"""First standard deviation percentage: {round(perc_within_1_std_dev *
          100,1)}%""")
    print(f"""Second standard deviation percentage: {round(perc_within_2_std_dev *
          100, 1)}%""")
    print(f"""Third standard deviation percentage: {round(perc_within_3_std_dev *
          100, 1)}%""")



if __name__ == "__main__":
    main()


