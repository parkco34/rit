#!/usr/bin/env pytho
# Cory Parker
# pp1548@rit.edu
import matplotlib.pyplot as plt
from textwrap import dedent
from math import sqrt
import numpy as np
import pandas as pd

# Reads file from file_path input
def read_file(file_path):
    try:
        with open(file_path, "r") as file:
            data = file.read()

    except Exception as e:
        print(f"OOPS! {str(e)}")
        exit()
    # Return a list of floating point numbers
    return [float(num) for num in data.split()]

def get_mode(data):
    """ Gets the mode of the data, if there is one """
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
    n = len(data)
    mean = sum(data) / n

    variance = sum((x - mean)**2 for x in data) / n
    return sqrt(variance)

def analyze_data(data):
    """
    Takes data list as input and converts list to numpy array for
     calculating Standard deviation, mean, median, variance and mode using numpy's
     built-in functions except for standard deviation and mode ...
    """
    # Calculations
    mean = np.mean(data)
    median = np.median(data)
    variance = np.var(data)
    mode = get_mode(data)
    std_dev = get_std_dev(data)
    # Percentages
    perc_std_dev1 = np.sum((data >= mean - std_dev) & (data <= mean + std_dev)) / len(data)
    perc_std_dev2 = np.sum((data >= mean - 2*std_dev) & (data <= mean + 2*std_dev)) / len(data)
    perc_std_dev3 = np.sum((data >= mean - 3*std_dev) & (data <= mean + 3*std_dev)) / len(data)


    # Output
    print(dedent(f"""
------------------------------------------
Mean: {mean}\n
Median: {median}\n
Mode: {mode}\n
Variance: {variance}\n
Standard Deviation: {std_dev}\n
------------------------------------------
Standard Deviation Percentages:
First: {perc_std_dev1}\n
Second: {perc_std_dev2}\n
Third: {perc_std_dev3}\n
------------------------------------------
          """))
    # return ??


def plot_data(data):
    """ Plot histogram, saving plot to file and showing it """
    # Histogram specs
    counts, bin_edges = np.histogram(data, bins=30, density=True)
    proper_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.title("Assignment 5 Dataset")
    plt.xlabel("Values")
    plt.ylabel("Frequencies")
    plt.plot(proper_bins, counts)

    # Save plot
    plt.savefig("./your_histogram")
    # Plot histogram
    plt.show()

def plot_std_dev_change(data):
    """ Calculates standard deviation at each step as data is read in and
    stores values in a list and plots it as well as saves the plot """
    # Initialize empty list for standard deviatoins
    std_devs = []

    # Fill list with std_devs
    for i in range(1, len(data)+1):
        std_dev = get_std_dev(data[:i])
        std_devs.append(std_dev)

    # PLotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(data)+1), std_devs, color="blue")
    plt.title("Change in Standard Deviation")
    plt.xlabel("Number of Data Points")
    plt.ylabel("Standard Deviation")
    plt.grid(True)

    # Save yo plot!
    plt.savefig("./std_dev_changes")
    # Show off yer weak skillz ᕙ(▀̿̿Ĺ̯̿̿▀̿ ̿) ᕗ
    plt.show()

def read_csv_file(file_name):
    """
    Assume using pandas to read csv file
    for counting the number of values between each range:
        (-5,0), (0,5), (5,10), (10,15), (15,20), (20,25)
    """
    try:
        df = pd.read_csv(file_name, header=None, names=["vals"])

    except Exception as oops:
        print(f"LO SIENTO --> {str(oops)}")
        return None

    return df

def values_between(series, minny, maxy):
    """ Takes in a pandas series and minimum/maximum range
    values """
    # series.loc[] for accessing group of rows/columns via boolean array
    return series.loc[(series >= minny) & (series <= maxy)]

def print_values(values, per_column, cols_per_row):
    # Creates a list in groups of ten
    columns = [values[n:n+per_column] for n in range(0, len(values), per_column)]
    # Get maximum number of rows for each interavl
    rows_per_block = max(len(column) for column in columns)
    # Floor division the number of groups of per_column and columns_per_row to
    # for the number of blocks for each interval
    num_blocks = len(columns) // cols_per_row

    # Iterate thru the blocks, printing the borders and values
    for i in range(num_blocks + 1):
        start = i * cols_per_row
        end = start + cols_per_row
        block = columns[start:end]

        # If column is shorter than row_per_block, it appends empty strings to
        # make length of column equal
        for column in block:
            column += [" "] * (rows_per_block - len(column))

        # Top border
        print("┏" + "━" * (cols_per_row*16 + cols_per_row - 1) + "┓")
        # Since I'm printing the block row-by-row, need to transpose the block
        for row in zip(*block):
            print("┃" + "\t".join(f"{str(val):<15}" for val in row) + "┃")
        # Bottom border
        print("┗" + "━" * (cols_per_row*16 + cols_per_row - 1) + "┛")
        print()

def main():
    # Bonus
    data = read_csv_file("a5_dataset.csv")
    # Convert Dataframe to numpy array
    data_values = data["vals"].to_numpy()

    # Analyze the data
    analyze_data(data_values)

    # Plot the data
    plot_data(data_values)

    # Plot the change in standard deviation
    plot_std_dev_change(data_values)

    if data is not None:
        ranges = [(-5,0), (0,5),(5,10), (10,15), (15,20), (20,25) ]

        # iterating thru a list of tuples
        for minny, maxy in ranges:
            values = values_between(data["vals"], minny, maxy)
            print(dedent(f"""
Values between {minny}:{maxy}
Count: {len(values)}"""))
            print("Values:")
            print_values(values.tolist(), 10, 5)

    else:
        raise Exception("Something is wrong!")

if __name__ == "__main__":
    main()

