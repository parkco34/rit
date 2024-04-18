#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def compute_index_of_difficulty(width, amplitude):
    """
    Compute the Index of Difficulty (ID) based on Fitts' Law.

    Parameters:
    - width (float): The width of the target.
    - amplitude (float): The amplitude or distance to the target.

    Returns:
    - float: The computed Index of Difficulty.
    """
    return np.log2((amplitude / width) + 1)

def plot_mt_vs_id(widths, amplitudes, movement_times):
    """
    Plot the average movement time (MT) as a function of the Index of Difficulty (ID).

    Parameters:
    - widths (list of float): List of target widths.
    - amplitudes (list of float): List of target amplitudes.
    - movement_times (list of float): List of average movement times for each condition.
    """
    # Compute the Index of Difficulty for each condition
    ids = [compute_index_of_difficulty(w, a) for w, a in zip(widths, amplitudes)]
    
    # Plot MT vs ID
    plt.figure(figsize=(10, 5))
    plt.scatter(ids, movement_times, color='blue', label='Data Points')
    plt.title('Average Movement Time vs Index of Difficulty')
    plt.xlabel('Index of Difficulty (ID)')
    plt.ylabel('Average Movement Time (MT)')
    
    # Fit a linear regression line to the data points
    coefficients = np.polyfit(ids, movement_times, 1)
    # Create a linear polynomial from the coefficients
    polynomial = np.poly1d(coefficients)
    # Generate a sequence of IDs for the line
    ids_line = np.linspace(min(ids), max(ids), 100)
    mts_line = polynomial(ids_line)
    
    # Plot the linear regression line
    plt.plot(ids_line, mts_line, color='red', label='Linear Fit')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return coefficients

# Example usage:
# Define widths, amplitudes, and movement times
example_widths = [0.2, 0.4, 0.6, 0.8]
example_amplitudes = [0.5, 1.0, 1.5, 2.0]
example_movement_times = [1.5, 2.0, 2.5, 3.0]

# Call the function to plot the graph and get the coefficients of the linear fit
coefficients = plot_mt_vs_id(example_widths, example_amplitudes, example_movement_times)
print(f"Coefficients of the linear fit: a = {coefficients[0]}, b = {coefficients[1]}")

