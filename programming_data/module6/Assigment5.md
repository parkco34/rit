Here's a structured plan on how to accomplish the assignment:

**Step 1: Reading the Text File**
1.1. Import necessary Python modules - in this case, `os` for file handling and error checks.
1.2. Create a function `read_file(filename)` which takes filename as input.
1.3. In this function, use a try-except block to catch errors. Open the file using the Python built-in function `open()`. If the file does not exist, throw a FileNotFoundError.
1.4. Read the file line by line using a for loop and store the values in a list.

**Step 2: Preliminary Analysis**
2.1. Import necessary Python modules - `numpy` for mathematical calculations and `matplotlib.pyplot` for graphing.
2.2. Create a function `analyze_data(data)` which takes the data list as input.
2.3. In this function, convert the list to a numpy array for ease of use. 
2.4. Calculate and print the standard deviation, variance, mean, median, and mode using numpy's built-in functions.
2.5. Calculate the percentages of data within one, two, and three standard deviations from the mean, print them as well.

**Step 3: Manual Calculation of Standard Deviation**
3.1. Write a function `manual_std_dev(data)` to calculate the standard deviation manually. This function takes the data list as an input.
3.2. The standard deviation can be calculated by finding the square root of the variance, which itself is the average of the squared deviations from the mean.

**Step 4: Plotting Data**
4.1. Create a function `plot_data(data)`.
4.2. In this function, plot a histogram using matplotlib's `plt.hist()` function.
4.3. Save the plot using `plt.savefig()` and display it using `plt.show()`.

**Step 5: Plotting Change in Standard Deviation**
5.1. Create a function `plot_std_dev_change(data)`.
5.2. In this function, calculate the standard deviation at each step as the data is read in and store the values in a list.
5.3. Plot this list and save the plot.

**Step 6: Reading CSV File and Analyzing Data Ranges**
6.1. Import the `pandas` module.
6.2. Create a function `read_csv_file(filename)` which takes the filename as an input.
6.3. In this function, read the csv file using pandas' `read_csv()` function. Again, use a try-except block to handle errors.
6.4. Analyze the data and count the number of values between each range (-5,0), (0,5), (5,10),(10,15),(15,20) and (20,25). Use pandas' series `between()` function to check this.
6.5. Print the counts.

**Step 7: Main Function**
7.1. Create a main function `main()` where you call all the above functions.
7.2. Execute the main function.

**Note:** It is assumed that the data is numeric, as calculations like mean, median, etc. are being performed on it. If the data contains non-numeric entries, they need to be handled during data cleaning.
