Given the instructions and notes provided, here's a structured plan on how you can accomplish this project:

1. **Data Generation and Representation**: As you are required to sort objects, it would be ideal to create a simple class, say `SortableItem`. This class can have attributes that could be ordered, such as `id` and `name`. The `id` could be a randomly generated integer and `name` could be a randomly generated string. This will provide two features, `id` and `name`, on which the sorting can be based.

2. **Sorting Classes**: Create an abstract base class named `SortableList` with methods like `__init__()`, `time_sort()`, `sort_list()`, and `report_time()`. Then, create three subclasses namely `PythonSortList`, `PythonSortedList`, and `ManualSortList` which inherit from `SortableList` and implement the `sort_list()` method to perform sorting using Python's `sort()`, `sorted()`, and a manual sorting method respectively.

3. **Sorting Method Implementation**: In the `PythonSortList` and `PythonSortedList` classes, use the `sort()` and `sorted()` functions of Python respectively. While sorting, you must allow the user to choose the sorting feature (either `id` or `name`) and sorting order (ascending or descending). In the `ManualSortList` class, implement a simple sorting algorithm, like Bubble Sort, to manually sort the data. Again, this method should accept sorting feature and sorting order as parameters.

4. **Timing the Sorting**: Implement the `time_sort()` method in the `SortableList` class to record the time taken by the `sort_list()` method. This would involve noting the time before and after the `sort_list()` method is called, and calculating the difference.

5. **Reporting the Results**: Implement the `report_time()` method in the `SortableList` class to return the time taken to sort the list. This method can be called after sorting to get the time taken by each sorting method.

6. **Creating Instances and Comparing Results**: Outside the classes, create three instances of each sorting class. Call the `time_sort()` and `report_time()` methods for each instance and calculate the average time taken by each sorting method.

7. **Complexity Analysis**: Analyze the time complexity of each sorting method by observing how the time taken changes with the size of the list. Compare this with the expected time complexity of each method (`sort()` and `sorted()` are expected to have a time complexity of O(n log n) while Bubble Sort is expected to have a time complexity of O(n^2)). Also, count the number of computation steps (like number of comparisons or swaps) in the manual sorting method and relate this with the time taken.

Finally, present your observations and analysis in a well-structured on-screen report.

Remember, this is a plan and you may need to refine it as you implement your project. Good luck with your assignment!