# Cory Parker
# pp1548@rit.edu
from random import randint
import time
from math import sqrt
from textwrap import dedent


# Class for generating data for future use
class GenerateData:
    def __init__(self):
        self.data = [randint(0, 7300) for num in range(randint(73, 37739))]

    def sort(self):
        """Just being goofy."""
        raise NotImplementedError("This is your children's job!  ( ͠° ͟ʖ ͡°)")
    

class TheSorts(GenerateData):
    """Class intended for the built-in sort and sorted method comparison."""
    
    def my_sort(self):
        data = self.data.copy()
        start = time.time()
        data.sort()
        end = time.time()
        return end - start

    def my_sorted(self):
        data = self.data.copy()
        start = time.time()
        sorted_data = sorted(data) # Saving for later ...
        end = time.time()
        return end - start


class ManualSort(GenerateData):
    """Manually sorting using different sorting algorithms."""
    
    def insertion_sort(self):
        steps = 0
        start = time.time()
        i = 1
        while i < len(self.data):
            j = i
            while j > 0 and self.data[j-1] > self.data[j]:
                self.data[j], self.data[j-1] = self.data[j-1], self.data[j]
                j -= 1
                steps += 1
            i += 1
        end = time.time()
        return end - start, steps

    def bubble_sort(self):
        steps = 0
        start = time.time()
        for num in range(len(self.data)-1, 0, -1):
            for i in range(num):
                if self.data[i] > self.data[i+1]:
                    temp = self.data[i]
                    self.data[i] = self.data[i+1]
                    self.data[i+1] = temp
                steps += 1

        end = time.time()
        return end - start, steps


def main():
    data = GenerateData().data

    sort = TheSorts()
    sort.data = data.copy()
    sort_time = sort.my_sort()
    sorted_time = sort.my_sorted()

    manual1 = ManualSort()
    manual1.data = data.copy()
    insertion_time, insertion_steps = manual1.insertion_sort()

    manual2 = ManualSort()
    manual2.data = data.copy()
    bubble_time, bubble_steps = manual2.bubble_sort()

    print(f"Built-in sort clocked in at: {sort_time}")
    print(f"Built-in sorted clocked in at: {sorted_time}")
    print("==================================================================================================")
    print(f"Manual Insertion Sort clocked in at: {insertion_time} and took {insertion_steps} steps.")
    print(dedent(f"""Based on the number of steps and time it took, the complexity is likely near O(n^2),
which is approximately proportional to the square of the number of elements
(assumed to be approximately √{insertion_steps} ≈ {sqrt(insertion_steps)} elements, which fits with the 
size of the randomly generated array: {len(manual1.data)})"""))
    print("==================================================================================================")
    print(f"Manual Bubble Sort clocked in at: {bubble_time} and took {bubble_steps} steps.")
    print(dedent(f"""Based on the number of steps and time it took, the complexity is likely near O(n^2),
which is approximately proportional to the square of the number of elements
(assumed to be approximately √{bubble_steps} ≈ {sqrt(bubble_steps)} elements, which fits with the size of the 
randomly generated array: {len(manual2.data)})"""))
    print("==================================================================================================")
    # Average time for all to run (Needed to ask about this ... (ノಠ益ಠ)ノ彡┻━┻)
    avg_time = (sort_time + sorted_time + insertion_time + bubble_time) / 4
    print(f"Average time: {avg_time}")


if __name__ == "__main__":
    main()
