from random import randint
import time
from math import sqrt
from textwrap import dedent


# Class for generating data for future use
class GenerateData:
    def __init__(self):
        self.data = [randint(0, 7300) for num in range(randint(73, 37739))]

    # Just being goofy   
    def sort(self):
        raise NotImplementedError("This is your children's job!  ( ͠° ͟ʖ ͡°)")

# Implement class to mimic python's built-in sort() and sorted() list methods
# Stolen from the prof. thanks  ᕙ(▀̿ĺ̯▀̿ ̿)ᕗ
class the_sorts(GenerateData):
    """ Class intended for the built-in sort and sorted method comparison"""
    
    def my_sort(self):
        data = self.data.copy() # So I don't affect the original list
        start = time.time()
        data.sort() # built-in method sorts inplace
        end = time.time()
        print(f"Built-in sort clocked in at: {end - start}")
        return data

    def my_sorted(self):
        data = self.data.copy() # So I don't affect the original list
        start = time.time()
        sorted_data = sorted(data) # Built-in sorted method
        end = time.time()
        print(f"Built-in sorted clocked in at: {end - start}")
        return sorted_data


class ManualSort(GenerateData):
    """ manual1ly sorting using different sorting algorithms"""
    def insertion_sort(self):
        """ Manually using INSERTION SORT ALGORITHM """
        steps = 0
        start = time.time()
        i = 1 # Initialize loop variable based on zero-index
        while i < len(self.data):
            j = i
            while j > 0 and self.data[j-1] > self.data[j]:
                self.data[j], self.data[j-1] = self.data[j-1], self.data[j] # proper swap
                j -= 1 # Decrement; move back an index
                steps += 1 # Increment step count
            i += 1 # Increment; move forward an index
        end = time.time()
        print("=================================================================================================================================================================")
        print(f"Manual Insertion Sort clocked in at: {end - start} and took {steps} steps.")
        print(dedent(f"""Based on the number of steps and time it took, the complexity is likely near O(n^2), which is approximately 
proportional to the square of the number of elements (assumed to be approximately √{steps} ≈ {sqrt(steps)} elements, 
which fits with the size of the randomly generated array)"""))
        print("=================================================================================================================================================================")
        return self.data
    
    def bubble_sort(self):
        """ Manually using BUBBLE SORT ALGORITHM"""
        steps = 0
        start = time.time()
        for num in range(len(self.data)-1,0,-1):
            for i in range(num):
                if self.data[i]>self.data[i+1]:
                    temp = self.data[i]
                    self.data[i] = self.data[i+1]
                    self.data[i+1] = temp
                steps += 1
                    
        end = time.time()
        print(f"Mnual Bubble Sort clocked in at: {end - start} and took {steps} steps!")
        print("=================================================================================================================================================================")
        print(f"Manual Insertion Sort clocked in at: {end - start} and took {steps} steps.")
        print(dedent(f"""Based on the number of steps and time it took, the complexity is likely near O(n^2), 
which is approximately proportional to the square of the number of elements (assumed to be approximately √{steps} ≈ {sqrt(steps)} elements, 
which fits with the size of the randomly generated array)"""))
        print("=================================================================================================================================================================")
        return self.data


def main():
    data = GenerateData().data # Generate data
    sort = the_sorts()
    sort.data = data.copy() # Copy data so sorting in one method is isolated to that method
    sort.my_sort()
    sort.my_sorted()

    manual1 = ManualSort()
    manual1.data = data.copy()
    manual1.insertion_sort()

    manual2 = ManualSort()
    manual2.data = data.copy()
    manual2.bubble_sort()

if __name__ == "__main__":
    main()
