#!/usr/bin/env python
import re

def remove_text_within_parentheses_count_words_handle_hyphens(input_string):
    # Step 1: Remove text within parentheses, handling nested parentheses
    nested_parentheses_pattern = r'\([^()]*\)'
    while re.search(nested_parentheses_pattern, input_string):
        input_string = re.sub(nested_parentheses_pattern, '', input_string)

    # Step 2: Replace hyphens with spaces to correctly identify hyphen-separated words
    input_string_without_hyphens = re.sub(r'-', ' ', input_string)

    # Step 3: Count words
    # The regex pattern now looks for sequences of characters that begin with a letter
    # and are followed by any number of alphanumeric characters or underscores, counting each as a word.
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', input_string_without_hyphens)
    word_count = len(words)

    return input_string_without_hyphens, word_count

# Example usage
# Reading from text file
# Path to the text file you want to read
file_path = input("Enter the text file to be read: ")

# Reading the text file into a string
with open(file_path, 'r', encoding='utf-8') as file:
    file_contents = file.read()

# Now, file_contents contains the entire content of the file as a single string
# You can pass file_contents to the previously defined function
cleaned_string, word_count = remove_text_within_parentheses_count_words_handle_hyphens(file_contents)

print(f"Cleaned string: {cleaned_string}")
print(f"Word count: {word_count}")


