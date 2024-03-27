#!/usr/bin/env python
import re
from math import ceil
import datetime

# Function to convert mixed numbers or pure numbers to float
def convert_string(string):
    try:
        parts = string.split()
        if len(parts) == 2:
            whole_number, fraction = int(parts[0]), parts[1]
            numerator, denominator = map(int, fraction.split('/'))
            return whole_number + numerator / denominator
        return float(string)
    except ValueError as e:
        print(f"An error occurred: {e}")
        return None

# Main function to calculate and save the reading rate
def calculate_reading_rate():
    pages_read = convert_string(input("Enter the number of pages you have read: "))
    less_than_hour = input("Did you read for less than an hour? (yes/no): ").strip().lower()
    if less_than_hour == "yes":
        hours_spent = 0
        minutes_spent = convert_string(input("Enter the number of minutes spent reading: "))
    else:
        hours_spent = convert_string(input("Enter the number of full hours spent reading: "))
        minutes_spent = convert_string(input("Enter the number of additional minutes spent reading: "))
    total_time_spent_hours = hours_spent + (minutes_spent / 60)
    reading_rate = pages_read / total_time_spent_hours
    print(f"Your reading rate is {reading_rate:.2f} pages per hour.")
    book_name = input("Enter the name of the book you are reading: ").strip()
    save_reading_session(book_name, pages_read, total_time_spent_hours, reading_rate)
    number_of_reading_sessions(reading_rate)

# Function to save reading session to a file
def save_reading_session(book_name, pages_read, hours_spent, reading_rate):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    session_data = f"{date_str}, {book_name}, {pages_read}, {hours_spent:.2f}, {reading_rate:.2f}\n"
    with open("reading_sessions.txt", "a") as file:
        file.write(session_data)
    print("Reading session saved.")

# Function to calculate the number of 30-minute sessions required based on reading rate
def number_of_reading_sessions(reading_rate):
    pages_to_read = convert_string(input("Enter the number of pages you need to read: "))
    total_reading_time_hours = pages_to_read / reading_rate
    sessions_needed = ceil((total_reading_time_hours * 2))  # 0.5 hour sessions
    print(f"You will need {sessions_needed} reading sessions to complete your reading.")

# Function to remove text within parentheses and count words, handling hyphens
def remove_text_within_parentheses_count_words_handle_hyphens(input_string):
    input_string = re.sub(r'\([^()]*\)', '', input_string)
    input_string_without_hyphens = re.sub(r'-', ' ', input_string)
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', input_string_without_hyphens)
    word_count = len(words)
    return input_string_without_hyphens, word_count

# Function to read from text file and process for word count
def process_file_for_word_count():
    file_path = input("Enter the text file to be read: ")
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()
    cleaned_string, word_count = remove_text_within_parentheses_count_words_handle_hyphens(file_contents)
    print(f"Cleaned string: {cleaned_string}")
    print(f"Word count: {word_count}")

# Main script logic
if __name__ == "__main__":
    pass
    # Here, you can choose what functionality to execute, for example:
    # calculate_reading_rate()  # To calculate reading rate and manage reading sessions
    # process_file_for_word_count()  # To process a text file for word count
