#!/usr/bin/env pythonL
from math import ceil
import datetime


def convert_string(string):
    """
    If string is in the form of 'a b/c', converts to a mixed number 'a + b/c' and returns the result as a float.
    """
    try:
        # Split the string into whole number and fraction parts
        parts = string.split()
        if len(parts) == 2:
            # Extract whole number and fraction parts
            whole_number = int(parts[0])
            numerator, denominator = map(int, parts[1].split('/'))
            # Convert to float and add whole number and fraction parts
            fraction_value = numerator / denominator
            
            return whole_number + fraction_value

        elif len(parts) == 1:
            return float(string)          

        else:
            raise ValueError("Invalid input format. Please use the 'a b/c' format.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_reading_rate():
    """
    Prompts user to enter the number of pages read and the total time taken in minutes,
    then calculates reading rate in pages per hour.

    OUTPUT:
        Reading rate in pages per hour (float)
    """
    pages_read = convert_string(input("Enter the number of pages you have read:"))

    # Ask the user if their reading time was less than an hour
    less_than_hour = input("Did you read for less than an hour? (yes/no): ").strip().lower()
    if less_than_hour == "yes":
        hours_spent = 0
        minutes_spent = float(input("Enter the number of minutes spent reading: "))

    else:
        hours_spent = float(input("Enter the number of full hours spent reading: "))
        minutes_spent = float(input("Enter the number of additional minutes spent reading: "))

    total_time_spent_hours = hours_spent + (minutes_spent / 60)  # Convert total minutes to hours
    breakpoint()
    reading_rate = pages_read / total_time_spent_hours
    print(f"Your reading rate is {reading_rate:.2f} pages per hour.")
    book_name = input("Enter the name of the book you are reading: ").strip()
    save_reading_session(book_name, pages_read, total_time_spent_hours, reading_rate)

    # Output number of reading sessions
    number_of_reading_sessions(reading_rate)

    return reading_rate

def number_of_reading_sessions(reading_rate):
    """
    Calculates the number of 30-minute reading sessions needed to read a specific number of pages.

    INPUT:
        reading_rate: Reading rate in pages per hour (float)

    OUTPUT:
        Number of 30-minute sessions required to complete the reading (int)
    """
    total_reading_time_hours = time_to_read_pages(reading_rate)  # Get total reading time in hours
    session_length_hours = 0.5  # 30 minutes expressed in hours
    sessions_needed = ceil(total_reading_time_hours / session_length_hours)
    print(f"You will need {sessions_needed} reading sessions to complete your reading.")
    return sessions_needed

def time_to_read_pages(reading_rate):
    """
    Calculates the time required to read a given number of pages based on the reading rate,
    outputs in hours or minutes depending on the total time.

    INPUT:
        reading_rate: Reading rate in pages per hour (float)

    OUTPUT:
        Time needed to read the specified number of pages, formatted in hours or minutes.
    """
    pages_to_read = convert_string(input("Enter the number of pages you need to read: "))
    time_needed_hours = pages_to_read / reading_rate
    if time_needed_hours < 1:
        time_needed_minutes = time_needed_hours * 60
        print(f"You will need approximately {time_needed_minutes:.0f} minutes to read {pages_to_read} pages.")

    else:
        print(f"You will need approximately {time_needed_hours:.2f} hours to read {pages_to_read} pages.")

    return time_needed_hours

def save_reading_session(book_name, pages_read, hours_spent, reading_rate):
    """
    Saves the reading session data to a file.

    INPUT:
        book_name: The name of the book (str)
        pages_read: Number of pages read in the session (float)
        hours_spent: Total hours spent in the session (float)
        reading_rate: Calculated reading rate (float)
    """
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    session_data = f"{date_str}, {book_name}, {pages_read}, {hours_spent:.2f}, {reading_rate:.2f}\n"

    with open("reading_sessions.txt", "a") as file:
        file.write(session_data)

    print("Reading session saved.")


# Example usage
reading_rate = calculate_reading_rate()


