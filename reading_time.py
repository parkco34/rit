#!/usr/bin/env pythonL
from math import ceil

def calculate_reading_rate():
    """
    Prompts user to enter the number of pages read and the time taken in hours and minutes,
    then calculates reading rate in pages per hour.

    OUTPUT:
        Reading rate in pages per hour (float)
    """
    pages_read = float(input("Enter the number of pages you have read: "))
    hours_spent = float(input("Enter the number of hours spent reading: "))
    minutes_spent = float(input("Enter the number of additional minutes spent reading: "))
    total_time_spent_hours = hours_spent + (minutes_spent / 60)  # Convert minutes to hours and add to hours
    reading_rate = pages_read / total_time_spent_hours
    print(f"Your reading rate is {reading_rate:.2f} pages per hour.")
    return reading_rate

def time_to_read_pages(reading_rate):
    """
    Calculates the time required to read a given number of pages based on the reading rate,
    outputs in hours or minutes depending on the total time.

    INPUT:
        reading_rate: Reading rate in pages per hour (float)

    OUTPUT:
        Time needed to read the specified number of pages, formatted in hours or minutes.
    """
    pages_to_read = float(input("Enter the number of pages you need to read: "))
    time_needed_hours = pages_to_read / reading_rate
    if time_needed_hours < 1:
        time_needed_minutes = time_needed_hours * 60
        print(f"You will need approximately {time_needed_minutes:.0f} minutes to read {pages_to_read} pages.")
    else:
        print(f"You will need approximately {time_needed_hours:.2f} hours to read {pages_to_read} pages.")
    return time_needed_hours

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

# Example usage
reading_rate = calculate_reading_rate()
number_of_reading_sessions(reading_rate)

