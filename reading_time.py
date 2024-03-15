#!/usr/bin/env python
from math import ceil, floor

def time_for_reading(pages, reading_rate):
    """
    INPUT:
        pages: total number of pages to read - (float)
        reading_rate: Avg reading speed (pages per hour) - (float)

    OUTPUT:
        Number of hours to complete the reading - (float)
    """
    # Convert decimal part to a proper time frame
    int_num_of_pages_per_hr = floor(reading_rate)
    fractional_part = reading_rate % 1
    minutes = fractional_part * 60

    return pages / reading_rate

def number_of_sessions(session, total_reading_time):
    """
    INPUT:
        session: Session length - (float)
        total_reading_time: Number of hours to complete reading - (float)

    OUTPUT:
        Number of sessions to complete reading, rounded up (of course) - (nt)
    """
    return ceil(total_reading_time / session)


reading_time = time_for_reading(27+(71-41), 9.800040000000001)
print("Time to complete the readings:")
print(reading_time)
print('')
print("Number of session to complete the readings: ")
print(number_of_sessions(45/60, reading_time))

