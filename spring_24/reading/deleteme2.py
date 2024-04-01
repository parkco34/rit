#!/usr/bin/env python
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import datetime
from math import ceil
import re

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
        messagebox.showerror("Error", f"An error occurred: {e}")
        return None

# Function to calculate and save the reading rate
def calculate_reading_rate(pages_read, less_than_hour, hours_spent, minutes_spent, book_name, pages_to_read):
    pages_read = convert_string(pages_read.get())
    if less_than_hour.get() == 1:  # Yes is selected
        hours_spent = 0
        minutes_spent = convert_string(minutes_spent.get())
    else:
        hours_spent = convert_string(hours_spent.get())
        minutes_spent = convert_string(minutes_spent.get())
    total_time_spent_hours = hours_spent + (minutes_spent / 60)
    reading_rate = pages_read / total_time_spent_hours
    save_reading_session(book_name.get(), pages_read, total_time_spent_hours, reading_rate)
    number_of_reading_sessions(reading_rate, pages_to_read)

# Function to save reading session to a file
def save_reading_session(book_name, pages_read, hours_spent, reading_rate):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    session_data = f"{date_str}, {book_name}, {pages_read}, {hours_spent:.2f}, {reading_rate:.2f}\n"
    with open("deletedis.txt", "a") as file:
        file.write(session_data)
    messagebox.showinfo("Info", "Reading session saved.")

# Function to calculate the number of 30-minute sessions required based on reading rate
def number_of_reading_sessions(reading_rate, pages_to_read):
    pages_to_read = convert_string(pages_to_read.get())
    total_reading_time_hours = pages_to_read / reading_rate
    sessions_needed = ceil((total_reading_time_hours * 2))  # 0.5 hour sessions
    messagebox.showinfo("Info", f"You will need {sessions_needed} reading sessions to complete your reading.")

def setup_gui():
    root = tk.Tk()
    root.title("Reading Rate Calculator")

    tk.Label(root, text="Enter the number of pages you have read:").pack()
    pages_read = tk.Entry(root)
    pages_read.pack()

    less_than_hour = tk.IntVar()
    tk.Checkbutton(root, text="Did you read for less than an hour?", variable=less_than_hour).pack()

    tk.Label(root, text="Enter the number of full hours spent reading:").pack()
    hours_spent = tk.Entry(root)
    hours_spent.pack()

    tk.Label(root, text="Enter the number of additional minutes spent reading:").pack()
    minutes_spent = tk.Entry(root)
    minutes_spent.pack()

    tk.Label(root, text="Enter the name of the book you are reading:").pack()
    book_name = tk.Entry(root)
    book_name.pack()

    tk.Label(root, text="Enter the number of pages you need to read:").pack()
    pages_to_read = tk.Entry(root)
    pages_to_read.pack()

    calculate_button = tk.Button(root, text="Calculate Reading Rate",
                                 command=lambda: calculate_reading_rate(pages_read, less_than_hour, hours_spent, minutes_spent, book_name, pages_to_read))
    calculate_button.pack()

    root.mainloop()

if __name__ == "__main__":
    setup_gui()


