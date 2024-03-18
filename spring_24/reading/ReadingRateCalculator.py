#!/usr/bin/env python
import tkinter as tk
from tkinter import messagebox
from math import ceil
import datetime

class ReadingRateCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reading Rate Calculator")
        self.geometry("500x350")
        self.resizable(False, False)

        # Input Fields
        tk.Label(self, text="Pages Read:").pack()
        self.pages_read_entry = tk.Entry(self)
        self.pages_read_entry.pack()

        tk.Label(self, text="Total Minutes Spent Reading:").pack()
        self.total_minutes_spent_entry = tk.Entry(self)
        self.total_minutes_spent_entry.pack()

        tk.Label(self, text="Book Name:").pack()
        self.book_name_entry = tk.Entry(self)
        self.book_name_entry.pack()

        # Buttons
        tk.Button(self, text="Calculate Reading Rate and Save Session", command=self.calculate_and_save).pack()

    def calculate_and_save(self):
        try:
            pages_read = float(self.pages_read_entry.get())
            total_minutes_spent = float(self.total_minutes_spent_entry.get())
            book_name = self.book_name_entry.get()

            total_time_spent_hours = total_minutes_spent / 60
            reading_rate = pages_read / total_time_spent_hours

            # Save the reading session
            self.save_reading_session(book_name, pages_read, total_time_spent_hours, reading_rate)

            messagebox.showinfo("Reading Rate", f"Your reading rate is {reading_rate:.2f} pages per hour.\n"
                                                f"Session saved for {book_name}.")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for pages read and minutes spent.")

    def save_reading_session(self, book_name, pages_read, hours_spent, reading_rate):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        session_data = f"{date_str}, {book_name}, {pages_read}, {hours_spent:.2f}, {reading_rate:.2f}\n"
        with open("reading_sessionz.txt", "a") as file:
            file.write(session_data)

if __name__ == "__main__":
    app = ReadingRateCalculator()
    app.mainloop()

