#!/usr/bin/env python
from tkinter import *
import sys

print(f"Python version: {sys.version}")

# Print the python version being used
print(f"Python version being used: {sys.version}")

root = Tk()
root.title("My Window")
root.geometry("300x200")  # Set the window size

# Create a frame to hold the widgets
frame = Frame(root)
frame.pack(expand=True)

# Create a label and add it to the frame
label = Label(frame, text="Hello World!", font=("Arial", 16))
label.pack(pady=20)

# Create a button and add it to the frame
button = Button(frame, text="Quit", command=root.quit)
button.pack()

root.mainloop()
