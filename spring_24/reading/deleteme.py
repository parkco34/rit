#!/usr/bin/env python

def input_validation():
    """
    Gets user input of a single integer, then outputting that integer.
    -------------------------
    Input:
        None

    Output:
        key: key to a key-value pair (str)
    """

    try:
        key = int(input("""Enter the number associated with the book: """))

    except ValueError as e:
        print(f"Incorrect input: {e}\n try again: ")

        try:
            key = int(input("""Enter the number associated with the book: """))

        except ValueError as e:
            print(f"You're plenty ( ͡° ͜ʖ ͡°  ) stupid aren't ya... ")

            return "(◕‿◕)╭∩╮"

    return key

def book_selection(books):
    """
    USER ENTERS THE NUMBER ASSOCIATED WITH THE DESIRED BOOK.
    Dictionary of integer-book pairs.
    Loops thru the dictionary for the the integer entered, returning that book
    title.
    while loop for proper user input.
    ------------------
    Input:
        books: Integer-book pairs (dict)

    Output:
        title: Book title (str)
    """
    key = input_validation()

    if (key > 1 or key <= len(books.values(list(books.values())))):
        print("Nice job asshole")

    


