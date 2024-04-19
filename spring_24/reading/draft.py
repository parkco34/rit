#!/usr/bin/env python
from text_art import thing
from textwrap import dedent

def get_input(new_book=False):
    """
    Gets user input as book number.
    """
    tries = 0
    while tries < 3:
        tries += 1

        if not new_book:
            user = input("\n\nEnter book number:")
            # Ensure input is numeric
            if user.isnumeric():
                return int(user)

            else:
                print("Invalid input, please try again: (◕‿◕)╭∩╮\n")

        else:
            user = input("\n\nEnter book title: (◕‿◕)╭∩╮\n")
            # Ensure input is alphabetic 
            if user.isalpha():
                return user

            else:
                print("Invalid input, please try again: ")

    print(thing)
    return None


def output_book_list(books):
    """
    Provides book list for user to select the given index for the corresponding
    book title, unless it must be added.
    --------------------------------------------------------
    Input:
        books: (dict)

    Ouput:
        None
    """
    for idx, title in books.items():
        print(f"[{idx}]: {title}")

    return None

def output_selection(key, books):
    """
    Output the selected book title.
    -------------------------------
    Input:
        key: (int)
        book title: (str)

    Output:
        selected book titile: (str)
    """
    return books[key]

def create_book(book_title ,books):
    """
    Takes user input and creates a book title.
    ---------------------------------
    Input:
        book_title: (str) 
        books: (dict)

    Output:
        books: (dict) with new index and title added.

    """
    idx = list(books.keys())
    new_indx = idx[-1] + 1
    
    books[new_indx] = book_title
    print(f"New book added: {new_indx}: {book_title}")

    return books


books = {
    1: "To Kill a Mockingbird",
    2: "1984",
    3: "Pride and Prejudice",
    4: "The Great Gatsby",
    5: "Moby Dick"
}


output_book_list(books)
key = get_input()

if key != None:

    if type(key) == int:
        print(f"Book selection: {books[key]}")

        if key == 0:
            new_book = create_book(key, books)

        else:
            book = output_selection(key, books)

else:
    print("Fuck off, yo")

#def main():
#    pass
#
#if __name__ == "__main__":
#    main()
