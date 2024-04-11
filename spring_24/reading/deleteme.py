#!/usr/bin/env python
from textwrap import dedent

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
        return key

    except ValueError as e:
        print(f"Incorrect input: {e}\nTRY AGAIN: ")

        try:
            key = int(input("""Enter the number associated with the book: """))

        except ValueError as e:
            # Stupid ( ͡° ͜ʖ ͡°  ) is making me do a ( ͡° ͜ʖ ͡°  ) while loop
            tries = 0
            while  tries < 3:
                    key = input(redent("""\n
Enter the ( ͡° ͜ʖ ͡°  ) number associated with your
book title, if you even read, you illiterate cunt.\n"""
                    ))
            
            try:
                _ = int(key)
                print(f"Variable: {key} can be converted to an integer")
                return int(key)

            except ValueError:
                tries += 1
                print(f"Variable: {variable} cannot be converted to an integer!")

                if tries == 3:

                    print(dedent(f"""\n\n
         SSSS   TTTTT  U   U  PPPP   III  DDDD  
        S        T    U   U  P   P   I   D   D 
         SSS     T    U   U  PPPP    I   D   D 
            S    T    U   U  P       I   D   D 
        SSSS     T     UUU   P      III  DDDD  

            FFFFF  U   U  CCCC  K   K
            F      U   U  C     K  K
            FFFF   U   U  C     KKK
            F      U   U  C     K  K
            F       UUU   CCCC  K   K



             _                         _
            |_|                       |_|
            | |         /^^^\         | |
           _| |_      (| "o" |)      _| |_
         _| | | | _    (_---_)    _ | | | |_ 
        | | | | |' |    _| |_    | `| | | | |
        \          /   /     \   \          /
         \        /  / /(. .)\ \  \        /
           \    /  / /  | . |  \ \  \    /
             \  \/ /    ||Y||    \ \/  /
               \_/      || ||      \_/
                        () ()
                        || ||
                       ooO Ooo

                          \n\n"""))
                return "(◕‿◕)╭∩╮"

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
        title: Book title (tuple: (int, str))
    """
    key = input_validation()

    if (key > 1 or key <= len(books.values(list(books.values())))):
        print("Nice job asshole")

        if key == 0:
            # Replace this was some input validation
            return (indx, input(dedent("""
Please enter the title of your new book:\n
                                """)))

        else:
            for idx in books.keys():
                    if idx == key:
                        return (indx, title)

    
books = {
    1: "To Kill a Mockingbird",
    2: "1984",
    3: "Pride and Prejudice",
    4: "The Great Gatsby",
    5: "Moby Dick"
}

book_selection(books)
breakpoint()

