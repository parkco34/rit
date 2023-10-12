#!/usr/bin/env python
import pandas as pd
from textwrap import dedent

def read_csv_file(file_name):
    try:
        df = pd.read_csv(file_name, header=None, names=["vals"])
    except Exception as oops:
        print(f"💣 Kaboom! Looks like the file decided to bail. Here's what it says: {str(oops)}")
        return None
    return df

def values_between(series, minny, maxy):
    return series.loc[(series >= minny) & (series <= maxy)]

def print_values(values, per_column, cols_per_row):
    columns = [values[n:n+per_column] for n in range(0, len(values), per_column)]
    rows_per_block = max(len(column) for column in columns)
    num_blocks = len(columns) // cols_per_row

    for i in range(num_blocks + 1):
        start = i * cols_per_row
        end = start + cols_per_row
        block = columns[start:end]

        for column in block:
            column += [' '] * (rows_per_block - len(column))

        print('┏' + '━' * (cols_per_row*16 + cols_per_row - 1) + '┓')
        for row in zip(*block):
            print('┃' + '\t'.join(f"{str(val):<15}" for val in row) + '┃')
        print('┗' + '━' * (cols_per_row*16 + cols_per_row - 1) + '┛')
        print()

def main():
    print("💼 Suit up! We're loading your data...")
    data = read_csv_file("a5_dataset.csv")
    print("🎉 Whoosh! Data loaded faster than a hot potato!")

    if data is not None:
        ranges = [(-5,0), (0,5),(5,10), (10,15), (15,20), (20,25) ]

        for minny, maxy in ranges:
            values = values_between(data["vals"], minny, maxy)
            print(dedent(f"""
🎯 Values between {minny}:{maxy}
🧮 Count: {len(values)}"""))
            print("🔢 Values: ")
            print_values(values.tolist(), 10, 5)
    else:
        print("😱 Oh no! Something is wrong! Did the data file go on a strike?")

    print("✨ Voila! All done! That was more fun than a barrel of monkeys, wasn't it? 🐵")

if __name__ == "__main__":
    main()


