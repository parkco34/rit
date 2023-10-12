from testfile2 import *

def main():
    test_cases = read_file("testfile.txt")

    for test_case in test_cases:
        print(f"Running: {test_case}")
        try:
            print(to_int(test_case))
        except ValueError:
            print("Caught ValueError: invalid literal for int()")

        try:
            print(to_float(test_case))
        except ValueError:
            print("Caught ValueError: invalid literal for float()")

        try:
            list_res = to_list(test_case)
            if list_res:
                print(list_res)
        except ValueError:
            print("No list found hither!")

        try:
            dict_res = to_dict(test_case)
            if dict_res:
                print(dict_res)
        except ValueError:
            print("No dict found hither!")

if __name__ == '__main__':
    main()
