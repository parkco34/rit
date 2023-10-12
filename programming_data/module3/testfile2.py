import os

def read_file(file_name):
    if not os.path.isfile(file_name):
        print(f"Test file {file_name} not found.")
        return []
    with open(file_name, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def to_int(test):
    return int(test)

def to_float(test):
    return float(test)

def to_list(test):
    if test.startswith('[') and test.endswith(']'):
        return list(map(int, [element.strip() for element in test[1:-1].split(',')]))

def dict_test(test):
    if test.startswith('{') and test.endswith('}'):
        pairs = test[1:-1].split(',')
        d = {}
        for pair in pairs:
            items = pair.split(':')
            if len(items) == 2:
                d[items[0].strip().replace('"', '')] = items[1].strip().replace('"', '')
            else:
                print(f"Invalid dictionary item: {pair}")
        return d
    return {}


def to_dict(test):
    return dict_test(test)
