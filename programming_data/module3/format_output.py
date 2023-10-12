def output_format(data_list):
    # Takes data list and outputs formatted data
    for data in data_list:
        print()
        print(f"Destination Address: {data[0]}")
        print(f"Source Address: {data[1]}")
        print(f"Ethertype: {data[2]}")
        print()
        
def output_to_file(data_list):
    # Takes data list and outputs formatted data to file
    output = input("Enter the path to the output file: \nExample: output.txt\n")
    with open(output, "w") as file:
        for data in data_list:
            file.write(f"Destination Address: {data[0]}\n")
            file.write(f"Source Address: {data[1]}\n")
            file.write(f"Ethertype: {data[2]}\n")
            file.write("\n")
            
        file.write(73 * "ᕙ(▀̿ĺ̯▀̿ ̿)ᕗ")
        file.write("\nCheers!\n")