import os

def expand_input(filename):
    base_dir = os.path.dirname(filename)
    with open(filename, 'r') as file:
        for line in file:
            if line.strip().startswith(r'\input{'):
                input_file = line.strip()[7:-1]  # Extract filename between \input{ and }
                input_path = os.path.join(base_dir, input_file + '.tex')
                yield from expand_input(input_path)
            else:
                yield line

def main():
    input_file = input("Enter the name of the input .tex file: ")
    output_file = 'flat.tex'

    with open(output_file, 'w') as out_file:
        for line in expand_input(input_file):
            out_file.write(line)

    print(f"Flattened file has been written to {output_file}")

if __name__ == "__main__":
    main()
