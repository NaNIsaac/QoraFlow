import os


def gather_python_files(directory: str, output_file: str):
    """Gather every texts in .py files and save to txt file."""
    total_length = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    outfile.write(f"===== {file_path} =====\n")
                    with open(file_path, "r", encoding="utf-8") as infile:
                        texts = infile.read()
                        total_length += len(texts)
                        outfile.write(texts)
                    outfile.write("\n\n")
    print(total_length)


if __name__ == "__main__":
    project_directory: str = ".\\"
    output_file: str = "project_code.txt"
    gather_python_files(project_directory, output_file)
