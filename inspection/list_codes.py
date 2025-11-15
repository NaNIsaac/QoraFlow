from pathlib import Path


def gather_python_files(path: Path | str, output_file: Path | str):
    """Gather every texts in .py files and save to txt file."""

    path: Path = Path(path)
    total_length = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file in path.glob("**/*.py"):
            outfile.write(f"===== {file} =====\n")
            with open(file, "r", encoding="utf-8") as infile:
                texts = infile.read()
                total_length += len(texts)
                outfile.write(texts)
            outfile.write("\n\n")
    print(total_length)


if __name__ == "__main__":
    project_directory: str = ".\\"
    output_file: str = "project_code.txt"
    gather_python_files(project_directory, output_file)
