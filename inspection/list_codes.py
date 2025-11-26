from pathlib import Path


EXCLUDE_DIRS = {".venv", "__pycache__", "build", "dist"}


def gather_python_files(project_dir: Path, output_file: Path) -> None:
    """Gather every .py file contents under project_dir into output_file, excluding EXCLUDE_DIRS."""
    total_length = 0
    with output_file.open("w", encoding="utf-8") as outfile:
        for py_file in sorted(project_dir.rglob("*.py")):
            # 상위 경로 중 제외할 디렉토리가 포함되어 있으면 스킵
            if any(part in EXCLUDE_DIRS for part in py_file.parts):
                continue

            outfile.write(f"===== {py_file} =====\n")
            texts = py_file.read_text(encoding="utf-8")
            total_length += len(texts)
            outfile.write(texts)
            outfile.write("\n\n")

    print(f"Total characters written: {total_length}")


if __name__ == "__main__":
    project_directory = Path(r"D:\02_Projects\Dev\ReflectoLearn")
    output_file = Path("project_code.txt")
    gather_python_files(project_directory, output_file)
