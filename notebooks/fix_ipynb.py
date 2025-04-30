import json
from json import JSONDecodeError


def clear_outputs(notebook_file: str, output_file: str):
    with open(notebook_file, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    import os

    root: str = "./"
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".ipynb"):
                notebook = os.path.join(root, file)
                try:
                    clear_outputs(notebook, notebook)
                except JSONDecodeError as e:
                    print(f"{type(e)}: {e}")
                    print(f"Failed to fix {file}")
                    continue
                print(notebook, "cleared outputs")
