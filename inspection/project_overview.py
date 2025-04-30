import ast
import os
from collections import defaultdict

import radon.complexity as rc
import radon.raw as rr
from radon.complexity import cc_rank


def analyze_project(root_dir):
    project_structure = defaultdict(lambda: {"files": [], "modules": {}})

    for dirpath, _, filenames in os.walk(root_dir):
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == ".":
            rel_path = ""

        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                module_name = os.path.splitext(filename)[0]

                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                tree = ast.parse(content)
                classes, functions = [], []

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        functions.append(node.name)

                loc = len(content.splitlines())

                project_structure[rel_path]["files"].append(filename)
                project_structure[rel_path]["modules"][module_name] = {
                    "classes": classes,
                    "functions": functions,
                    "loc": loc,
                }

    return project_structure


def print_project_structure(structure, indent=""):
    for dir_name, dir_content in structure.items():
        print(f"{indent}└── {dir_name}/")

        for module_name, module_info in dir_content["modules"].items():
            print(f"{indent}    ├── {module_name}.py:")
            print(f"{indent}    │   ├── Classes: {', '.join(module_info['classes'])}")
            print(
                f"{indent}    │   ├── Functions: {', '.join(module_info['functions'])}"
            )
            # print(f"{indent}    │   └── Lines of Code: {module_info['loc']}")

        # if dir_content["files"]:
        #     print(f"{indent}    └── Files: {', '.join(dir_content['files'])}")


def analyze_code_complexity(root_dir):
    complexity_data = {}

    for dirpath, _, filenames in os.walk(root_dir):
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == ".":
            rel_path = ""

        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)

                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                raw_metrics = rr.analyze(content)
                cc_metrics = rc.cc_visit(content)

                complexity_data[file_path] = {
                    "loc": raw_metrics.loc,
                    "lloc": raw_metrics.lloc,
                    "sloc": raw_metrics.sloc,
                    "comments": raw_metrics.comments,
                    "multi": raw_metrics.multi,
                    "single_comments": raw_metrics.single_comments,
                    "cc_complexity": {
                        "average": calculate_average_complexity(cc_metrics),
                        "details": [
                            {"name": node.name, "complexity": node.complexity}
                            for node in cc_metrics
                        ],
                    },
                }

    return complexity_data


def calculate_average_complexity(cc_metrics):
    total_complexity = sum(node.complexity for node in cc_metrics)
    return total_complexity / len(cc_metrics) if cc_metrics else 0


def print_code_complexity(complexity_data):
    for file_path, data in complexity_data.items():
        print(f"{file_path}:")
        print(f"  Lines of Code (LOC): {data['loc']}")
        print(f"  Logical Lines of Code (LLOC): {data['lloc']}")
        print(f"  Source Lines of Code (SLOC): {data['sloc']}")
        print(f"  Comments: {data['comments']}")
        print(f"  Multi-line Comments: {data['multi']}")
        print(f"  Single-line Comments: {data['single_comments']}")
        print("  Cyclomatic Complexity (CC):")
        print(f"    Average: {data['cc_complexity']['average']}")
        print("    Details:")
        for detail in data["cc_complexity"]["details"]:
            print(f"      {detail['name']}: {detail['complexity']}")
        print()


def print_complexity_grades(root_dir):

    for dirpath, _, filenames in os.walk(root_dir):
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == ".":
            rel_path = ""

        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)

                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                cc_metrics = rc.cc_visit(content)

                print(f"{file_path}:")
                for node in cc_metrics:
                    grade = cc_rank(node.complexity)
                    print(f"  {node.name}: {node.complexity} ({grade})")
                print()


if __name__ == "__main__":
    root_dir: str = "./"

    project_structure = analyze_project(root_dir)
    print("Resent Project Structure:")
    print_project_structure(project_structure)

    # project_structure = analyze_project("C:\\develop\\center4upt\\XFEL_data\\legacy")
    # print("Old Project Structure:")
    # print_project_structure(project_structure)

    # print("Project Complexity Grades:")
    # print_complexity_grades(root_dir)
    # print("Old Complexity Grades:")
    # print_complexity_grades("C:\\develop\\center4upt\\XFEL_data\\legacy")
    # complexity_data = analyze_code_complexity(root_dir)
    # print("Code Complexity Analysis:")
    # print_code_complexity(complexity_data)
