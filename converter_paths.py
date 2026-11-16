from pathlib import Path


def output_contains_input(input_path, output_path):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    return output_path == input_path or output_path in input_path.parents
