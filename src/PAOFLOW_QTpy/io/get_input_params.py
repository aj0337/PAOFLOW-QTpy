import yaml
from pathlib import Path
from PAOFLOW_QTpy.io.input_parameters import ConductorData


def get_input_from_yaml(yaml_file: str) -> dict:
    with open(yaml_file) as f:
        content = f.read()
        return yaml.safe_load(content)


def load_summary_data_from_yaml(yaml_path: str, comm=None) -> dict:
    user_input = get_input_from_yaml(yaml_path)
    input_conductor = user_input.get("input_conductor", {})
    dummy_filename = Path(yaml_path).absolute()
    defaults = ConductorData(
        filename=dummy_filename, validate=False, comm=comm
    ).model_dump()
    summary_data = {**defaults, **input_conductor}

    return summary_data
