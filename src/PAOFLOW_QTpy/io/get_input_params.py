import yaml
from PAOFLOW_QTpy.io.input_parameters import ConductorData


def get_input_from_yaml(yaml_file: str) -> dict:
    with open(yaml_file) as f:
        content = f.read()
        return yaml.safe_load(content)


def load_summary_data_from_yaml(yaml_path: str, comm=None) -> dict:
    user_input = get_input_from_yaml(yaml_path)
    input_conductor = user_input.get("input_conductor", {})
    validated = ConductorData(filename=yaml_path, validate=True, **input_conductor)
    return validated.model_dump()
