import yaml
from PAOFLOW_QTpy.io.input_parameters import ConductorData


def get_input_from_yaml(yaml_file: str) -> dict:
    with open(yaml_file) as f:
        content = f.read()
        return yaml.safe_load(content)


def load_summary_data_from_yaml(yaml_path: str, comm=None) -> dict:
    full_yaml = get_input_from_yaml(yaml_path)
    input_conductor = full_yaml.get("input_conductor", {})
    hamiltonian_data = full_yaml.get("hamiltonian_data", {})
    validated = ConductorData(filename=yaml_path, validate=True, **input_conductor)
    result = validated.model_dump()
    result["hamiltonian_data"] = hamiltonian_data

    return result
