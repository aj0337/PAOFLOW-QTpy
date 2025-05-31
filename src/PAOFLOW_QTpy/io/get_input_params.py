import yaml


def get_input_from_yaml(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)
