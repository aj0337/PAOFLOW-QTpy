import yaml
from PAOFLOW_QTpy.io.input_parameters import ConductorData, CurrentData


def get_input_from_yaml(yaml_file: str) -> dict:
    with open(yaml_file) as f:
        content = f.read()
        return yaml.safe_load(content)


def load_conductor_data_from_yaml(yaml_path: str, comm=None) -> dict:
    """
    Load and validate conductor input parameters from a YAML configuration file.

    This function parses the YAML file, extracts the `input_conductor` and
    `hamiltonian_data` sections, validates the conductor input against the
    `ConductorData` schema, and returns the combined result as a dictionary.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file containing the conductor input configuration.
    comm : optional
        MPI communicator (default: None).

    Returns
    -------
    dict
        Dictionary of validated conductor input parameters. Contains all fields
        defined in `ConductorData`, along with a `hamiltonian_data` entry.
    """

    full_yaml = get_input_from_yaml(yaml_path)
    input_conductor = full_yaml.get("input_conductor", {})
    hamiltonian_data = full_yaml.get("hamiltonian_data", {})
    validated = ConductorData(filename=yaml_path, validate=True, **input_conductor)
    result = validated.model_dump()
    result["hamiltonian_data"] = hamiltonian_data

    return result


def load_current_data_from_yaml(yaml_path: str) -> dict | None:
    """
    Load current input parameters from a YAML file.

    Parameters
    ----------
    yaml_path : str
        Path to the `current.yaml` file.

    Returns
    -------
    dict or None
        Parsed dictionary of input parameters if file exists, otherwise None.
    """

    validated = CurrentData(filename=yaml_path, validate=True)
    return validated.model_dump()
