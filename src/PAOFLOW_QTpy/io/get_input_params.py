import yaml
from transport.run import run_conductor
from transport.kpoints import kpoints_init
from transport.egrid import egrid_init
from transport.hamiltonian_setup import initialize_operator_blocks


def load_config(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def run_conductor_from_yaml(yaml_path: str) -> None:
    config = load_config(yaml_path)
    params = config["input_conductor"]

    egrid, eta = egrid_init(
        params["emin"], params["emax"], params["ne"], delta=params["delta"]
    )

    kpts, nkpts_par = kpoints_init(nkpts=[1, 1, 1])

    operator_blocks = initialize_operator_blocks(params["dimC"])

    run_conductor(
        postfix=params["postfix"],
        egrid=egrid,
        eta=eta,
        nkpts_par=nkpts_par,
        kpoints=kpts,
        operator_blocks=operator_blocks,
        transport_dir=params["transport_dir"],
        do_orthoovp=params["do_orthoovp"],
        verbose=True,
    )
