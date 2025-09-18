from mpi4py import MPI
from typing import Dict, Optional
import datetime

from PAOFLOW_QTpy import __version__

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def log_startup(main_name):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    if rank == 0:
        print("=" * 70)
        print("              =                                            =")
        print("              =           Quantum Transport Code           =")
        print("              =     (https://aflowlib.org/src/paoflow/)    =")
        print("              =                                            =")
        print("=" * 70)
        print(f"Program <{main_name}>  v. {__version__}  starts ...")
        print(f"Date {current_date} at {current_time}")
        print(f"Number of MPI processes:    {size}")


def log_rank0(message: str):
    if rank == 0:
        print(message)


def log_section_start(name: str):
    log_rank0(f"Begins {name}")


def log_section_end(name: str):
    log_rank0(f"Ends {name}")


def log_proj_data(
    proj_data: Dict,
    *,
    atmproj_sh: float,
    atmproj_thr: float,
    atmproj_nbnd: Optional[int],
    atmproj_do_norm: bool,
    do_orthoovp: bool,
) -> list[str]:
    lines = []
    lines.append("  Dimensions found in atomic_proj.{dat,xml}:")
    lines.append(f"    nbnd     : {proj_data['nbnd']:>5}")
    lines.append(f"    nkpts    : {proj_data['nkpts']:>5}")
    lines.append(f"    nspin    : {proj_data['nspin']:>5}")
    lines.append(f"    natomwfc : {proj_data['natomwfc']:>5}")
    lines.append(f"    nelec    : {proj_data['nelec']:>12.6f}")
    lines.append(f"    efermi   : {proj_data['efermi_raw']:>12.6f}")
    lines.append(f"    energy_units :  {proj_data['energy_units']}   ")
    lines.append("")
    lines.append("  ATMPROJ conversion to be done using:")
    lines.append(
        f"    atmproj_nbnd : {atmproj_nbnd if atmproj_nbnd is not None else proj_data['nbnd']:>5}"
    )
    lines.append(f"    atmproj_thr  : {atmproj_thr:>12.6f}")
    lines.append(f"    atmproj_sh   : {atmproj_sh:>12.6f}")
    lines.append(f"    atmproj_do_norm:  {atmproj_do_norm}")
    if do_orthoovp:
        lines.append("Using an orthogonal basis. do_orthoovp=.true.")
    return lines
