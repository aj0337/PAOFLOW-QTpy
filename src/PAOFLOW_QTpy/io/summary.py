import numpy as np
from typing import Dict, Any

from PAOFLOW_QTpy.io.log_module import log_rank0
from PAOFLOW_QTpy.utils.constants import rydcm1, amconv


def print_summary(summary_data: Dict[str, Any]) -> None:
    """
    Print the summary of transport parameters.

    Parameters
    ----------
    summary_data : dict
        Dictionary containing all required keys:
        'calculation_type', 'prefix', 'postfix', 'work_dir', 'dimL', 'dimC', 'dimR',
        'conduct_formula', 'carriers', 'transport_dir', 'lhave_corr', 'write_kdata',
        'write_lead_sgm', 'write_gf', 'niterx', 'nprint', 'datafile_C', 'datafile_L',
        'datafile_R', 'datafile_L_sgm', 'datafile_C_sgm', 'datafile_R_sgm',
        'leads_are_identical', 'do_orthoovp', 'ne', 'ne_buffer', 'emin', 'emax', 'energy_step',
        'delta', 'smearing_type', 'nx_smear', 'xmax', 'shift_L', 'shift_C', 'shift_R',
        'shift_corr', 'nkpts_par', 'nrtot_par', 'use_symm', 'nk_par', 's_par',
        'vkpt_par3D', 'wk_par', 'ivr_par3D', 'wr_par', 'nproc'
    """

    def format_float(val):
        return f"{val:10.5f}"

    log_rank0("")
    log_rank0(
        "  ======================================================================"
    )
    log_rank0(
        "  =  INPUT Summary                                                     ="
    )
    log_rank0(
        "  ======================================================================"
    )
    log_rank0("")
    log_rank0("  <INPUT>")
    log_rank0(f"          Calculation Type :     {summary_data['calculation_type']}")
    log_rank0(f"                    prefix :     {summary_data['prefix']}")
    log_rank0(f"                   postfix :     {summary_data['postfix']}")
    log_rank0(f"                  work_dir :     {summary_data['work_dir']}")
    log_rank0(f"               L-lead dim. :{summary_data['dimL']:>10}")
    log_rank0(f"            conductor dim. :{summary_data['dimC']:>10}")
    log_rank0(f"               R-lead dim. :{summary_data['dimR']:>10}")
    log_rank0(f"       Conductance Formula :     {summary_data['conduct_formula']}")
    log_rank0(f"                  Carriers :     {summary_data['carriers']}")
    log_rank0(f"       Transport Direction :{summary_data['transport_dir']:>10}")
    log_rank0(f"          Have Correlation :     {summary_data['lhave_corr']}")
    log_rank0(f"              Write k-data :     {summary_data['write_kdata']}")
    log_rank0(f"            Write sgm lead :     {summary_data['write_lead_sgm']}")
    log_rank0(f"                Write gf C :     {summary_data['write_gf']}")
    log_rank0(f"           Max iter number :{summary_data['niterx']:>10}")
    log_rank0(f"                    nprint :{summary_data['nprint']:>10}")
    log_rank0("")
    log_rank0(f"        Conductor datafile :     {summary_data['datafile_C']}")
    if summary_data["calculation_type"].strip().lower() == "conductor":
        log_rank0(f"           L-lead datafile :     {summary_data['datafile_L']}")
        log_rank0(f"           R-lead datafile :     {summary_data['datafile_R']}")
    if summary_data["lhave_corr"]:
        log_rank0(f"            L-Sgm datafile :     {summary_data['datafile_L_sgm']}")
        log_rank0(f"            C-Sgm datafile :     {summary_data['datafile_C_sgm']}")
        log_rank0(f"            R-Sgm datafile :     {summary_data['datafile_R_sgm']}")
    log_rank0(
        f"         leads are identical :     {summary_data['leads_are_identical']}"
    )
    log_rank0(f"           ovp orthogonaliz. :     {summary_data['do_orthoovp']}")
    log_rank0("  </INPUT>")
    log_rank0("")

    # ENERGY GRID
    log_rank0("  <ENERGY_GRID>")
    log_rank0(f"                 Dimension :{summary_data['ne']:>10}")
    log_rank0(f"                 Buffering :{summary_data['ne_buffer']:>10}")

    if summary_data["carriers"].strip().lower() == "phonons":
        scale = (rydcm1 / np.sqrt(amconv)) ** 2
        log_rank0(
            f"            Min Frequency :{format_float(summary_data['emin'] * scale)}"
        )
        log_rank0(
            f"            Max Frequency :{format_float(summary_data['emax'] * scale)}"
        )
        log_rank0(
            f"              Energy Step :{format_float(summary_data['energy_step'] * scale)}"
        )
    else:
        log_rank0(f"               Min Energy :{format_float(summary_data['emin'])}")
        log_rank0(f"               Max Energy :{format_float(summary_data['emax'])}")
        log_rank0(
            f"              Energy Step :{format_float(summary_data['energy_step'])}"
        )
    log_rank0(f"                     Delta :{format_float(summary_data['delta'])}")
    log_rank0(f"             Smearing Type :     {summary_data['smearing_type']}")
    log_rank0(f"             Smearing grid :{summary_data['nx_smear']:>10}")
    log_rank0(f"             Smearing gmax :{format_float(summary_data['xmax'])}")
    log_rank0(f"                   Shift_L :{format_float(summary_data['shift_L'])}")
    log_rank0(f"                   Shift_C :{format_float(summary_data['shift_C'])}")
    log_rank0(f"                   Shift_R :{format_float(summary_data['shift_R'])}")
    log_rank0(f"                Shift_corr :{format_float(summary_data['shift_corr'])}")
    log_rank0("  </ENERGY_GRID>")
    log_rank0("")

    # K-POINTS
    log_rank0("  <K-POINTS>")
    log_rank0(f"       nkpts_par = {summary_data['nkpts_par']:>4}")
    log_rank0(f"       nrtot_par = {summary_data['nrtot_par']:>4}")
    log_rank0(f"        use_symm = {summary_data['use_symm']}")

    nk_par3d = summary_data["nk_par3d"]
    s_par3d = summary_data["s_par3d"]
    log_rank0(
        f"\n       Parallel kpoints grid:        nk = ( {nk_par3d[0]:3} {nk_par3d[1]:3}  {nk_par3d[2]:3} )   s = ( {s_par3d[0]:3} {s_par3d[1]:3}  {s_par3d[2]:3} )"
    )
    for i, (vkpt, weight) in enumerate(
        zip(summary_data["vkpt_par3D"].T, summary_data["wk_par"]), 1
    ):
        log_rank0(
            f"       k ({i:3}) =    ( {vkpt[0]:9.5f} {vkpt[1]:9.5f} {vkpt[2]:9.5f} ),   weight = {weight:8.4f}"
        )
    nr_par3d = summary_data["nr_par3d"]
    log_rank0(
        f"\n       Parallel R vector grid:       nr = ( {nr_par3d[0]:3} {nr_par3d[1]:3}  {nr_par3d[2]:3} )"
    )
    for i, (ivr, weight) in enumerate(
        zip(summary_data["ivr_par3D"].T, summary_data["wr_par"]), 1
    ):
        log_rank0(
            f"       R ({i:3}) =    ( {ivr[0]:9.5f} {ivr[1]:9.5f} {ivr[2]:9.5f} ),   weight = {weight:8.4f}"
        )
    log_rank0("  </K-POINTS>")
    log_rank0("")

    # PARALLELISM
    log_rank0("  <PARALLELISM>")
    log_rank0("       Paralellization over frequencies")
    log_rank0(f"       # of processes: {summary_data['nproc']:>5}")
    log_rank0("  </PARALLELISM>")
    log_rank0("")
