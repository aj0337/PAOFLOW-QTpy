import numpy as np
from pathlib import Path

from typing import Dict


import logging
import numpy.typing as npt
import xml.etree.ElementTree as ET
from xml.dom import minidom

from PAOFLOW_QTpy.compute_rham import compute_rham
from PAOFLOW_QTpy.utils.converters import crystal_to_cartesian


logger = logging.getLogger(__name__)


def write_dos_and_conductance(
    egrid: np.ndarray,
    data: np.ndarray,
    label: str,
    output_dir: Path,
    precision: int = 9,
    verbose: bool = True,
) -> Path:
    """
    Write energy-resolved data (DOS or conductance) to a plain text file.

    Parameters
    ----------
    `egrid` : (ne,) ndarray
        Energy grid in eV.
    `data` : (nchannels, ne) ndarray
        Data array where each row is a channel or observable.
    `label` : str
        Label used to construct filename, e.g., 'conductance', 'doscond'.
    `output_dir` : Path
        Directory to write output file.
    `precision` : int
        Floating-point precision in output.
    `verbose` : bool
        If True, print the output file path.

    Returns
    -------
    `filepath` : Path
        Full path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{label}.dat"
    filepath = output_dir / filename

    with filepath.open("w") as f:
        for i in range(egrid.shape[0]):
            line = f"{egrid[i]: .{precision}f} " + " ".join(
                f"{val:.{precision}f}" for val in data[:, i]
            )
            f.write(line + "\n")

    if verbose:
        print(f"[INFO] {label} written to: {filepath}")
    return filepath


def write_data(
    egrid: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    label: str,
    output_dir: Path,
    prefix: str = "",
    postfix: str = "",
    precision: int = 9,
    verbose: bool = True,
) -> None:
    """
    Write general data (e.g., conductance or DOS) into a single text file.

    Parameters
    ----------
    `egrid` : (ne,) ndarray
        Energy grid.
    `data` : (dim, ne) or (ne,) ndarray
        Data to write.
    `label` : str
        Data type label used for header and filename (e.g., "conductance", "doscond").
    `output_dir` : Path
        Directory to store the output files.
    `prefix` : str
        Optional prefix to prepend to the filename.
    `postfix` : str
        Optional postfix to append to the filename.
    `precision` : int
        Number of decimal places to write.
    `verbose` : bool
        Whether to print output file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{label}_{postfix}.dat" if prefix else f"{label}{postfix}.dat"
    filepath = output_dir / filename

    width = 15
    fmt = f"{{:{width}.{precision}f}}"

    with filepath.open("w") as f:
        if label == "conductance":
            f.write("# E (eV)   cond(E)\n")
        elif label == "doscond":
            f.write("# E (eV)   doscond(E)\n")

        ne = egrid.shape[0]
        if data.ndim == 1:
            for ie in range(ne):
                f.write(f"{fmt.format(egrid[ie])}{fmt.format(data[ie])}\n")
        else:
            dim = data.shape[0]
            for ie in range(ne):
                values = " ".join(fmt.format(data[i, ie]) for i in range(dim))
                f.write(f"{fmt.format(egrid[ie])}{values}\n")

    if verbose:
        print(f"Writing {label} to {filepath}")


def write_eigenchannels(
    data: np.ndarray,
    ie: int,
    ik: int,
    vkpt: np.ndarray,
    transport_dir: int,
    output_dir: Path,
    prefix: str = "eigchn",
    overwrite: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Write eigenchannel data to a compressed .npz file with metadata.

    Parameters
    ----------
    `data` : (n, m) complex ndarray
        Eigenchannel matrix. Columns correspond to eigenchannels.
    `ie` : int
        Energy index.
    `ik` : int
        k-point index.
    `vkpt` : (3,) float ndarray
        Coordinates of the k-point in crystal units.
    `transport_dir` : int
        Direction of transport (typically 1, 2, or 3).
    `output_dir` : Path
        Directory to write the output file.
    `prefix` : str
        Prefix for the filename (default: "eigchn").
    `overwrite` : bool
        If True, overwrite existing file.
    `verbose` : bool
        If True, print where the file was written.

    Returns
    -------
    `filepath` : Path
        Path to the written file.

    Notes
    -----
    This uses `.npz` to store:
        - eigenchannel data
        - metadata: ie, ik, vkpt, dims, transport_dir
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_ik{ik:04d}_ie{ie:04d}.npz"
    filepath = output_dir / filename

    if filepath.exists() and not overwrite:
        raise FileExistsError(f"File {filepath} already exists.")

    np.savez_compressed(
        filepath,
        eigenchannels=data,
        ie=ie,
        ik=ik,
        vkpt=vkpt,
        transport_dir=transport_dir,
        dim1=data.shape[0],
        dim2=data.shape[1],
    )

    if verbose:
        print(f"[INFO] Eigenchannels written to: {filepath}")

    return filepath


def iotk_index(n: int) -> str:
    """Return IOTK index tag used in XML labels (e.g., 1 → '.1')."""
    return f".{n}"


def write_internal_format_files(
    output_prefix: str,
    hk_data: Dict[str, np.ndarray],
    proj_data: Dict[str, np.ndarray],
    lattice_data: Dict[str, np.ndarray],
    do_orthoovp: bool,
) -> None:
    """
    Write Hamiltonian and optional overlap matrices in a format that matches the legacy IOTK-style .ham file structure.

    The output includes:
    - Dimensional and symmetry metadata in a DATA tag
    - Real-space and reciprocal lattice vectors
    - K-point list and weights
    - R-vectors and their weights
    - Hamiltonian matrix blocks (VR.#)
    - Overlap matrix blocks (OVERLAP.#), if enabled

    Parameters
    ----------
    `output_prefix` : str
        Prefix for the output file (e.g., 'al5_bulk' → 'al5_bulk.ham').
    `hk_data` : Dict[str, np.ndarray]
        Dictionary containing:
            - "Hk": shape (nspin, nrtot, dim, dim), Hamiltonian matrices
            - "S" (optional): shape (nspin, nrtot, dim, dim), Overlap matrices
            - "ivr": shape (nrtot, 3), R-vectors
            - "wr": shape (nrtot,), R-vector weights
    `proj_data` : Dict[str, np.ndarray]
        Dictionary containing:
            - "kpts": shape (nkpts, 3), list of k-points
            - "wk": shape (nkpts,), k-point weights
    `lattice_data` : Dict[str, np.ndarray]
        Dictionary containing:
            - "avec": shape (3, 3), direct lattice vectors
            - "bvec": shape (3, 3), reciprocal lattice vectors
    `do_orthoovp` : bool
        If False and overlap matrices are provided, overlap blocks will be written to the output.
    """
    ham_file = output_prefix + ".ham"

    Hk = hk_data["Hk"]
    Sk = hk_data.get("S", None)
    ivr = hk_data["ivr"]
    wr = hk_data["wr"]

    avec = lattice_data["avec"]
    bvec = lattice_data["bvec"]
    kpts = proj_data["kpts"]
    vkpts_crystal = proj_data["vkpts_crystal"]
    vkpts_cartesian = proj_data["vkpts_cartesian"]

    wk = proj_data["wk"]
    spin_component = "all"
    shift = np.zeros(3, dtype=float)  # No shift in k-point grid for crystal coordinates
    nspin, _, dim, _ = Hk.shape
    nkpts = kpts.shape[1]
    nrtot = ivr.shape[0]
    nk = hk_data["nk"]
    nr = hk_data["nr"]
    have_overlap = Sk is not None and not do_orthoovp
    fermi_energy = 0.0

    vr_crystal = ivr.astype(np.float64).T
    rgrid_cart = crystal_to_cartesian(vr_crystal, avec).T  # (nrtot, 3)
    Hr = np.empty((nrtot, dim, dim), dtype=np.complex128)

    for ir in range(nrtot):
        Hr[ir] = compute_rham(rgrid_cart[ir], Hk[0], vkpts_cartesian, wk)

    if have_overlap:
        Sr = np.empty((nrtot, dim, dim), dtype=np.complex128)
        for ir in range(nrtot):
            Sr[ir] = compute_rham(rgrid_cart[ir], Sk[0], vkpts_cartesian, wk)

    with open(ham_file, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<?iotk version="1.2.0"?>\n')
        f.write('<?iotk file_version="1.0"?>\n')
        f.write('<?iotk binary="F"?>\n')
        f.write("<Root>\n")
        f.write("  <HAMILTONIAN>\n")

        # DATA tag with attributes
        f.write(
            f'    <DATA dimwann="{dim}" nkpts="{nkpts}" nspin="{nspin}" spin_component="{spin_component}" '
        )
        f.write(
            f'nk="{nk[0]} {nk[1]} {nk[2]}" shift="{shift}" nrtot="{nrtot}" nr="{nr[0]} {nr[1]} {nr[2]}" '
        )
        f.write(f'have_overlap="{"T" if have_overlap else "F"}"\n')
        f.write(f'fermi_energy="{fermi_energy:.15E}"/>\n')

        # DIRECT LATTICE
        f.write('    <DIRECT_LATTICE type="real" size="9" columns="3" units="bohr">\n')
        for row in avec.T:
            f.write(" " + "  ".join(f"{x:.15E}" for x in row) + "\n")
        f.write("    </DIRECT_LATTICE>\n")

        # RECIPROCAL LATTICE
        f.write(
            '    <RECIPROCAL_LATTICE type="real" size="9" columns="3" units="bohr^-1">\n'
        )
        for row in bvec.T:
            f.write(" " + "  ".join(f"{x:.15E}" for x in row) + "\n")
        f.write("    </RECIPROCAL_LATTICE>\n")

        # VKPT
        f.write(
            f'    <VKPT type="real" size="{3 * nkpts}" columns="3" units="crystal">\n'
        )
        for i in range(vkpts_crystal.shape[1]):
            f.write(
                " " + "  ".join(f"{vkpts_crystal[j, i]:.15E}" for j in range(3)) + "\n"
            )
        f.write("    </VKPT>\n")
        # WK
        f.write(f'    <WK type="real" size="{nkpts}">\n')
        for w in wk:
            f.write(f" {w:.15E}\n")
        f.write("    </WK>\n")

        # IVR
        f.write(
            f'    <IVR type="integer" size="{3 * nrtot}" columns="3" units="crystal">\n'
        )
        for row in ivr:
            f.write(" {:10d}{:10d}{:10d} \n".format(*row))
        f.write("    </IVR>\n")

        # WR
        f.write(f'    <WR type="real" size="{nrtot}">\n')
        for w in wr:
            f.write(f" {w:.15E}\n")
        f.write("    </WR>\n")

        # RHAM section
        f.write("    <RHAM>\n")
        for ir in range(nrtot):
            tag = f"VR.{ir + 1}"
            f.write(f'      <{tag} type="complex" size="{dim * dim}">\n')
            for z in Hr[ir].flatten():
                f.write(f" {z.real:> .15E},{z.imag:> .15E}\n")
            f.write(f"      </{tag}>\n")

            if have_overlap:
                tag = f"OVERLAP.{ir + 1}"
                f.write(f'      <{tag} type="complex" size="{dim * dim}">\n')
                for z in Sr[ir].flatten():
                    f.write(f" {z.real:> .15E},{z.imag:> .15E}\n")
                f.write(f"      </{tag}>\n")
        f.write("    </RHAM>\n")

        # KHAM section
        write_kham(Hk, f)

        f.write("  </HAMILTONIAN>\n")
        f.write("</Root>\n")


def write_kham(
    Hk: np.ndarray,
    f: object,
    spin_component: str = "all",
    tag: str = "KHAM",
    block_prefix: str = "KH",
) -> None:
    """
    Write Hk to an IOTK-style XML file.

    Parameters
    ----------
    `Hk` : (nspin, nkpts, dim, dim) complex ndarray
        Hamiltonian matrices in k-space.
    `output_file` : Path
        Destination XML file.
    `spin_component` : str
        One of: "all", "up", "down".
    `tag` : str
        Name of the XML block (default: "KHAM").
    `block_prefix` : str
        Prefix for matrix block tags (default: "KH" → <KH.1>, <KH.2>, ...)
    """
    f.write("  <HAMILTONIAN>\n")
    nspin, nkpts, _, _ = Hk.shape

    for isp in range(nspin):
        if spin_component == "up" and isp == 1:
            continue
        if spin_component == "down" and isp == 0:
            continue

        if spin_component == "all" and nspin == 2:
            f.write(f"    <SPIN.{isp + 1}>\n")

        f.write(f"      <{tag}>\n")
        for ik in range(nkpts):
            tagname = f"{block_prefix}.{ik + 1}"
            mat = Hk[isp, ik]
            dim = mat.shape[0]
            f.write(f'        <{tagname} type="complex" size="{dim * dim}">\n')
            for i in range(dim):
                for j in range(dim):
                    z = mat[i, j]
                    f.write(f" {z.real: .15E},{z.imag: .15E}\n")
            f.write(f"        </{tagname}>\n")
        f.write(f"      </{tag}>\n")

        if spin_component == "all" and nspin == 2:
            f.write(f"    </SPIN.{isp + 1}>\n")

    f.write("  </HAMILTONIAN>\n")


def complex_matrix_to_text(matrix: np.ndarray) -> str:
    return "\n".join(
        " ".join(f"({v.real:.10e},{v.imag:.10e})" for v in row) for row in matrix
    )


def write_operator_xml(
    filename: str,
    operator_matrix: np.ndarray,
    ivr: np.ndarray,
    grid: np.ndarray,
    dimwann: int,
    dynamical: bool,
    analyticity: str,
    eunits: str = "eV",
) -> None:
    nomega, nrtot, _, _ = operator_matrix.shape
    iomg_s, iomg_e = 1, nomega

    root = ET.Element("OPERATOR")

    data_attr = {
        "dimwann": str(dimwann),
        "nrtot": str(nrtot),
        "dynamical": str(dynamical).upper(),
        "nomega": str(nomega),
        "iomg_s": str(iomg_s),
        "iomg_e": str(iomg_e),
        "analyticity": analyticity,
    }
    ET.SubElement(root, "DATA", data_attr)

    ivr_el = ET.SubElement(root, "IVR")
    ivr_el.text = "\n" + "\n".join(" ".join(map(str, row)) for row in ivr)

    grid_el = ET.SubElement(root, "GRID", {"units": eunits})
    grid_el.text = "\n" + "\n".join(" ".join(f"{x:.10e}" for x in row) for row in grid)

    for ie in range(nomega):
        opr_el = ET.SubElement(root, f"OPR{ie + 1:03d}")
        for ir in range(nrtot):
            mat_el = ET.SubElement(opr_el, f"VR{ir + 1:03d}")
            mat_el.text = complex_matrix_to_text(operator_matrix[ie, ir])

    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

    with open(filename, "w") as f:
        f.write(xmlstr)
