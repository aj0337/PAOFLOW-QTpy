import numpy as np
from pathlib import Path

from typing import Dict, Optional


import logging
import numpy.typing as npt

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
    transport_direction: int,
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
    `transport_direction` : int
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
        - metadata: ie, ik, vkpt, dims, transport_direction
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
        transport_direction=transport_direction,
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

        f.write(
            f'    <DATA dimwann="{dim}" nkpts="{nkpts}" nspin="{nspin}" spin_component="{spin_component}" '
        )
        f.write(
            f'nk="{nk[0]} {nk[1]} {nk[2]}" shift="{shift}" nrtot="{nrtot}" nr="{nr[0]} {nr[1]} {nr[2]}" '
        )
        f.write(f'have_overlap="{"T" if have_overlap else "F"}"\n')
        f.write(f'fermi_energy="{fermi_energy:.15E}"/>\n')

        f.write('    <DIRECT_LATTICE type="real" size="9" columns="3" units="bohr">\n')
        for row in avec.T:
            f.write(" " + "  ".join(f"{x:.15E}" for x in row) + "\n")
        f.write("    </DIRECT_LATTICE>\n")

        f.write(
            '    <RECIPROCAL_LATTICE type="real" size="9" columns="3" units="bohr^-1">\n'
        )
        for row in bvec.T:
            f.write(" " + "  ".join(f"{x:.15E}" for x in row) + "\n")
        f.write("    </RECIPROCAL_LATTICE>\n")

        f.write(
            f'    <VKPT type="real" size="{3 * nkpts}" columns="3" units="crystal">\n'
        )
        for i in range(vkpts_crystal.shape[1]):
            f.write(
                " " + "  ".join(f"{vkpts_crystal[j, i]:.15E}" for j in range(3)) + "\n"
            )
        f.write("    </VKPT>\n")

        f.write(f'    <WK type="real" size="{nkpts}">\n')
        for w in wk:
            f.write(f" {w:.15E}\n")
        f.write("    </WK>\n")
        f.write(
            f'    <IVR type="integer" size="{3 * nrtot}" columns="3" units="crystal">\n'
        )
        for row in ivr:
            f.write(" {:10d}{:10d}{:10d} \n".format(*row))
        f.write("    </IVR>\n")
        f.write(f'    <WR type="real" size="{nrtot}">\n')
        for w in wr:
            f.write(f" {w:.15E}\n")
        f.write("    </WR>\n")
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


def write_operator_xml(
    filename: str,
    operator_matrix: Optional[np.ndarray] = None,
    ivr: Optional[np.ndarray] = None,
    vr: Optional[np.ndarray] = None,
    grid: Optional[np.ndarray] = None,
    dimwann: int = 0,
    dynamical: bool = False,
    analyticity: str = "",
    eunits: str = "eV",
    nomega: Optional[int] = None,
    iomg_s: Optional[int] = None,
    iomg_e: Optional[int] = None,
    nrtot: Optional[int] = None,
) -> None:
    """
    Write operator data to XML file in the exact format produced by Fortran iotk library.

    This function mimics the Fortran subroutine operator_write_aux exactly, including
    formatting, spacing, and element ordering.
    """

    if dynamical and grid is None:
        raise ValueError("grid must be present for dynamical operators")
    if dynamical and not analyticity:
        raise ValueError("analyticity must be present for dynamical operators")
    if vr is None and ivr is None:
        raise ValueError("both VR and IVR not present")
    if not dynamical and nomega is not None and nomega != 1:
        raise ValueError("invalid nomega for static operator")

    if operator_matrix is not None:
        if nomega is None:
            nomega = operator_matrix.shape[0]
        if nrtot is None:
            nrtot = operator_matrix.shape[1]
    else:
        if nomega is None:
            nomega = 1
        if nrtot is None:
            nrtot = len(ivr) if ivr is not None else len(vr)

    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n')

        f.write("<OPERATOR>\n")

        f.write("  <DATA")
        f.write(f' dimwann="{dimwann}"')
        f.write(f' nrtot="{nrtot}"')
        f.write(f' dynamical="{str(dynamical).upper()}"')
        f.write(f' nomega="{nomega}"')

        if iomg_s is not None:
            f.write(f' iomg_s="{iomg_s}"')
        if iomg_e is not None:
            f.write(f' iomg_e="{iomg_e}"')

        if dynamical:
            f.write(f' analyticity="{analyticity}"')

        f.write(" />\n")

        if vr is not None:
            f.write("  <VR>\n")

            rows, cols = vr.shape
            for i in range(rows):
                for j in range(cols):
                    val = vr[i, j]
                    f.write(f"    {val.real:18.15E},{val.imag:18.15E}\n")
            f.write("  </VR>\n")

        if ivr is not None:
            f.write("  <IVR>\n")
            rows, cols = ivr.shape
            for i in range(rows):
                f.write("    ")
                for j in range(cols):
                    if j > 0:
                        f.write(" ")
                    f.write(f"{ivr[i, j]:8d}")
                f.write("\n")
            f.write("  </IVR>\n")

        if grid is not None:
            f.write("  <GRID")
            if eunits:
                f.write(f' units="{eunits}"')
            f.write(">\n")

            grid_flat = np.array(grid).flatten()

            for i in range(len(grid_flat)):
                if i % 4 == 0:
                    if i > 0:
                        f.write(" \n")

                else:
                    f.write(" ")

                f.write(f"{grid_flat[i]:18.15E}")
            if len(grid_flat) > 0:
                f.write(" \n")
            f.write("  </GRID>\n")

        if operator_matrix is not None:
            for ie in range(nomega):
                f.write(f"  <OPR.{ie + 1}>\n")

                for ir in range(nrtot):
                    matrix = operator_matrix[ie, ir]
                    rows, cols = matrix.shape
                    total_elements = rows * cols

                    f.write(
                        f'    <VR.{ir + 1} type="complex" size="{total_elements}">\n'
                    )

                    for i in range(rows):
                        for j in range(cols):
                            val = matrix[i, j]
                            f.write(f"{val.real: .15E},{val.imag: .15E}\n")

                    f.write(f"    </VR.{ir + 1}>\n")

                f.write(f"  </OPR.{ie + 1}>\n")

            f.write("</OPERATOR>\n")


def complex_matrix_to_text(matrix: np.ndarray) -> str:
    """Convert complex matrix to text format matching Fortran output."""
    rows, cols = matrix.shape
    lines = []

    for i in range(rows):
        row_parts = []
        for j in range(cols):
            val = matrix[i, j]
            row_parts.append(f"{val.real:18.15E},{val.imag:18.15E}")
        lines.append("      " + " ".join(row_parts))

    return "\n" + "\n".join(lines) + "\n    "


def write_kresolved_operator_xml(
    filename: str,
    operator_k: np.ndarray,
    *,
    dimwann: int,
    vkpt: Optional[np.ndarray] = None,
) -> None:
    """
    Write a k-resolved operator snapshot to XML using the existing iotk-like format.

    Parameters
    ----------
    `filename` : str
        Output path.
    `operator_k` : ndarray
        Array of shape ``(nkpts, dim, dim)`` with the operator at a fixed energy for all k-points.
    `dimwann` : int
        Operator dimension.
    `vkpt` : ndarray, optional
        K-points of shape ``(3, nkpts)`` or ``(nkpts, 3)``. Stored only as metadata proxy
        by populating the ``<IVR>`` block with integer indices. Actual k-vectors are not
        written because the base writer does not yet support a dedicated ``<VKPT>`` tag.

    Notes
    -----
    The data are written with ``nomega = 1`` and ``nrtot = nkpts``. To reuse the existing
    writer unmodified, k-points are enumerated into a 3-column integer array stored under
    ``<IVR>``. The matrix values are exact; only the tag semantics differ from a true
    ``<VKPT>`` representation.
    """
    from PAOFLOW_QTpy.io.write_data import (
        write_operator_xml,
    )

    nk, dim1, dim2 = operator_k.shape
    if dim1 != dimwann or dim2 != dimwann:
        raise ValueError("operator_k shape mismatch with `dimwann`")

    ivr_k = np.column_stack(
        [
            np.arange(nk, dtype=int),
            np.zeros(nk, dtype=int),
            np.zeros(nk, dtype=int),
        ]
    )

    write_operator_xml(
        filename=filename,
        operator_matrix=operator_k[np.newaxis, ...],
        ivr=ivr_k,
        dimwann=dimwann,
        dynamical=False,
        nomega=1,
        nrtot=nk,
    )
