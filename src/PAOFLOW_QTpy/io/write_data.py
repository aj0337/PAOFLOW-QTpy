import numpy as np
from pathlib import Path

from typing import Dict


import logging


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


def write_kresolved_data(
    egrid: np.ndarray,
    data: np.ndarray,
    label: str,
    output_dir: Path,
    prefix: str = "",
    postfix: str = "",
    precision: int = 9,
    verbose: bool = True,
) -> None:
    """
    Write k-resolved data (e.g. conductance_k or dos_k) into per-k-point .dat files,
    matching the Fortran output structure.

    Parameters
    ----------
    `egrid` : (ne,) ndarray
        Energy grid.
    `data` : (nch, nkpts, ne) or (ne, nkpts) ndarray
        k-resolved data to write.
    `label` : {"cond", "doscond"}
        Type of data. Used for file naming and header.
    `output_dir` : Path
        Directory to store the output files.
    `prefix` : str
        Optional prefix to prepend to filenames.
    `postfix` : str
        Optional postfix to append to filenames (e.g. ".test").
    `precision` : int
        Floating-point precision to write (Fortran uses f15.9).
    `verbose` : bool
        If True, print file paths as they are written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    nch, nkpts, ne = (1, *data.shape) if data.ndim == 2 else data.shape

    for ik in range(nkpts):
        ik_str = f"{ik+1:04d}"
        filename = f"{prefix}_{label}-{ik_str}{postfix}.dat"
        filepath = output_dir / filename

        with filepath.open("w") as f:
            if label == "cond":
                f.write("# E (eV)   cond(E)\n")
            elif label == "doscond":
                f.write("# E (eV)   doscond(E)\n")

            for ie in range(ne):
                if nch == 1:
                    val = data[ie, ik] if data.ndim == 2 else data[0, ik, ie]
                    f.write(f"{egrid[ie]:15.{precision}f} {val:15.{precision}f}\n")
                else:
                    vals = " ".join(
                        f"{data[ch, ik, ie]:15.{precision}f}" for ch in range(nch)
                    )
                    f.write(f"{egrid[ie]:15.{precision}f} {vals}\n")

        if verbose:
            print(f"[INFO] Wrote {label} for k-point {ik+1} → {filepath}")


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

    wk = proj_data["wk"]
    wk_sum = np.sum(wk)
    wk = wk / wk_sum

    spin_component = "all"
    shift = (0.0, 0.0, 0.0)
    nspin, _, dim, _ = Hk.shape
    nkpts = kpts.shape[1]
    nrtot = ivr.shape[0]
    nk = hk_data["nk"]
    nr = hk_data["nr"]
    have_overlap = Sk is not None and not do_orthoovp
    fermi_energy = 0.0

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
        f.write(f"have_overlap=\"{'T' if have_overlap else 'F'}\"\n")
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
            f'    <VKPT type="real" size="{3*nkpts}" columns="3" units="crystal">\n'
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
            f'    <IVR type="integer" size="{3*nrtot}" columns="3" units="crystal">\n'
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
            tag = f"VR.{ir+1}"
            f.write(f'      <{tag} type="complex" size="{dim*dim}">\n')
            flat = Hk[0, ir].flatten()
            for z in flat:
                f.write(f" {z.real:> .15E},{z.imag:> .15E}\n")
            f.write(f"      </{tag}>\n")

            if have_overlap:
                tag = f"OVERLAP.{ir+1}"
                f.write(f'      <{tag} type="complex" size="{dim*dim}">\n')
                flat = Sk[0, ir].flatten()
                for z in flat:
                    f.write(f" {z.real:> .15E},{z.imag:> .15E}\n")
                f.write(f"      </{tag}>\n")

        f.write("    </RHAM>\n")
        f.write("  </HAMILTONIAN>\n")
        f.write("</Root>\n")
