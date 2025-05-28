import numpy as np
from pathlib import Path

from typing import Dict


import xml.etree.ElementTree as ET
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
    ham_file = output_prefix + ".ham"

    Hk = hk_data["Hk"]
    Sk = hk_data.get("S", None)

    kpts = proj_data["kpts"]
    wk = proj_data["wk"]

    avec = lattice_data["avec"]
    bvec = lattice_data["bvec"]

    ivr = hk_data["ivr"]
    wr = hk_data["wr"]

    nspin = Hk.shape[0]
    nrtot = ivr.shape[0]

    root = ET.Element("HAMILTONIAN")

    def mat_to_text(matrix: np.ndarray) -> str:
        flat = matrix.flatten()
        if np.iscomplexobj(matrix):
            return "\n" + " ".join(f"{z.real:.12e} {z.imag:.12e}" for z in flat) + "\n"
        else:
            return "\n" + " ".join(f"{x:.12e}" for x in flat) + "\n"

    def add_array(parent, tag, array, attrib=None):
        elem = ET.SubElement(parent, tag)
        if attrib:
            for key, val in attrib.items():
                elem.set(key, val)
        elem.text = mat_to_text(array)
        return elem

    add_array(root, "DIRECT_LATTICE", avec.T, {"units": "bohr"})
    add_array(root, "RECIPROCAL_LATTICE", bvec.T, {"units": "bohr^-1"})
    add_array(root, "VKPT", kpts.T, {"units": "crystal"})
    add_array(root, "WK", wk)
    add_array(root, "IVR", ivr)
    add_array(root, "WR", wr)

    for isp in range(nspin):
        spin_tag = ET.SubElement(root, f"SPIN{iotk_index(isp + 1)}")
        rham = ET.SubElement(spin_tag, "RHAM")
        for ir in range(nrtot):
            rtag = f"VR{iotk_index(ir + 1)}"
            add_array(rham, rtag, Hk[isp, 0])

            if not do_orthoovp and Sk is not None:
                stag = f"OVERLAP{iotk_index(ir + 1)}"
                add_array(rham, stag, Sk[:, :, 0, isp])

    tree = ET.ElementTree(root)
    tree.write(ham_file, encoding="utf-8", xml_declaration=True)
