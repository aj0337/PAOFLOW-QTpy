import os
import re
import numpy as np
import xml.etree.ElementTree as ET
from typing import Optional, Dict
from scipy.linalg import eigh, inv

from PAOFLOW_QTpy.grid.get_rgrid import grids_get_rgrid
from PAOFLOW_QTpy.io.write_data import write_internal_format_files, iotk_index
from PAOFLOW_QTpy.io.write_header import headered_function
from PAOFLOW_QTpy.parsers.qexml import qexml_read_cell
from PAOFLOW_QTpy.io.log_module import log_rank0
from PAOFLOW_QTpy.utils.converters import cartesian_to_crystal

from PAOFLOW_QTpy.utils.timing import timed_function


@timed_function("atmproj_to_internal")
@headered_function("Conductor Initialization")
def parse_atomic_proj(
    *,
    input_dict: Optional[Dict] = None,
    file_proj: str,
    work_dir: str,
    prefix: str = "",
    postfix: str = "",
    atmproj_sh: float = 10.0,
    atmproj_thr: float = 0.0,
    atmproj_nbnd: Optional[int] = None,
    atmproj_do_norm: bool = False,
    do_orthoovp: bool = True,
    write_intermediate: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Parse atomic_proj.xml (QE projwfc.x output) and build internal Hamiltonian and projection data.

    Parameters
    ----------
    `input_dict` : dict, optional
        Optional input dictionary for additional parameters.
    `file_proj` : str
        Path to atomic_proj.xml.
    `work_dir` : str
        Working directory (typically ".").
    `prefix` : str
        QE calculation prefix (e.g., 'al5').
    `postfix` : str
        Suffix for output files (e.g., '_bulk').
    `atmproj_sh` : float
        Energy shift to avoid spurious zero modes.
    `atmproj_thr` : float
        Threshold for projectability filtering.
    `atmproj_nbnd` : int, optional
        Max number of bands to use. If None, use all available.
    `atmproj_do_norm` : bool
        Whether to normalize the projection vectors before constructing Hk.
    `do_orthoovp` : bool
        Whether to orthogonalize the overlap matrix.
    `write_intermediate` : bool
        Whether to write out .ham, .space, .wan intermediate files for debugging.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary with keys like 'Hk', 'S', 'eigvals', 'proj', etc.

    Notes
    -----
    The function parses QE atomic_proj.xml (via `parse_atomic_proj_xml`) and constructs
    the k-space Hamiltonian matrix `Hk` as:

        Hk(isp, ik) = Σ_{ib} eig(ib, ik, isp) ⋅ |proj_ib⟩⟨proj_ib|,

    where `proj_ib` is the projection vector ⟨atomic_wfc | Bloch_state(ib, ik, isp)⟩.
    Bands with energies above `atmproj_sh` or with projection weights below `atmproj_thr`
    are excluded from this sum. If `atmproj_do_norm` is True, the projection vectors are
    normalized prior to constructing `Hk`.

    The real-space Hamiltonian `Hr` is computed from `Hk` via:

        Hr(isp, r) = Σ_k w(k) ⋅ e^{i k ⋅ r} ⋅ Hk(isp, k),

    where `w(k)` are the k-point weights, and `r` are the real-space grid points.
    The overlaps `Sr` are computed analogously when `S` is available.
    """
    savedir = os.path.dirname(file_proj)
    file_data = os.path.join(savedir, "data-file.xml")
    if not os.path.exists(file_data):
        raise FileNotFoundError(f"Expected data-file.xml at: {file_data}")

    log_rank0(f"  {file_proj} file fmt: atmproj")

    lattice_data = qexml_read_cell(file_data)

    # --- Begin reading header info ---
    proj_data = parse_atomic_proj_xml(
        file_proj, lattice_data
    )  # Reads eigvals, proj, overlap

    # Preserve raw (QE) eigenvalues and Fermi energy
    proj_data["eigvals_raw"] = proj_data["eigvals"]
    proj_data["efermi_raw"] = proj_data["efermi"]

    # Determine conversion factor to eV
    energy_units = proj_data["energy_units"].lower()
    if energy_units in ("ha", "hartree", "au"):
        factor = 2 * 13.605693009  # Ha to eV
    elif energy_units in ("ry", "ryd", "rydberg"):
        factor = 13.605693009  # Ry to eV
    elif energy_units in ("ev", "electronvolt"):
        factor = 1.0
    else:
        raise ValueError(f"Unknown energy unit: {energy_units}")

    # Apply conversion and shift eigenvalues
    proj_data["efermi"] = proj_data["efermi_raw"] * factor
    proj_data["eigvals"] = proj_data["eigvals_raw"] * factor - proj_data["efermi"]

    log_rank0("  Dimensions found in atomic_proj.{dat,xml}:")
    log_rank0(f"    nbnd     : {proj_data['nbnd']:>5}")
    log_rank0(f"    nkpts    : {proj_data['nkpts']:>5}")
    log_rank0(f"    nspin    : {proj_data['nspin']:>5}")
    log_rank0(f"    natomwfc : {proj_data['natomwfc']:>5}")
    log_rank0(f"    nelec    : {proj_data['nelec']:>12.6f}")
    log_rank0(f"    efermi   : {proj_data['efermi_raw']:>12.6f}")
    log_rank0(f"    energy_units :  {proj_data['energy_units']}   ")
    log_rank0("")
    log_rank0("  ATMPROJ conversion to be done using:")
    log_rank0(
        f"    atmproj_nbnd : {atmproj_nbnd if atmproj_nbnd is not None else proj_data['nbnd']:>5}"
    )
    log_rank0(f"    atmproj_thr  : {atmproj_thr:>12.6f}")
    log_rank0(f"    atmproj_sh   : {atmproj_sh:>12.6f}")
    log_rank0(f"    atmproj_do_norm:  {atmproj_do_norm}")
    # --- Begin main data read ---
    log_rank0("Begins atmproj_read_ext --massive data")
    if do_orthoovp:
        log_rank0("Using an orthogonal basis. do_orthoovp=.true.")

    hk_data = build_hamiltonian_from_proj(
        proj_data,
        atmproj_sh=atmproj_sh,
        atmproj_thr=atmproj_thr,
        atmproj_nbnd=atmproj_nbnd,
        do_orthoovp=do_orthoovp,
    )
    log_rank0("Ends atmproj_read_ext --massive data")

    Hk = hk_data["Hk"]
    Sk = hk_data.get("S", None)

    nk = np.zeros(3, dtype=int)
    nk[0], nk[1], nk[2] = 1, 1, 4  # TODO : Why is this hardcoded?
    nr = nk
    ivr, wr = grids_get_rgrid(nr)

    hk_data["ivr"] = ivr
    hk_data["wr"] = wr
    hk_data["nk"] = nk
    hk_data["nr"] = nr

    if write_intermediate:
        output_dir = os.path.join(work_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        output_prefix = os.path.join(output_dir, prefix + postfix)
        write_internal_format_files(
            output_prefix, hk_data, proj_data, lattice_data, do_orthoovp
        )

        log_rank0(f"{file_proj} converted from ATMPROJ to internal fmt")

        proj = proj_data["proj"]
        eigvals = proj_data["eigvals"]

        nspin, nkpts, natomwfc, _ = Hk.shape
        nbnd = proj.shape[1]

        for isp in range(nspin):
            if nspin == 2:
                proj_file = os.path.join(
                    output_dir, f"projectability_{['up', 'dn'][isp]}.txt"
                )
            else:
                proj_file = os.path.join(output_dir, "projectability.txt")

            with open(proj_file, "w") as f:
                f.write("# Energy (eV)        Projectability\n")
                for ik in range(nkpts):
                    for ib in range(nbnd):
                        proj_vec = proj[:, ib, ik, isp]
                        weight = np.vdot(proj_vec, proj_vec).real
                        energy = eigvals[ib, ik, isp]
                        f.write(f"{energy:20.13f}  {weight:20.13f}\n")
        log_rank0("Printed projectabilities to projectability.txt")

        if not do_orthoovp and Sk is not None:
            kovp_file = os.path.join(output_dir, "kovp.txt")
            with open(kovp_file, "w") as f:
                f.write("# Overlap Real        Overlap Imag\n")
                for ik in range(nkpts):
                    for isp in range(nspin):
                        mat = Sk[:, :, ik, isp]
                        for i in range(natomwfc):
                            for j in range(natomwfc):
                                f.write(
                                    f"{mat[i, j].real:20.13f}  {mat[i, j].imag:20.13f}\n"
                                )
            log_rank0("Printed overlap matrices to kovp.txt")

    return hk_data


def parse_atomic_proj_xml(file_proj: str, lattice_data: Dict) -> Dict:
    """
    Parse the Quantum ESPRESSO atomic_proj.xml file (from projwfc.x) into structured NumPy arrays.

    Parameters
    ----------
    file_proj : str
        Path to the atomic_proj.xml file generated by QE's projwfc.x. This XML file contains
        the bandstructure eigenvalues, k-point grid, projection data, and optionally, overlaps.
    lattice_data : dict
        Dictionary containing lattice parameters, including:
        - `alat` : float
            Lattice constant in Bohr.
        - `bvec` : (3, 3) ndarray
            Reciprocal lattice vectors in columns (Bohr^-1).

    Returns
    -------
    dict
        A dictionary containing:
        - `nbnd` : int
            Number of bands.
        - `nkpts` : int
            Number of k-points.
        - `nspin` : int
            Number of spin components (1 for non-magnetic, 2 for collinear magnetic calculations).
        - `natomwfc` : int
            Number of atomic wavefunctions (projectors) in the system.
        - `nelec` : float
            Number of electrons in the system.
        - `efermi` : float
            Fermi energy in the units specified by `energy_units`.
        - `energy_units` : str
            Units of energy (e.g., 'eV', 'Ha', 'Ry') as specified in the XML file.
        - `kpts` : ndarray of shape (3, nkpts)
            K-point coordinates in crystal units, transposed for consistency with PAOFLOW conventions.
        - `wk` : ndarray of shape (nkpts,)
            K-point weights.
        - `eigvals` : ndarray of shape (nbnd, nkpts, nspin)
            Eigenvalues of the bands at each k-point and spin.
        - `proj` : ndarray of shape (natomwfc, nbnd, nkpts, nspin), complex
            Projection matrix elements ⟨atomic_wfc | Bloch_state⟩.
        - `overlap` : ndarray of shape (natomwfc, natomwfc, nkpts, nspin), complex or None
            Overlap matrices S_{ij}(k) = ⟨atomic_wfc_i | atomic_wfc_j⟩ if present in the XML; else None.

    Notes
    -----
    The function replicates the logic of the Fortran `atmproj_read_ext` routine from the PAOFLOW transport code,
    faithfully reproducing the Fortran logging flow:
    - Logs the start of eigenvalue and projection parsing.
    - Reads XML sections: HEADER, K-POINTS, WEIGHT_OF_K-POINTS, EIGENVALUES, PROJECTIONS, and optional OVERLAPS.
    - Supports both single and spin-polarized cases (1 or 2 spin channels).

    The projection matrix `proj` stores complex coefficients:
        `proj[ias, ib, ik, isp] = ⟨ atomic_wfc (ias) | Bloch_state (ib, ik, isp) ⟩`.

    The eigenvalues `eigvals` are:
        `eigvals[ib, ik, isp] = E_{ib}(k, spin)`, where `ib` is the band index.

    The overlap matrices `overlap` (if present) are:
        `overlap[i, j, ik, isp] = ⟨ atomic_wfc_i | atomic_wfc_j ⟩ at k-point ik, spin isp`.

    All arrays are stored in NumPy-friendly formats with dtype float64 or complex128.

    The XML structure parsed corresponds to QE's projwfc.x output:
    - HEADER block with general info.
    - K-POINTS and WEIGHT_OF_K-POINTS blocks.
    - EIGENVALUES and PROJECTIONS blocks, nested by k-points and optionally by spin.
    - OVERLAPS block if present.
    """

    log_rank0("Begins atmproj_read_ext")
    log_rank0("Begins reading eigenvalues")

    tree = ET.parse(file_proj)
    root = tree.getroot()

    # Read header information
    header = root.find("HEADER")
    nbnd = int(header.findtext("NUMBER_OF_BANDS"))
    nkpt = int(header.findtext("NUMBER_OF_K-POINTS"))
    nspin = int(header.findtext("NUMBER_OF_SPIN_COMPONENTS"))
    natomwfc = int(header.findtext("NUMBER_OF_ATOMIC_WFC"))
    nelec = float(header.findtext("NUMBER_OF_ELECTRONS"))
    efermi = float(header.findtext("FERMI_ENERGY"))
    energy_units = header.find("UNITS_FOR_ENERGY").attrib["UNITS"]

    # Read kpoints and weights (if present)
    kpoints = np.array(
        [
            [float(val) for val in line.strip().split()]
            for line in root.find("K-POINTS").text.strip().split("\n")
        ]
    ).T
    wk = np.array(
        [float(val) for val in root.find("WEIGHT_OF_K-POINTS").text.strip().split()]
    )
    wk_sum = np.sum(wk)
    wk = wk / wk_sum
    vkpts = kpoints * 2 * np.pi / lattice_data["alat"]
    vkpts_crystal = cartesian_to_crystal(
        vkpts, lattice_data["bvec"]
    )  # convert to crystal coordinates

    # === Eigenvalues ===
    eigvals = np.zeros((nbnd, nkpt, nspin))
    eig_section = root.find("EIGENVALUES")
    for ik, kpoint in enumerate(eig_section):
        for isp in range(nspin):
            spin_tag = f"EIG{iotk_index(isp + 1)}" if nspin > 1 else "EIG"
            eig_tag = kpoint.find(spin_tag) if nspin > 1 else kpoint.find("EIG")
            eigvals[:, ik, isp] = [float(x) for x in eig_tag.text.strip().split()]

    log_rank0("Finished reading eigenvalues")

    # === Projections ===
    log_rank0("Begins reading projections")
    proj = np.zeros((natomwfc, nbnd, nkpt, nspin), dtype=np.complex128)
    projections_section = root.find("PROJECTIONS")
    for ik, kpoint in enumerate(projections_section):
        for isp in range(nspin):
            spin_node = (
                kpoint.find(f"SPIN{iotk_index(isp + 1)}") if nspin == 2 else kpoint
            )
            for ias in range(natomwfc):
                tag = f"ATMWFC{iotk_index(ias + 1)}"
                for ib in range(nbnd):
                    data = re.split(r"[\s,]+", spin_node.find(tag).text.strip())
                    real, im = float(data[2 * ib]), float(data[2 * ib + 1])
                    proj[ias, ib, ik, isp] = real + 1j * im

    log_rank0("Ends reading projections")
    log_rank0("Ends atmproj_read_ext")

    # === Overlap ===
    overlap_section = root.find("OVERLAPS")
    overlap = None
    if overlap_section is not None:
        overlap = np.zeros((natomwfc, natomwfc, nkpt, nspin), dtype=np.complex128)
        for ik, kpoint in enumerate(overlap_section):
            for isp in range(nspin):
                tag = f"OVERLAP{iotk_index(isp + 1)}"
                data = re.split(r"[\s,]+", kpoint.find(tag).text.strip())
                matrix = np.array(
                    [
                        complex(float(data[i]), float(data[i + 1]))
                        for i in range(0, len(data), 2)
                    ]
                )
                overlap[:, :, ik, isp] = matrix.reshape(natomwfc, natomwfc)

    return {
        "nbnd": nbnd,
        "nkpts": nkpt,
        "nspin": nspin,
        "natomwfc": natomwfc,
        "nelec": nelec,
        "efermi": efermi,
        "energy_units": energy_units,
        "kpts": kpoints,
        "vkpts_cartesian": vkpts,
        "vkpts_crystal": vkpts_crystal,
        "wk": wk,
        "eigvals": eigvals,
        "proj": proj,
        "overlap": overlap,
    }


def build_hamiltonian_from_proj(
    proj_data: Dict,
    atmproj_sh: float,
    atmproj_thr: float,
    atmproj_nbnd: Optional[int],
    do_orthoovp: bool,
    atmproj_do_norm: bool = False,
    shifting_scheme: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Construct H(k) from projection data.

    Parameters
    ----------
    `proj_data` : Dict
        Output from parse_atomic_proj_xml.
    `atmproj_sh` : float
        Energy shift used as a band filter.
    `atmproj_thr` : float
        Minimum projector weight to include a band.
    `atmproj_nbnd` : int or None
        Maximum number of bands to include.
    `do_orthoovp` : bool
        If False, include the non-orthogonal overlaps. If True, projector basis has been orthonormalized.
    `atmproj_do_norm` : bool
        If True, normalize the projectors.
    `shifting_scheme` : int
        1 = direct sum over projectors; 2 = projected subspace with inverse PA

    Returns
    -------
    Dict[str, np.ndarray]
        Includes keys:
        - 'Hk': complex ndarray, shape (nspin, nkpts, natomwfc, natomwfc)
        - 'S' : complex ndarray, shape (natomwfc, natomwfc, nkpts, nspin) if available

    Notes
    -----
    Two schemes are used for constructing the k-dependent Hamiltonian:

    1. **Shifting scheme 1 (default):**

       The Hamiltonian is constructed from the outer product of the projectors:

           H(k) = ∑_b [eig_b(k) - κ] · |proj_b(k)><proj_b(k)|

       where the summation runs over bands b with eigenvalues below `atmproj_sh` κ.
       Optional projector normalization is applied before the outer product.

    2. **Shifting scheme 2 (subspace projection):**

       Let A be the matrix whose columns are selected projectors for eigenvalues < κ:

           A_{i b} = <ϕ_i | ψ_b>

       Let E be a diagonal matrix with corresponding eigenvalues:

           E_{bb} = eig_b(k)

       Define the projector overlap:

           PA = A⁺ · A

       Compute its inverse, IPA = (A⁺ A)⁻¹. Then construct:

           H_aux = (E - κ IPA) · A⁺
           H(k) = A · H_aux = A · (E - κ IPA) · A⁺

       This is a more accurate low-rank projection of the Hamiltonian into the atomic subspace.

    In both cases, a final shift of +κ is added to the Hamiltonian either as:
    - H(k) += κ · S(k) for non-orthogonal basis
    - H(k) += κ · I for orthonormal basis

    If `do_orthoovp` is False, an additional transformation is applied:
        H(k) -> S(k)½ · H(k) · S(k)½
    where S(k)½ is the square root of the overlap matrix.
    """
    eig = proj_data["eigvals"]
    proj = proj_data["proj"]
    S_raw = proj_data["overlap"]

    nbnd = proj_data["nbnd"]
    nkpts = proj_data["nkpts"]
    nspin = proj_data["nspin"]
    natomwfc = proj_data["natomwfc"]

    atmproj_nbnd_ = (
        min(atmproj_nbnd, nbnd) if atmproj_nbnd and atmproj_nbnd > 0 else nbnd
    )
    Hk = np.zeros((nspin, nkpts, natomwfc, natomwfc), dtype=np.complex128)
    Sk = S_raw.copy() if not do_orthoovp and S_raw is not None else None

    for isp in range(nspin):
        for ik in range(nkpts):
            if shifting_scheme == 2:
                mask = [ib for ib in range(nbnd) if eig[ib, ik, isp] < atmproj_sh]
                if not mask:
                    raise RuntimeError(f"No eigenvalues below shift at ik = {ik}")

                E = np.diag(eig[mask, ik, isp])
                A = proj[:, mask, ik, isp].copy()

                if atmproj_do_norm:
                    for ib in range(A.shape[1]):
                        norm = np.vdot(A[:, ib], A[:, ib]).real
                        A[:, ib] /= np.sqrt(norm)

                PA = A.conj().T @ A
                IPA = inv(PA)
                H_aux = (E - atmproj_sh * IPA) @ A.conj().T
                Hk[isp, ik] = A @ H_aux

                Hk[isp, ik] = 0.5 * (Hk[isp, ik] + Hk[isp, ik].conj().T)

            else:
                for ib in range(atmproj_nbnd_):
                    if atmproj_thr > 0.0:
                        proj_b = proj[:, ib, ik, isp]
                        weight = np.vdot(proj_b, proj_b).real
                        if eig[ib, ik, isp] > atmproj_sh:
                            continue
                    else:
                        if eig[ib, ik, isp] >= atmproj_sh:
                            continue
                        proj_b = proj[:, ib, ik, isp]
                        weight = np.vdot(proj_b, proj_b).real

                    if atmproj_do_norm:
                        proj_b /= np.sqrt(weight)

                    Hk[isp, ik] += (eig[ib, ik, isp] - atmproj_sh) * np.outer(
                        proj_b, proj_b.conj()
                    )
                Hk[isp, ik] = 0.5 * (Hk[isp, ik] + Hk[isp, ik].conj().T)

            if not do_orthoovp and S_raw is not None:
                S = S_raw[:, :, ik, isp]
                w, U = eigh(S)
                if np.any(w <= 0):
                    raise ValueError(
                        f"Negative or zero overlap eigenvalue at ik={ik}, isp={isp}"
                    )
                sqrtS = U @ np.diag(np.sqrt(w)) @ U.conj().T
                Hk[isp, ik] = sqrtS @ Hk[isp, ik] @ sqrtS.conj().T

            if not do_orthoovp and S_raw is not None:
                Hk[isp, ik] += atmproj_sh * S_raw[:, :, ik, isp]
            else:
                Hk[isp, ik] += atmproj_sh * np.eye(natomwfc, dtype=np.complex128)

            if not np.allclose(Hk[isp, ik], Hk[isp, ik].conj().T, atol=1e-8):
                raise ValueError(f"Hk not Hermitian at ik={ik}, isp={isp}")

    return {"Hk": Hk, "S": Sk}
