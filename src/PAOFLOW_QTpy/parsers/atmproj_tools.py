import os
import numpy as np
from typing import Optional, Dict
from compute_rham import compute_rham
from get_rgrid import grids_get_rgrid


import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)


def parse_atomic_proj(
    file_proj: str,
    work_dir: str,
    prefix: str,
    postfix: str,
    atmproj_sh: float = 10.0,
    atmproj_thr: float = 0.0,
    atmproj_nbnd: Optional[int] = None,
    do_orthoovp: bool = False,
    write_intermediate: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Parse atomic_proj.xml (QE projwfc.x output) and build internal Hamiltonian and projection data.

    Parameters
    ----------
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
    `do_orthoovp` : bool
        Whether to orthogonalize the overlap matrix.
    `write_intermediate` : bool
        Whether to write out .ham, .space, .wan intermediate files for debugging.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary with keys like 'Hk', 'S', 'eigvals', 'proj', etc.
    """

    savedir = os.path.dirname(file_proj)
    file_data = os.path.join(savedir, "data-file.xml")
    if not os.path.exists(file_data):
        raise FileNotFoundError(f"Expected data-file.xml at: {file_data}")

    lattice_data = parse_data_file(file_data)

    proj_data = parse_atomic_proj_xml(file_proj)

    hk_data = build_hamiltonian_from_proj(
        proj_data,
        lattice_data,
        atmproj_sh=atmproj_sh,
        atmproj_thr=atmproj_thr,
        atmproj_nbnd=atmproj_nbnd,
        do_orthoovp=do_orthoovp,
    )

    Hk = hk_data["Hk"]
    Sk = hk_data.get("S", None)

    kpts = proj_data["kpts"]
    wk = proj_data["wk"]
    avec = lattice_data["avec"]
    nr = np.array([1, 1, kpts.shape[1]])
    ivr, wr = grids_get_rgrid(nr)

    vr = (avec @ ivr.T).T

    nspin, nkpts, n, _ = Hk.shape
    nR = vr.shape[0]
    Hr = np.zeros((nspin, nR, n, n), dtype=np.complex128)
    Sr = (
        np.zeros((nspin, nR, n, n), dtype=np.complex128)
        if (not do_orthoovp and Sk is not None)
        else None
    )

    for isp in range(nspin):
        for ir in range(nR):
            Hr[isp, ir] = compute_rham(vr[ir], Hk[isp], kpts, wk)
            if Sr is not None:
                Sr[isp, ir] = compute_rham(vr[ir], Sk[..., isp], kpts, wk)

    hk_data["Hr"] = Hr
    if Sr is not None:
        hk_data["Sr"] = Sr
    hk_data["ivr"] = ivr
    hk_data["wr"] = wr
    hk_data["vr"] = vr

    if write_intermediate:
        output_prefix = os.path.join(work_dir, prefix + postfix)
        write_internal_format_files(
            output_prefix, hk_data, proj_data, lattice_data, do_orthoovp
        )

    return hk_data


def parse_data_file(file_data: str) -> Dict:
    """
    Parse data-file.xml to extract lattice vectors, atomic species, and positions.

    Parameters
    ----------
    `file_data` : str
        Path to QE's data-file.xml

    Returns
    -------
    Dict
        Dictionary with keys: 'alat', 'avec', 'bvec', 'tau', 'atm_symb', 'ityp', 'nat', 'nsp'
    """
    ns = {"qes": "http://www.quantum-espresso.org/ns/qes"}
    tree = ET.parse(file_data)
    root = tree.getroot()

    def read_vec3(elem):
        return np.array([float(x) for x in elem.text.strip().split()])

    alat = float(
        root.find(".//qes:CELL/qes:UNITS_FOR_LATTICE_PARAMETER", ns).attrib["alat"]
    )
    a1 = read_vec3(root.find(".//qes:CELL/qes:A1", ns))
    a2 = read_vec3(root.find(".//qes:CELL/qes:A2", ns))
    a3 = read_vec3(root.find(".//qes:CELL/qes:A3", ns))
    avec = np.vstack([a1, a2, a3]).T

    b1 = read_vec3(root.find(".//qes:CELL/qes:B1", ns))
    b2 = read_vec3(root.find(".//qes:CELL/qes:B2", ns))
    b3 = read_vec3(root.find(".//qes:CELL/qes:B3", ns))
    bvec = np.vstack([b1, b2, b3]).T

    atoms = root.findall(".//qes:IONS/qes:ATOM", ns)
    nat = len(atoms)
    tau = np.zeros((3, nat))
    ityp = np.zeros(nat, dtype=int)
    atm_symb = []

    for i, atom in enumerate(atoms):
        tau[:, i] = read_vec3(atom.find("qes:TAU", ns))
        atm_symb.append(atom.attrib["SPECIES"])
        ityp[i] = int(atom.attrib["ITYP"])

    nsp = max(ityp)

    return {
        "alat": alat,
        "avec": avec,
        "bvec": bvec,
        "tau": tau,
        "atm_symb": atm_symb,
        "ityp": ityp,
        "nat": nat,
        "nsp": nsp,
    }


def parse_atomic_proj_xml(file_proj: str) -> Dict:
    """
    Parse atomic_proj.xml to extract band structure and projection data.

    Parameters
    ----------
    `file_proj` : str
        Path to atomic_proj.xml.

    Returns
    -------
    Dict
        Dictionary with keys:
        - 'nbnd' : int
        - 'nkpts' : int
        - 'nspin' : int
        - 'natomwfc' : int
        - 'nelec' : float
        - 'efermi' : float
        - 'energy_units' : str
        - 'eigvals' : ndarray, shape (nbnd, nkpts, nspin)
        - 'proj' : ndarray, shape (natomwfc, nbnd, nkpts, nspin)
        - 'overlap' : ndarray or None, shape (natomwfc, natomwfc, nkpts, nspin)
        - 'kpts' : ndarray, shape (3, nkpts)
        - 'wk' : ndarray, shape (nkpts,)
    """
    tree = ET.parse(file_proj)
    root = tree.getroot()

    def get_text(tag):
        return root.find(tag).text.strip()

    nbnd = int(get_text("HEADER/NUMBER_OF_BANDS"))
    nkpts = int(get_text("HEADER/NUMBER_OF_K-POINTS"))
    nspin = int(get_text("HEADER/NUMBER_OF_SPIN_COMPONENTS"))
    natomwfc = int(get_text("HEADER/NUMBER_OF_ATOMIC_WFC"))
    nelec = float(get_text("HEADER/NUMBER_OF_ELECTRONS"))
    efermi = float(get_text("HEADER/FERMI_ENERGY"))
    energy_units = root.find("HEADER/UNITS_FOR_ENERGY").attrib["UNITS"]

    eigvals = np.zeros((nbnd, nkpts, nspin))
    proj = np.zeros((natomwfc, nbnd, nkpts, nspin), dtype=np.complex128)
    overlap = np.zeros((natomwfc, natomwfc, nkpts, nspin), dtype=np.complex128)

    kpts = np.zeros((3, nkpts))
    wk = np.zeros(nkpts)

    for ik, kp in enumerate(root.find("K-POINTS").text.strip().split("\n")):
        kpts[:, ik] = [float(x) for x in kp.strip().split()]

    for ik, w in enumerate(root.find("WEIGHT_OF_K-POINTS").text.strip().split()):
        wk[ik] = float(w)

    for ik, kpoint in enumerate(root.find("EIGENVALUES")):
        for isp in range(nspin):
            spin_tag = f"EIG{iotk_index(isp + 1)}" if nspin > 1 else "EIG"
            eig_tag = kpoint.find(spin_tag)
            eigvals[:, ik, isp] = [float(e) for e in eig_tag.text.strip().split()]

    for ik, kpoint in enumerate(root.find("PROJECTIONS")):
        for isp in range(nspin):
            spin_node = (
                kpoint.find(f"SPIN{iotk_index(isp + 1)}") if nspin > 1 else kpoint
            )
            for ia in range(natomwfc):
                tag = f"ATMWFC{iotk_index(ia + 1)}"
                line = spin_node.find(tag).text.strip().split()
                for ib in range(nbnd):
                    re, im = float(line[2 * ib]), float(line[2 * ib + 1])
                    proj[ia, ib, ik, isp] = re + 1j * im

    ovlp_root = root.find("OVERLAPS")
    if ovlp_root is not None:
        for ik, kpoint in enumerate(ovlp_root):
            for isp in range(nspin):
                tag = f"OVERLAP{iotk_index(isp + 1)}"
                data = kpoint.find(tag).text.strip().split()
                mat = np.array(
                    [
                        complex(float(data[i]), float(data[i + 1]))
                        for i in range(0, len(data), 2)
                    ]
                )
                overlap[:, :, ik, isp] = mat.reshape((natomwfc, natomwfc))

    return {
        "nbnd": nbnd,
        "nkpts": nkpts,
        "nspin": nspin,
        "natomwfc": natomwfc,
        "nelec": nelec,
        "efermi": efermi,
        "energy_units": energy_units,
        "eigvals": eigvals,
        "proj": proj,
        "overlap": overlap if ovlp_root is not None else None,
        "kpts": kpts,
        "wk": wk,
    }


def iotk_index(n: int) -> str:
    """Return IOTK index tag used in XML labels (e.g., 1 → '001')."""
    return f"{n:03d}"


def build_hamiltonian_from_proj(
    proj_data: Dict,
    atmproj_sh: float,
    atmproj_thr: float,
    do_orthoovp: bool,
) -> Dict[str, np.ndarray]:
    """
    Construct H(k) from projection data.

    Parameters
    ----------
    `proj_data` : Dict
        Output from parse_atomic_proj_xml.
    `lattice_data` : Dict
        Output from parse_data_file.
    `atmproj_sh` : float
        Energy shift used as a band filter.
    `atmproj_thr` : float
        Minimum projector weight to include a band.
    `atmproj_nbnd` : int or None
        Maximum number of bands to include.
    `do_orthoovp` : bool
        If False, include the non-orthogonal overlaps. If True, the projector basis has been orthonormalized, i.e., S = I.

    Returns
    -------
    Dict[str, np.ndarray]
        Includes keys:
        - 'Hk': complex ndarray, shape (nspin, nkpts, natomwfc, natomwfc)
        - 'S' : complex ndarray, shape (nspin, nkpts, natomwfc, natomwfc) if available
    """
    nbnd = proj_data["nbnd"]
    nkpts = proj_data["nkpts"]
    nspin = proj_data["nspin"]
    natomwfc = proj_data["natomwfc"]

    eig = proj_data["eigvals"]
    proj = proj_data["proj"]
    S_raw = proj_data["overlap"]

    Hk = np.zeros((nspin, nkpts, natomwfc, natomwfc), dtype=np.complex128)
    Sk = None

    if not do_orthoovp and S_raw is not None:
        Sk = np.copy(S_raw)

    for isp in range(nspin):
        for ik in range(nkpts):
            for ib in range(nbnd):
                if eig[ib, ik, isp] >= atmproj_sh:
                    continue

                proj_b = proj[:, ib, ik, isp]
                weight = np.vdot(proj_b, proj_b).real

                if atmproj_thr > 0.0 and weight < atmproj_thr:
                    continue

                outer = np.outer(proj_b, proj_b.conj())
                Hk[isp, ik] += eig[ib, ik, isp] * outer

    return {
        "Hk": Hk,
        "S": Sk,
    }


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

    ivr = np.indices(kpts.shape[1:]).reshape(3, -1).T
    wr = np.ones(ivr.shape[0]) / ivr.shape[0]

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
