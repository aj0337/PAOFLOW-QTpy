from __future__ import annotations

import re

import numpy as np
import xml.etree.ElementTree as ET

from PAOFLOW_QTpy.io.write_data import iotk_index
from PAOFLOW_QTpy.utils.converters import cartesian_to_crystal


# -------------------------
# Dispatch / format detection
# -------------------------
def _is_new_atomic_proj(root: ET.Element) -> bool:
    """
    Newer QE atomic_proj.xml typically has:
      - <HEADER ...attributes.../>
      - <EIGENSTATES> with repeating K-POINT / E / PROJS triplets
    Legacy (QE 5.3-ish) typically has:
      - <HEADER> with nested tags NUMBER_OF_BANDS, ...
      - <K-POINTS>, <WEIGHT_OF_K-POINTS>, <EIGENVALUES>, <PROJECTIONS>, ...
    """
    if root.find("EIGENSTATES") is not None:
        return True
    header = root.find("HEADER")
    if header is not None and len(header.attrib) > 0:
        return True
    return False


# -------------------------
# NEW atomic_proj.xml helpers
# -------------------------
def _iter_kpoint_blocks(eigenstates: ET.Element):
    """
    New atomic_proj.xml EIGENSTATES is a repeating sequence of:
      K-POINT, E, PROJS
    """
    kids = list(eigenstates)
    if len(kids) % 3 != 0:
        raise ValueError(
            "Unexpected <EIGENSTATES> layout (expected K-POINT/E/PROJS triplets)"
        )
    for i in range(0, len(kids), 3):
        kp, e, projs = kids[i], kids[i + 1], kids[i + 2]
        if kp.tag != "K-POINT" or e.tag != "E" or projs.tag != "PROJS":
            raise ValueError(
                "Unexpected <EIGENSTATES> layout (expected K-POINT/E/PROJS triplets)"
            )
        yield kp, e, projs


# -------------------------
# Public API (combined)
# -------------------------
def parse_header(root: ET.Element) -> dict:
    if _is_new_atomic_proj(root):
        header = root.find("HEADER")
        if header is None:
            raise ValueError("Missing <HEADER> in atomic_proj.xml")

        return {
            "nbnd": int(header.attrib["NUMBER_OF_BANDS"]),
            "nkpts": int(header.attrib["NUMBER_OF_K-POINTS"]),
            "nspin": int(header.attrib["NUMBER_OF_SPIN_COMPONENTS"]),
            "natomwfc": int(header.attrib["NUMBER_OF_ATOMIC_WFC"]),
            "nelec": float(header.attrib["NUMBER_OF_ELECTRONS"]),
            "efermi": float(header.attrib["FERMI_ENERGY"]),
            # In practice many atomic_proj.xml outputs behave like Ry.
            # If you have a reliable source of units elsewhere, wire it in.
            "energy_units": "Rydberg",
        }

    # Legacy (QE 5.3-ish)
    header = root.find("HEADER")
    if header is None:
        raise ValueError("Missing <HEADER> in legacy atomic_proj.xml")

    return {
        "nbnd": int(header.findtext("NUMBER_OF_BANDS")),
        "nkpts": int(header.findtext("NUMBER_OF_K-POINTS")),
        "nspin": int(header.findtext("NUMBER_OF_SPIN_COMPONENTS")),
        "natomwfc": int(header.findtext("NUMBER_OF_ATOMIC_WFC")),
        "nelec": float(header.findtext("NUMBER_OF_ELECTRONS")),
        "efermi": float(header.findtext("FERMI_ENERGY")),
        "energy_units": header.find("UNITS_FOR_ENERGY").attrib["UNITS"],
    }


def parse_kpoints(root: ET.Element, lattice_data: dict) -> dict:
    alat = float(lattice_data["alat"])
    bvec = np.asarray(lattice_data["bvec"], dtype=float)  # columns b1,b2,b3 in bohr^-1

    if _is_new_atomic_proj(root):
        eigenstates = root.find("EIGENSTATES")
        if eigenstates is None:
            raise ValueError("Missing <EIGENSTATES> in atomic_proj.xml")

        kpts_list = []
        wk_list = []

        for kp_el, _, _ in _iter_kpoint_blocks(eigenstates):
            if kp_el.text is None:
                raise ValueError("<K-POINT> is empty")

            # IMPORTANT CHANGE:
            # treat K-POINT as cartesian components in units of (2π/alat),
            # same as legacy K-POINTS block
            k = np.fromstring(kp_el.text, sep=" ")
            if k.size != 3:
                raise ValueError("<K-POINT> does not contain 3 numbers")

            kpts_list.append(k)
            wk_list.append(float(kp_el.attrib.get("Weight", "1.0")))

        kpts = np.array(kpts_list, dtype=float).T  # (3, nkpts)
        wk = np.array(wk_list, dtype=float)
        wk = wk / np.sum(wk)

        vkpts = kpts * (2.0 * np.pi / alat)  # bohr^-1
        vkpts_crystal = cartesian_to_crystal(vkpts, bvec)

        return {
            "kpts": kpts,
            "wk": wk,
            "vkpts_cartesian": vkpts,
            "vkpts_crystal": vkpts_crystal,
        }

    # Legacy (QE 5.3-ish): already in the old convention
    kpoints = np.array(
        [
            [float(val) for val in line.strip().split()]
            for line in root.find("K-POINTS").text.strip().split("\n")
        ]
    ).T
    wk = np.array(
        [float(val) for val in root.find("WEIGHT_OF_K-POINTS").text.strip().split()]
    )
    wk = wk / np.sum(wk)

    vkpts = kpoints * 2.0 * np.pi / alat
    vkpts_crystal = cartesian_to_crystal(vkpts, bvec)

    return {
        "kpts": kpoints,
        "wk": wk,
        "vkpts_cartesian": vkpts,
        "vkpts_crystal": vkpts_crystal,
    }


def parse_eigenvalues(
    root: ET.Element, nbnd: int, nkpts: int, nspin: int
) -> np.ndarray:
    if _is_new_atomic_proj(root):
        eigenstates = root.find("EIGENSTATES")
        if eigenstates is None:
            raise ValueError("Missing <EIGENSTATES> in atomic_proj.xml")

        eigvals = np.zeros((nbnd, nkpts, nspin), dtype=float)

        ik = 0
        for _, e_el, _ in _iter_kpoint_blocks(eigenstates):
            if e_el.text is None:
                raise ValueError("<E> is empty")
            vals = np.fromstring(e_el.text, sep=" ")

            if vals.size == nbnd:
                eigvals[:, ik, 0] = vals
                if nspin == 2:
                    eigvals[:, ik, 1] = vals
            elif vals.size == nbnd * nspin:
                vals = vals.reshape((nspin, nbnd))
                for isp in range(nspin):
                    eigvals[:, ik, isp] = vals[isp]
            else:
                raise ValueError(
                    f"Unexpected number of eigenvalues at ik={ik}: got {vals.size}, expected {nbnd} or {nbnd * nspin}"
                )

            ik += 1

        if ik != nkpts:
            raise ValueError(f"Parsed {ik} k-points, but header says nkpts={nkpts}")

        return eigvals

    # Legacy
    eigvals = np.zeros((nbnd, nkpts, nspin), dtype=float)
    eig_section = root.find("EIGENVALUES")
    if eig_section is None:
        raise ValueError("Missing <EIGENVALUES> in legacy atomic_proj.xml")

    for ik, kpoint in enumerate(eig_section):
        for isp in range(nspin):
            spin_tag = f"EIG{iotk_index(isp + 1)}" if nspin > 1 else "EIG"
            eig_tag = kpoint.find(spin_tag)
            if eig_tag is None or eig_tag.text is None:
                raise ValueError(f"Missing <{spin_tag}> eigenvalues at ik={ik}")
            eigvals[:, ik, isp] = np.fromstring(eig_tag.text, sep=" ")

    return eigvals


def parse_projections(
    root: ET.Element, nbnd: int, nkpts: int, nspin: int, natomwfc: int
) -> np.ndarray:
    if _is_new_atomic_proj(root):
        eigenstates = root.find("EIGENSTATES")
        if eigenstates is None:
            raise ValueError("Missing <EIGENSTATES> in atomic_proj.xml")

        proj = np.zeros((natomwfc, nbnd, nkpts, nspin), dtype=np.complex128)

        ik = 0
        for _, _, projs_el in _iter_kpoint_blocks(eigenstates):
            for awfc_el in list(projs_el):
                if awfc_el.tag != "ATOMIC_WFC":
                    continue
                if awfc_el.text is None:
                    raise ValueError("<ATOMIC_WFC> is empty")

                ias = int(awfc_el.attrib["index"]) - 1
                isp = int(awfc_el.attrib.get("spin", "1")) - 1

                data = np.fromstring(awfc_el.text, sep=" ")
                if data.size != 2 * nbnd:
                    raise ValueError(
                        f"Unexpected projection length at ik={ik}, ias={ias + 1}, spin={isp + 1}: "
                        f"got {data.size}, expected {2 * nbnd}"
                    )

                proj[ias, :, ik, isp] = data[0::2] + 1j * data[1::2]

            ik += 1

        if ik != nkpts:
            raise ValueError(f"Parsed {ik} k-points, but header says nkpts={nkpts}")

        return proj

    # Legacy
    proj = np.zeros((natomwfc, nbnd, nkpts, nspin), dtype=np.complex128)
    projections_section = root.find("PROJECTIONS")
    if projections_section is None:
        raise ValueError("Missing <PROJECTIONS> in legacy atomic_proj.xml")

    for ik, kpoint in enumerate(projections_section):
        for isp in range(nspin):
            spin_node = (
                kpoint.find(f"SPIN{iotk_index(isp + 1)}") if nspin == 2 else kpoint
            )
            if spin_node is None:
                raise ValueError(f"Missing SPIN node at ik={ik}, isp={isp}")

            for ias in range(natomwfc):
                tag = f"ATMWFC{iotk_index(ias + 1)}"
                node = spin_node.find(tag)
                if node is None or node.text is None:
                    raise ValueError(f"Missing <{tag}> at ik={ik}, isp={isp}")

                data = re.split(r"[\s,]+", node.text.strip())
                if len(data) != 2 * nbnd:
                    raise ValueError(
                        f"Unexpected {tag} length at ik={ik}, isp={isp}: got {len(data)}"
                    )

                re_part = np.array([float(data[2 * ib]) for ib in range(nbnd)])
                im_part = np.array([float(data[2 * ib + 1]) for ib in range(nbnd)])
                proj[ias, :, ik, isp] = re_part + 1j * im_part

    return proj


def parse_overlaps(
    root: ET.Element, nkpts: int, nspin: int, natomwfc: int
) -> np.ndarray | None:
    overlap_section = root.find("OVERLAPS")
    if overlap_section is None:
        return None

    if _is_new_atomic_proj(root):
        # New: <OVPS dim="N" spin="1"> r i r i ... </OVPS> (one per k-point per spin)
        per_spin = {isp: [] for isp in range(nspin)}

        for ovps_el in list(overlap_section):
            if ovps_el.tag != "OVPS":
                continue
            if ovps_el.text is None:
                raise ValueError("<OVPS> is empty")

            dim = int(ovps_el.attrib.get("dim", str(natomwfc)))
            if dim != natomwfc:
                raise ValueError(f"OVPS dim={dim} does not match natomwfc={natomwfc}")

            isp = int(ovps_el.attrib.get("spin", "1")) - 1
            data = np.fromstring(ovps_el.text, sep=" ")

            expected = 2 * natomwfc * natomwfc
            if data.size != expected:
                raise ValueError(
                    f"Unexpected OVPS length: got {data.size}, expected {expected}"
                )

            mat = (data[0::2] + 1j * data[1::2]).reshape((natomwfc, natomwfc))
            per_spin[isp].append(mat)

        overlap = np.zeros((natomwfc, natomwfc, nkpts, nspin), dtype=np.complex128)
        for isp in range(nspin):
            mats = per_spin[isp]
            if len(mats) == 0:
                continue
            if len(mats) != nkpts:
                raise ValueError(
                    f"Parsed {len(mats)} overlap matrices for spin={isp + 1}, expected nkpts={nkpts}"
                )
            for ik in range(nkpts):
                overlap[:, :, ik, isp] = mats[ik]
        return overlap

    # Legacy: <OVERLAP001> ... </OVERLAP001> etc per k-point
    overlap = np.zeros((natomwfc, natomwfc, nkpts, nspin), dtype=np.complex128)

    for ik, kpoint in enumerate(overlap_section):
        for isp in range(nspin):
            tag = f"OVERLAP{iotk_index(isp + 1)}"
            node = kpoint.find(tag)
            if node is None or node.text is None:
                raise ValueError(f"Missing <{tag}> at ik={ik}, isp={isp}")

            data = re.split(r"[\s,]+", node.text.strip())
            if len(data) != 2 * natomwfc * natomwfc:
                raise ValueError(
                    f"Unexpected {tag} length at ik={ik}, isp={isp}: got {len(data)}"
                )

            mat = np.array(
                [
                    complex(float(data[i]), float(data[i + 1]))
                    for i in range(0, len(data), 2)
                ],
                dtype=np.complex128,
            )
            overlap[:, :, ik, isp] = mat.reshape(natomwfc, natomwfc)

    return overlap
