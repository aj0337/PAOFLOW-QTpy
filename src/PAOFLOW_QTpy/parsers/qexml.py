from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xml.etree.ElementTree as ET


def _local(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _find_first_by_local(root: ET.Element, local_name: str) -> ET.Element | None:
    for el in root.iter():
        if _local(el.tag) == local_name:
            return el
    return None


def _find_text_by_local(root: ET.Element, local_name: str) -> str | None:
    el = _find_first_by_local(root, local_name)
    if el is None or el.text is None:
        return None
    txt = el.text.strip()
    return txt if txt else None


def qexml_read_cell(file_path: str) -> dict[str, Any]:
    """
    Read lattice vectors and cell parameters from either:
      - QE legacy data-file.xml (QE 5.3 style)
      - QE schema data-file-schema.xml (qes-*.xsd style)

    Returns
    -------
    dict:
      - alat : float
      - avec : (3,3) ndarray (columns a1,a2,a3) in bohr
      - bvec : (3,3) ndarray (columns b1,b2,b3) in bohr^-1
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} not found")

    root = ET.parse(file_path).getroot()

    # ---------- Try schema-style first (ONLY if atomic_structure@alat exists) ----------
    atomic_structure = _find_first_by_local(root, "atomic_structure")
    if atomic_structure is not None and "alat" in atomic_structure.attrib:
        alat = float(atomic_structure.attrib["alat"])

        def vec3(name: str) -> np.ndarray:
            txt = _find_text_by_local(root, name)
            if not txt:
                raise ValueError(f"Schema XML: missing <{name}>")
            v = np.fromstring(txt, sep=" ")
            if v.size != 3:
                raise ValueError(f"Schema XML: <{name}> does not contain 3 numbers")
            return v

        a1, a2, a3 = vec3("a1"), vec3("a2"), vec3("a3")
        avec = np.column_stack((a1, a2, a3))

        b1, b2, b3 = vec3("b1"), vec3("b2"), vec3("b3")
        bvec = np.column_stack((b1, b2, b3)) * (2.0 * np.pi / alat)

        return {"alat": alat, "avec": avec, "bvec": bvec}

    # ---------- Legacy QE 5.3-style parsing ----------
    ns = {"q": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

    def find_text(tag: str) -> str | None:
        el = root.find(f".//q:{tag}" if ns else f".//{tag}", namespaces=ns)
        if el is None or el.text is None:
            return None
        txt = el.text.strip()
        return txt if txt else None

    def find_array(tag: str) -> np.ndarray | None:
        txt = find_text(tag)
        return np.fromstring(txt, sep=" ") if txt else None

    alat_txt = find_text("LATTICE_PARAMETER")
    if not alat_txt:
        raise ValueError("Legacy XML: missing LATTICE_PARAMETER")
    alat = float(alat_txt)

    a1, a2, a3 = find_array("a1"), find_array("a2"), find_array("a3")
    b1, b2, b3 = find_array("b1"), find_array("b2"), find_array("b3")
    if any(x is None for x in (a1, a2, a3, b1, b2, b3)):
        raise ValueError("Legacy XML: missing one of a1/a2/a3/b1/b2/b3")

    avec = np.column_stack((a1, a2, a3))
    bvec = np.column_stack((b1, b2, b3)) * (2.0 * np.pi / alat)

    return {"alat": alat, "avec": avec, "bvec": bvec}
