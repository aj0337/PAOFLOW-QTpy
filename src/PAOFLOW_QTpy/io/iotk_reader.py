import re
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np


class IOTKReader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.lines: List[str] = []
        self.pointer: int = 0

        if not self.file_path.exists():
            raise FileNotFoundError(f"IOTK file {file_path} not found")

        with open(self.file_path) as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def rewind(self):
        self.pointer = 0

    def find_section(self, section: str) -> None:
        pattern = f"@{section.upper()}"
        for idx, line in enumerate(self.lines):
            if line.upper().startswith(pattern):
                self.pointer = idx + 1
                return
        raise ValueError(f"Section @{section} not found in file")

    def read_attr_block(self) -> Dict[str, str]:
        attrs = {}
        while self.pointer < len(self.lines):
            line = self.lines[self.pointer]
            if line.startswith("@"):  # new section starts
                break
            if "=" in line:
                key, val = line.split("=", 1)
                attrs[key.strip().lower()] = val.strip()
            self.pointer += 1
        return attrs

    def read_matrix(self, shape: Tuple[int, int]) -> np.ndarray:
        nrows, ncols = shape
        data = []
        while len(data) < nrows * ncols:
            if self.pointer >= len(self.lines):
                raise ValueError("Unexpected end of file while reading matrix")
            line = self.lines[self.pointer]
            tokens = re.split(r"[\s,]+", line.strip())
            floats = list(map(float, tokens))
            data.extend(floats)
            self.pointer += 1
        return np.array(data).reshape((nrows, ncols))

    def read_complex_matrix(self, shape: Tuple[int, int]) -> np.ndarray:
        nrows, ncols = shape
        data = []
        while len(data) < 2 * nrows * ncols:
            if self.pointer >= len(self.lines):
                raise ValueError("Unexpected end of file while reading complex matrix")
            line = self.lines[self.pointer]
            tokens = re.split(r"[\s,]+", line.strip())
            floats = list(map(float, tokens))
            data.extend(floats)
            self.pointer += 1
        complex_data = np.array(data).reshape(-1, 2)
        return complex_data[:, 0] + 1j * complex_data[:, 1]

    def read_vector_array(self, shape: Tuple[int, int]) -> np.ndarray:
        nrows, ncols = shape
        data = []
        while len(data) < nrows * ncols:
            if self.pointer >= len(self.lines):
                raise ValueError("Unexpected end of file while reading vector array")
            line = self.lines[self.pointer]
            tokens = re.split(r"[\s,]+", line.strip())
            ints = list(map(int, tokens))
            data.extend(ints)
            self.pointer += 1
        return np.array(data).reshape((nrows, ncols))

    def read_block(
        self, name: str, shape: Tuple[int, int], complex_values: bool = False
    ) -> np.ndarray:
        self.rewind()
        found = False
        for idx, line in enumerate(self.lines):
            if line.upper().startswith(name.upper()):
                self.pointer = idx + 1
                found = True
                break
        if not found:
            raise ValueError(f"Block {name} not found in IOTK file")

        if complex_values:
            return self.read_complex_matrix(shape)
        else:
            return self.read_matrix(shape)

    def find_spin_section(self, ispin: int):
        tag = f"@SPIN{ispin}"
        for idx, line in enumerate(self.lines):
            if line.upper().startswith(tag):
                self.pointer = idx + 1
                return
        raise ValueError(f"Spin section {tag} not found")
