import numpy as np


def parse_index_array(index_string: str, max_value: int, xval: int = -1) -> np.ndarray:
    """
    Parse a string representing a range or list of integer indices into a 0-based numpy array.

    The string format allows specifying intervals using either comma-separated values or ranges.
    Each interval may be of the form:

    - ``i1``          : a single index
    - ``i1-i2``       : an inclusive range from i1 to i2
    - ``Nx``          : a placeholder that yields N entries of `xval`

    Examples
    --------
    ``"1-3,5,2x"`` would yield: [0, 1, 2, 4, -1, -1]
    assuming `xval = -1` (default).

    Parameters
    ----------
    `index_string` : str
        A string representing a list of indices or ranges (1-based indexing).

    `max_value` : int
        If an entry evaluates to a negative value or is unspecified (e.g. via ``x``), then `xval` is substituted.

    `xval` : int, optional
        The value to assign for ``Nx`` constructs (default is -1).

    Returns
    -------
    `indices` : ndarray of int
        Parsed array of integer indices (0-based), ready for array slicing.

    Raises
    ------
    ValueError
        If the input format is invalid or inconsistent with expected size.
    """
    index_string = index_string.strip().lower()
    if index_string == "all":
        return np.arange(max_value)

    tokens = index_string.split(",")
    indices = []

    for token in tokens:
        token = token.strip()
        if "x" in token:
            repeat = int(token.replace("x", "")) if token.replace("x", "") else 1
            indices.extend([xval] * repeat)
        elif "-" in token:
            start_str, end_str = token.split("-")
            start, end = int(start_str), int(end_str)
            indices.extend(range(start - 1, end))
        else:
            indices.append(int(token) - 1)

    if len(indices) != max_value:
        raise ValueError(
            f"Expected {max_value} entries, got {len(indices)} from: {index_string}"
        )

    return np.array(indices, dtype=int)
