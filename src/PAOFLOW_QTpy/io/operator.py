def read_operator_aux(filename: str, read_grid: bool = False):
    """
    Placeholder for reading auxiliary operator metadata from an XML-based IOTK file.

    Parameters
    ----------
    filename : str
        Path to the operator file (e.g., `.sgm`, `.xml`).
    read_grid : bool
        Whether to read the energy grid from the file.

    Returns
    -------
    metadata : dict
        Dictionary containing parsed attributes like `dimwann`, `nrtot`, `dynamical`, etc.

    Notes
    -----
    This function requires knowledge of the IOTK XML format and structure,
    including attributes like:
        - `dimwann`, `nrtot`, `dynamical`, `nomega`
        - optional blocks: <GRID>, <IVR>, etc.

    TODO: Implement once example file structure is available.
    """
    raise NotImplementedError(
        "IOTK file structure not yet known. Provide example to proceed."
    )


def read_operator_data(filename: str, ie: int = None):
    """
    Placeholder for reading complex-valued operator matrices from an IOTK XML file.

    Parameters
    ----------
    filename : str
        Path to the operator file.
    ie : int, optional
        If set, reads the `ie`-th dynamical slice (e.g., <OPR3>). Otherwise assumes static.

    Returns
    -------
    opr : np.ndarray
        3D or 2D complex-valued operator block.

    TODO
    ----
    Implementation pending access to example `.sgm` or `.xml` file contents.
    """
    raise NotImplementedError("Cannot parse operator data without IOTK file example.")


def write_operator_aux():
    raise NotImplementedError("Not implemented. Awaiting file structure definition.")


def write_operator_data():
    raise NotImplementedError("Not implemented. Awaiting file structure definition.")
