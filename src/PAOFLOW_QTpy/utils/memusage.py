from typing import Callable, Dict
from io import StringIO
import psutil


class MemoryTracker:
    """
    Tracks memory usage for different components of the transport calculation.
    """

    def __init__(self):
        self.sections: Dict[str, Dict] = {}

    def register_section(
        self, name: str, usage_func: Callable[[], float], is_allocated: bool
    ):
        """
        Register a memory-tracked section.

        Parameters
        ----------
        `name` : str
            Section name, e.g., "smearing".
        `usage_func` : Callable[[], float]
            Function returning memory usage in MB.
        `is_allocated` : bool
            Whether the section is currently allocated.
        """
        self.sections[name] = {"usage_func": usage_func, "is_allocated": is_allocated}

    def report(self, include_real_memory: bool = False) -> str:
        """
        Generate the memory usage report.

        Parameters
        ----------
        `include_real_memory` : bool
            Whether to include system-level memory usage.

        Returns
        -------
        `report` : str
            Formatted memory usage summary.
        """
        output = StringIO()
        print("  <MEMORY_USAGE>", file=output)

        memsum = 0.0
        for section, data in self.sections.items():
            if data["is_allocated"]:
                usage = data["usage_func"]()
                memsum += usage
                print(f"{section:>24}: {usage:15.3f} MB", file=output)

        print("", file=output)  # Equivalent to WRITE(iunit, "()")
        print(f"{'Total allocated. Memory':>24}: {memsum:15.3f} MB", file=output)

        if include_real_memory:
            process = psutil.Process()
            tmem = process.memory_info().rss / 1024.0 / 1024.0  # Convert bytes to MB
            print(f"{'Real allocated. Memory':>24}: {tmem:15.3f} MB", file=output)

        print("  </MEMORY_USAGE>\n", file=output)

        print(output.getvalue())


def hamiltonian_memusage(mode: str) -> float:
    """
    Return memory used by the hamiltonian module in MB.
    Placeholder for actual logic.

    Parameters
    ----------
    `mode` : str
        Either 'ham' for Hamiltonian data or 'corr' for correlation data.
    """
    if mode == "ham":
        return 0.0
    elif mode == "corr":
        return 0.0
    else:
        raise ValueError("Invalid mode for hamiltonian_memusage")


def workspace_memusage() -> float:
    """
    Return memory used by the workspace module in MB.
    Placeholder for actual logic.
    """
    return 0.0
