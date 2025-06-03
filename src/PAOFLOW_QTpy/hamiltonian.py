from typing import Literal
from PAOFLOW_QTpy.operator_blc import OperatorBlock


class HamiltonianSystem:
    """
    Represents a transport system defined by block Hamiltonians, overlap matrices, and correlation self-energies.

    This class holds and manages all matrix blocks used in quantum transport setups including the left/right leads,
    the central conductor, and inter-region couplings. Each block is stored as an OperatorBlock instance.

    Methods are provided for memory allocation, deallocation, and estimating memory usage.

    Attributes
    ----------
    dimL : int
        Dimension of the left lead region.
    dimC : int
        Dimension of the central conductor region.
    dimR : int
        Dimension of the right lead region.
    nkpts_par : int
        Number of parallel k-points.
    shift_L : float
        Energy shift for the left region.
    shift_C : float
        Energy shift for the center region.
    shift_R : float
        Energy shift for the right region.
    shift_corr : float
        Energy shift for correlation self-energy.
    """

    def __init__(self, dimL: int, dimC: int, dimR: int, nkpts_par: int) -> None:
        self.dimL = dimL
        self.dimC = dimC
        self.dimR = dimR
        self.nkpts_par = nkpts_par

        self.shift_L = 0.0
        self.shift_C = 0.0
        self.shift_R = 0.0
        self.shift_corr = 0.0

        self.allocated = False
        self.dimx = max(dimL, dimC, dimR)
        self.dimx_lead = max(dimL, dimR)

        self.blc_00L = OperatorBlock("block_00L")
        self.blc_01L = OperatorBlock("block_01L")
        self.blc_00R = OperatorBlock("block_00R")
        self.blc_01R = OperatorBlock("block_01R")
        self.blc_00C = OperatorBlock("block_00C")
        self.blc_LC = OperatorBlock("block_LC")
        self.blc_CR = OperatorBlock("block_CR")

    def allocate(self) -> None:
        """
        Allocate memory for all matrix blocks. Raises RuntimeError if already allocated or invalid dimensions.
        """
        if self.allocated:
            raise RuntimeError("Hamiltonian blocks already allocated.")
        if min(self.dimL, self.dimC, self.dimR, self.nkpts_par) <= 0:
            raise ValueError("Invalid dimensions for Hamiltonian allocation.")

        self.blc_00L.allocate(self.dimL, self.dimL, self.nkpts_par)
        self.blc_01L.allocate(self.dimL, self.dimL, self.nkpts_par)
        self.blc_00R.allocate(self.dimR, self.dimR, self.nkpts_par)
        self.blc_01R.allocate(self.dimR, self.dimR, self.nkpts_par)
        self.blc_00C.allocate(self.dimC, self.dimC, self.nkpts_par)
        self.blc_LC.allocate(self.dimL, self.dimC, self.nkpts_par)
        self.blc_CR.allocate(self.dimC, self.dimR, self.nkpts_par)

        self.allocated = True

    def deallocate(self) -> None:
        """
        Deallocate all matrix blocks if currently allocated.
        """
        if not self.allocated:
            return

        self.blc_00L.deallocate()
        self.blc_01L.deallocate()
        self.blc_00R.deallocate()
        self.blc_01R.deallocate()
        self.blc_00C.deallocate()
        self.blc_LC.deallocate()
        self.blc_CR.deallocate()

        self.allocated = False

    def memusage(self, memtype: Literal["ham", "corr", "all"] = "all") -> float:
        """
        Estimate total memory usage of all blocks in MB.

        Parameters
        ----------
        memtype : {"ham", "corr", "all"}
            Type of memory to report.

        Returns
        -------
        usage_mb : float
            Estimated memory in megabytes.
        """
        usage = 0.0
        if self.blc_00L.allocated:
            usage += self.blc_00L.memusage(memtype)
        if self.blc_01L.allocated:
            usage += self.blc_01L.memusage(memtype)
        if self.blc_00R.allocated:
            usage += self.blc_00R.memusage(memtype)
        if self.blc_01R.allocated:
            usage += self.blc_01R.memusage(memtype)
        if self.blc_00C.allocated:
            usage += self.blc_00C.memusage(memtype)
        if self.blc_LC.allocated:
            usage += self.blc_LC.memusage(memtype)
        if self.blc_CR.allocated:
            usage += self.blc_CR.memusage(memtype)

        return usage
