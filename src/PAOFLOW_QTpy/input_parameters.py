# BUG
### Note : Broadcasting of input_dict can't be tested because mpirun only works
### correctly on Python 3.9. pydantic and typing_extensions only work on
### Python 3.10. Code runs in series but when running in parallel
# ModuleNotFoundError: No module named 'typing_extensions'. Output of running
# mpi4py on two different versions of python shown below.

##mpirun -np 2 python3.10 test.py
#World Size: 1   Rank: 0
#World Size: 1   Rank: 0
##mpirun -np 2 python3.9 test.py
#World Size: 2   Rank: 1
#World Size: 2   Rank: 0

from typing import (
    Any,
    List,
    Dict,
    Literal,
)

from typing_extensions import Annotated
from pathlib import Path
from yaml import load, SafeLoader
from mpi4py import MPI
import numpy as np

from PAOFLOW_QTpy.unitconverters import rydcm1, amconv

from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    confloat,
    conint,
    BaseModel as PydanticBaseModel,
    validator,
)

CalculationType = Literal['conductor', 'bulk', ]
ConductFormula = Literal['landauer', 'generalized', ]
Carriers = Literal['electrons', 'phonons', ]
SmearingType = Literal['lorentzian', 'gaussian', 'fermi-dirac', 'fd',
                       'methfessel-paxton', 'mp', 'marzari-vanderbilt', 'mv', ]
FileFormat = Literal['internal', 'crystal', 'wannier90', 'cp2k', 'atmproj', ]


class ConductorData(PydanticBaseModel):

    dimL: NonNegativeInt = 0
    dimR: NonNegativeInt = 0
    dimC: NonNegativeInt = 0
    transport_dir: Annotated[int, conint(ge=1, le=3)] = 0
    calculation_type: CalculationType = 'conductor'
    conduct_formula: ConductFormula = 'landauer'
    carriers: Carriers = 'electrons'
    ne: Annotated[PositiveInt, conint(gt=1)] = 1000
    ne_buffer: PositiveInt = 1
    emin: float = -10.0
    emax: float = 10.0
    delta: Annotated[NonNegativeFloat, confloat(ge=0.0, le=0.3)] = 1e-5
    smearing_type: SmearingType = 'lorentzian'
    delta_ratio: Annotated[NonNegativeFloat, confloat(ge=0.0, le=0.1)] = 5.0e-3
    xmax: Annotated[NonNegativeFloat, confloat(ge=10)] = 25.0
    bias: NonNegativeFloat = 0.0
    nk: List[NonNegativeInt] = 0
    s: List[NonNegativeInt] = 0
    use_sym = True
    nprint: PositiveInt = 20
    niterx: PositiveInt = 200
    nfailx: PositiveInt = 5
    transfer_thr: NonNegativeFloat = 1e-7
    write_kdata = False
    write_lead_sgm = False
    write_gf = False
    do_eigenchannels = False
    neigchnx: NonNegativeInt = 200000
    do_eigplot = False
    ie_eigplot: NonNegativeInt = 0
    ik_eigplot: NonNegativeInt = 0
    work_dir: str = './'
    prefix: str = ''
    postfix: str = ''
    datafile_L: str = ''
    datafile_C: str = ''
    datafile_R: str = ''
    datafile_sgm: str = ''
    datafile_L_sgm: str = ''
    datafile_C_sgm: str = ''
    datafile_R_sgm: str = ''
    do_orthovp = False
    atmproj_sh: NonNegativeFloat = 5.0
    atmproj_thr: Annotated[NonNegativeFloat, confloat(ge=0.0, le=1.0)] = 0.9
    atmproj_nbnd: NonNegativeInt = 0.0
    shift_L: NonNegativeFloat = 0.0
    shift_C: NonNegativeFloat = 0.0
    shift_R: NonNegativeFloat = 0.0
    shift_corr: NonNegativeFloat = 0.0
    debug_level: int = 0
    ispin: int = 0
    surface = False
    efermi_bulk: NonNegativeFloat = 0.0

    # Should probably be split into classes separate for conductor and
    # Hamiltonian. Keeping them combined for now.

    H00_C: Dict[str, Any] = None
    H_CR: Dict[str, Any] = None
    H_LC: Dict[str, Any] = None
    H00_L: Dict[str, Any] = None
    H01_L: Dict[str, Any] = None
    H00_R: Dict[str, Any] = None
    H01_R: Dict[str, Any] = None

    def __init__(self, filename, comm, **data: Any) -> None:

        input_dict = self.read(filename)
        data.update(input_dict)
        super().__init__(**data)
        self.validate_input()
        self.broadcast_dict(comm)

    def read(self, filename: str) -> Dict[str, Any]:
        try:
            with open(Path(filename).absolute()) as f:
                return load(f, SafeLoader)
        except Exception:
            raise

    def validate_input(self) -> None:

        if self.datafile_C is None:
            raise ValueError('Unable to find %s' % self.datafile_C)

        if self.ie_eigplot > 0.0 and not self.do_eigplot:
            raise ValueError('ie_eigplot needs do_eigplot')

        if self.ik_eigplot > 0.0 and not self.do_eigplot:
            raise ValueError('ik_eigplot needs do_eigplot')

        if self.emax <= self.emin:
            raise ValueError('emax has to be greater than emin')

        if self.calculation_type == 'conductor':
            if self.dimL <= 0:
                raise ValueError('dimL needs to be positive')
            if self.dimR <= 0:
                raise ValueError('dimR needs to be positive')
            if len(self.datafile_L) == 0:
                raise ValueError('datafile_L unspecified')
            if len(self.datafile_R) == 0:
                raise ValueError('datafile_R unspecified')
            if not self.datafile_L:
                raise ValueError('Unable to find %s' % self.datafile_L)
            if not self.datafile_R:
                raise ValueError('Unable to find %s' % self.datafile_R)

        else:
            if self.dimL != 0:
                raise ValueError('dimL should not be specified')
            if self.dimR != 0:
                raise ValueError('dimR should not be specified')
            if len(self.datafile_L) != 0:
                raise ValueError('datafile_L should not be specified')
            if len(self.datafile_R) != 0:
                raise ValueError('datafile_R should not be specified')

        if self.conduct_formula != 'landauer' and \
            len(self.datafile_sgm) == 0 and \
                len(self.datafile_C_sgm) == 0:
            raise ValueError('Invalid conduct formula')

        if self.do_eigplot and not self.do_eigenchannels:
            raise ValueError('do_eigplot needs do_eigenchannels')

        if self.write_lead_sgm or self.write_gf and self.use_sym:
            raise ValueError(
                'use_symm and write_sgm or write_gf not implemented')

        if self.carriers == 'phonons':
            self.emin = self.emin**2 / (rydcm1 / np.sqrt(amconv))**2
            if self.emin < 0.0:
                raise ValueError('emin < 0.0, invalid emin')
            self.emax = self.emax**2 / (rydcm1 / np.sqrt(amconv))**2

    @validator('transport_dir')
    def check_transport_dir(cls, value) -> None:
        if value < 1 or value > 3:
            raise ValueError(
                'Invalid value for transport_dir. Allowed values are 1,2 or 3')
        return value

    @validator('dimC')
    def check_dimC(cls, value) -> None:
        if value <= 0:
            raise ValueError('dimC needs to be positive')
        return value

    @validator('datafile_C')
    def check_datafile_C(cls, value) -> str:
        if len(value) == 0:
            raise ValueError('datafile_C unspecified')
        return value

    @validator('ne')
    def check_ne(cls, value) -> None:
        if value <= 1:
            raise ValueError('ne has to be greater than 1')
        return value

    @validator('ne_buffer')
    def check_ne_buffer(cls, value) -> None:
        if value <= 0:
            raise ValueError('ne_buffer has to be greater than 0')
        return value

    @validator('niterx')
    def check_niterx(cls, value) -> None:
        if value <= 0:
            raise ValueError('niterx has to be greater than 0')
        return value

    @validator('nprint')
    def check_nprint(cls, value) -> None:
        if value <= 0:
            raise ValueError('nprint has to be greater than 0')
        return value

    @validator('nk')
    def check_nk(cls, value) -> None:
        if any(value) < 0:
            raise ValueError('Invalid nk')
        return value

    @validator('s')
    def check_s(cls, value) -> None:
        if any(value) < 0 or any(value) > 1:
            raise ValueError('Invalid s')
        return value

    @validator('xmax')
    def check_xmax(cls, value) -> None:
        if value < 10.0:
            raise ValueError('xmax is too small')
        return value

    @validator('delta_ratio')
    def check_delta_ratio(cls, value) -> None:
        if value < 0:
            raise ValueError('delta_ratio is negative')
        if value > 0.1:
            raise ValueError('delta_ratio is too large')
        return value

    @validator('ispin')
    def check_ispin(cls, value) -> None:
        if value < 0 or value > 2:
            raise ValueError('Invalid ispin')
        return value

    @validator('neigchnx')
    def check_neigchnx(cls, value) -> None:
        if value < 0:
            raise ValueError('invalid neigchnx')
        return value

    @validator('ie_eigplot')
    def check_ie_eigplot(cls, value) -> None:
        if value < 0:
            raise ValueError('invalid ie_eigplot')
        return value

    @validator('ik_eigplot')
    def check_ik_eigplot(cls, value) -> None:
        if value < 0:
            raise ValueError('invalid ik_eigplot')
        return value

    @validator('transfer_thr')
    def check_transfer_thr(cls, value) -> None:
        if value <= 0:
            raise ValueError('invalid value for transfer_thr')
        return value

    @validator('atmproj_thr')
    def check_atmproj_thr(cls, value) -> None:
        if value > 1.0 or value < 0.0:
            raise ValueError('invalid atmproj_thr')
        return value

    @validator('atmproj_nbnd')
    def check_atmproj_nbnd(cls, value) -> None:
        if value < 0.0:
            raise ValueError('invalid atmproj_nbnd')
        return value

    def broadcast_dict(self, comm, root=0):
        input_dict = self.dict()
        input_dict = comm.bcast(input_dict, root=root)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    con = ConductorData('test.yaml', comm)
    rank = comm.rank
