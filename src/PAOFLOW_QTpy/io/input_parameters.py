from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
)

import numpy as np
from pydantic import (
    BaseModel as PydanticBaseModel,
)
from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
    confloat,
    conint,
    field_validator,
)
from typing_extensions import Annotated
from yaml import SafeLoader, load

from PAOFLOW_QTpy.utils.constants import amconv, rydcm1

CalculationType = Literal[
    "conductor",
    "bulk",
]
ConductFormula = Literal[
    "landauer",
    "generalized",
]
Carriers = Literal[
    "electrons",
    "phonons",
]
SmearingType = Literal[
    "lorentzian",
    "gaussian",
    "fermi-dirac",
    "fd",
    "methfessel-paxton",
    "mp",
    "marzari-vanderbilt",
    "mv",
]
FileFormat = Literal[
    "internal",
    "crystal",
    "wannier90",
    "cp2k",
    "atmproj",
]


class FileNamesData(PydanticBaseModel):
    work_dir: str = "./"
    output_dir: str = "./"
    prefix: str = ""
    postfix: str = ""
    datafile_L: str = ""
    datafile_C: str = ""
    datafile_R: str = ""
    datafile_sgm: str = ""
    datafile_L_sgm: str = ""
    datafile_C_sgm: str = ""
    datafile_R_sgm: str = ""

    @field_validator("datafile_C")
    @classmethod
    def check_datafile_C(cls, value: str) -> str:
        if len(value) == 0:
            raise ValueError("datafile_C unspecified")
        return value


class HamiltonianData(PydanticBaseModel):
    H00_C: Optional[Dict[str, Any]] = None
    H_CR: Optional[Dict[str, Any]] = None
    H_LC: Optional[Dict[str, Any]] = None
    H00_L: Optional[Dict[str, Any]] = None
    H01_L: Optional[Dict[str, Any]] = None
    H00_R: Optional[Dict[str, Any]] = None
    H01_R: Optional[Dict[str, Any]] = None


class KPointGridSettings(PydanticBaseModel):
    nk: List[NonNegativeInt] = [0, 0]
    s: List[NonNegativeInt] = [0, 0]
    nkpts_par: NonNegativeInt = 1
    nrtot_par: NonNegativeInt = 1

    @field_validator("nk")
    @classmethod
    def check_nk(cls, value: List[int]) -> List[int]:
        if any(v < 0 for v in value):
            raise ValueError("Invalid nk: all values must be non-negative")
        return value

    @field_validator("s")
    @classmethod
    def check_s(cls, value: List[int]) -> List[int]:
        if any(v < 0 or v > 1 for v in value):
            raise ValueError("Invalid s: all values must be 0 or 1")
        return value


class EnergySettings(PydanticBaseModel):
    emin: float = -10.0
    emax: float = 10.0
    ne: Annotated[PositiveInt, conint(gt=1)] = 1000
    ne_buffer: Annotated[PositiveInt, conint(gt=0)] = 1
    delta: Annotated[NonNegativeFloat, confloat(ge=0.0, le=0.3)] = 1e-5
    smearing_type: SmearingType = "lorentzian"
    delta_ratio: Annotated[NonNegativeFloat, confloat(ge=0.0, le=0.1)] = 5.0e-3
    xmax: Annotated[NonNegativeFloat, confloat(ge=10)] = 25.0
    energy_step: NonNegativeFloat = 0.001
    nx_smear: NonNegativeInt = 20000

    @field_validator("emax")
    @classmethod
    def check_emax(cls, value: float, info) -> float:
        emin = info.data.get("emin", None)
        if emin is not None and value <= emin:
            raise ValueError("emax has to be greater than emin")
        return value

    @field_validator("ne")
    @classmethod
    def check_ne(cls, value: int) -> int:
        if value <= 1:
            raise ValueError("ne has to be greater than 1")
        return value

    @field_validator("ne_buffer")
    @classmethod
    def check_ne_buffer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("ne_buffer has to be greater than 0")
        return value

    @field_validator("xmax")
    @classmethod
    def check_xmax(cls, value: float) -> float:
        if value < 10.0:
            raise ValueError("xmax is too small")
        return value

    @field_validator("delta_ratio")
    @classmethod
    def check_delta_ratio(cls, value: float) -> float:
        if value < 0:
            raise ValueError("delta_ratio is negative")
        if value > 0.1:
            raise ValueError("delta_ratio is too large")
        return value


class SymmetryOutputOptions(PydanticBaseModel):
    use_sym: bool = True
    write_kdata: bool = False
    write_lead_sgm: bool = False
    write_gf: bool = False
    do_eigenchannels: bool = False
    neigchnx: NonNegativeInt = 200000
    do_eigplot: bool = False
    ie_eigplot: NonNegativeInt = 0
    ik_eigplot: NonNegativeInt = 0

    @field_validator("neigchnx")
    @classmethod
    def check_neigchnx(cls, value: int) -> int:
        if value < 0:
            raise ValueError("invalid neigchnx")
        return value

    @field_validator("ie_eigplot")
    @classmethod
    def check_ie_eigplot(cls, value: int) -> int:
        if value < 0:
            raise ValueError("invalid ie_eigplot")
        return value

    @field_validator("ik_eigplot")
    @classmethod
    def check_ik_eigplot(cls, value: int) -> int:
        if value < 0:
            raise ValueError("invalid ik_eigplot")
        return value


class IterationConvergenceSettings(PydanticBaseModel):
    nprint: PositiveInt = 20
    niterx: PositiveInt = 200
    nfailx: PositiveInt = 5
    transfer_thr: NonNegativeFloat = 1e-7

    @field_validator("nprint")
    @classmethod
    def check_nprint(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("nprint has to be greater than 0")
        return value

    @field_validator("niterx")
    @classmethod
    def check_niterx(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("niterx has to be greater than 0")
        return value

    @field_validator("transfer_thr")
    @classmethod
    def check_transfer_thr(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("invalid value for transfer_thr")
        return value


class AtomicProjectionOverlapSettings(PydanticBaseModel):
    do_orthoovp: bool = False
    atmproj_sh: NonNegativeFloat = 5.0
    atmproj_thr: Annotated[NonNegativeFloat, confloat(ge=0.0, le=1.0)] = 0.9
    atmproj_nbnd: NonNegativeInt = 0
    atmproj_do_norm: bool = False
    write_intermediate: bool = True

    @field_validator("atmproj_thr")
    @classmethod
    def check_atmproj_thr(cls, value: float) -> float:
        if value > 1.0 or value < 0.0:
            raise ValueError("invalid atmproj_thr")
        return value

    @field_validator("atmproj_nbnd")
    @classmethod
    def check_atmproj_nbnd(cls, value: int) -> int:
        if value < 0.0:
            raise ValueError("invalid atmproj_nbnd")
        return value


class AdvancedSettings(PydanticBaseModel):
    debug_level: int = 0
    ispin: int = 0
    surface: bool = False
    efermi_bulk: NonNegativeFloat = 0.0
    lhave_corr: bool = False
    ldynam_corr: bool = False
    leads_are_identical: bool = True
    shifting_scheme: NonNegativeInt = 1

    @field_validator("ispin")
    @classmethod
    def check_ispin(cls, value: int) -> int:
        if value < 0 or value > 2:
            raise ValueError("Invalid ispin")
        return value


@dataclass
class RuntimeData:
    nproc: int
    prefix: str
    work_dir: str
    nk_par: list[int]
    s_par: list[int]
    nk_par3d: np.ndarray
    s_par3d: np.ndarray
    nr_par3d: np.ndarray
    vkpt_par3D: np.ndarray
    wk_par: np.ndarray
    ivr_par3D: np.ndarray
    wr_par: np.ndarray
    nkpts_par: int
    nrtot_par: int


class ConductorData(PydanticBaseModel):
    file_names: FileNamesData
    hamiltonian: HamiltonianData
    kpoint_grid: KPointGridSettings
    energy: EnergySettings
    symmetry: SymmetryOutputOptions
    iteration: IterationConvergenceSettings
    atomic_proj: AtomicProjectionOverlapSettings
    advanced: AdvancedSettings
    dimL: NonNegativeInt = 0
    dimR: NonNegativeInt = 0
    dimC: NonNegativeInt = 0
    transport_direction: Annotated[int, conint(ge=1, le=3)] = 0
    calculation_type: CalculationType = "conductor"
    conduct_formula: ConductFormula = "landauer"
    carriers: Carriers = "electrons"

    bias: NonNegativeFloat = 0.0
    shift_L: NonNegativeFloat = 0.0
    shift_C: NonNegativeFloat = 0.0
    shift_R: NonNegativeFloat = 0.0
    shift_corr: NonNegativeFloat = 0.0

    _runtime: RuntimeData = PrivateAttr(default=None)

    def set_runtime_data(self, runtime: RuntimeData) -> None:
        self._runtime = runtime

    def get_runtime_data(self) -> RuntimeData:
        return self._runtime

    def __init__(self, filename: str, *, validate: bool = True, **data: Any) -> None:
        def filter_keys(cls, d):
            return {k: d[k] for k in cls.__fields__ if k in d}

        data["file_names"] = FileNamesData(**filter_keys(FileNamesData, data))
        data["hamiltonian"] = HamiltonianData(**filter_keys(HamiltonianData, data))
        data["kpoint_grid"] = KPointGridSettings(
            **filter_keys(KPointGridSettings, data)
        )
        data["energy"] = EnergySettings(**filter_keys(EnergySettings, data))
        data["symmetry"] = SymmetryOutputOptions(
            **filter_keys(SymmetryOutputOptions, data)
        )
        data["iteration"] = IterationConvergenceSettings(
            **filter_keys(IterationConvergenceSettings, data)
        )
        data["atomic_proj"] = AtomicProjectionOverlapSettings(
            **filter_keys(AtomicProjectionOverlapSettings, data)
        )
        data["advanced"] = AdvancedSettings(**filter_keys(AdvancedSettings, data))
        for key in [
            "dimL",
            "dimR",
            "dimC",
            "transport_direction",
            "calculation_type",
            "conduct_formula",
            "carriers",
            "ne",
            "ne_buffer",
            "bias",
            "shift_L",
            "shift_C",
            "shift_R",
            "shift_corr",
        ]:
            if key in data:
                data[key] = data[key]
        super().__init__(**data)
        if validate:
            self.validate_input()

    def validate_input(self) -> None:
        if self.file_names.datafile_C is None:
            raise ValueError(f"Unable to find {self.file_names.datafile_C}")

        if self.symmetry.ie_eigplot > 0.0 and not self.symmetry.do_eigplot:
            raise ValueError("ie_eigplot needs do_eigplot")

        if self.symmetry.ik_eigplot > 0.0 and not self.symmetry.do_eigplot:
            raise ValueError("ik_eigplot needs do_eigplot")

        if self.energy.emax <= self.energy.emin:
            raise ValueError("emax has to be greater than emin")

        if self.calculation_type == "conductor":
            if self.dimL <= 0:
                raise ValueError("dimL needs to be positive")
            if self.dimR <= 0:
                raise ValueError("dimR needs to be positive")
            if len(self.file_names.datafile_L) == 0:
                raise ValueError("datafile_L unspecified")
            if len(self.file_names.datafile_R) == 0:
                raise ValueError("datafile_R unspecified")
            if not self.file_names.datafile_L:
                raise ValueError(f"Unable to find {self.file_names.datafile_L}")
            if not self.file_names.datafile_R:
                raise ValueError(f"Unable to find {self.file_names.datafile_R}")

        if self.calculation_type == "bulk":
            user_provided_fields = set(self.model_fields_set)
            if "dimL" in user_provided_fields or "dimR" in user_provided_fields:
                raise ValueError("dimL and dimR should not be set in bulk mode")
            self.dimL = self.dimC
            self.dimR = self.dimC

            if len(self.file_names.datafile_L.strip()) != 0:
                raise ValueError("datafile_L should not be specified in bulk mode")
            if len(self.file_names.datafile_R.strip()) != 0:
                raise ValueError("datafile_R should not be specified in bulk mode")

            self.dimL = self.dimC
            self.dimR = self.dimC

        if (
            self.conduct_formula != "landauer"
            and len(self.file_names.datafile_sgm) == 0
            and len(self.file_names.datafile_C_sgm) == 0
        ):
            raise ValueError("Invalid conduct formula")

        if self.symmetry.do_eigplot and not self.symmetry.do_eigenchannels:
            raise ValueError("do_eigplot needs do_eigenchannels")

        if self.symmetry.write_lead_sgm and self.symmetry.use_sym:
            raise ValueError("use_sym and write_lead_sgm not implemented")

        if self.symmetry.write_gf and self.symmetry.use_sym:
            raise ValueError("use_sym and write_gf not implemented")

        if self.carriers == "phonons":
            self.energy.emin = self.energy.emin**2 / (rydcm1 / np.sqrt(amconv)) ** 2
            if self.energy.emin < 0.0:
                raise ValueError("emin < 0.0, invalid emin")
            self.energy.emax = self.energy.emax**2 / (rydcm1 / np.sqrt(amconv)) ** 2

    @field_validator("transport_direction")
    @classmethod
    def check_transport_direction(cls, value: int) -> int:
        if value < 1 or value > 3:
            raise ValueError(
                "Invalid value for transport_direction. Allowed values are 1,2 or 3"
            )
        return value

    @field_validator("dimC")
    @classmethod
    def check_dimC(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("dimC needs to be positive")
        return value


class CurrentData(PydanticBaseModel):
    filein: str
    fileout: str
    Vmin: float
    Vmax: float
    nV: PositiveInt
    sigma: NonNegativeFloat
    mu_L: float
    mu_R: float

    def __init__(self, filename: str, *, validate: bool = True, **data: Any) -> None:
        input_dict = self.read(filename)
        data.update(input_dict.get("input", {}))
        super().__init__(**data)
        if validate:
            self.validate_input()

    def read(self, filename: str) -> Dict[str, Any]:
        with open(Path(filename).absolute()) as f:
            return load(f, SafeLoader)

    def validate_input(self) -> None:
        if self.Vmax <= self.Vmin:
            raise ValueError("Vmax must be greater than Vmin")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    @field_validator("nV")
    @classmethod
    def check_nV(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("nV must be positive")
        return value
