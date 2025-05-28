from PAOFLOW_QTpy.io.startup import startup
from PAOFLOW_QTpy.io.write_header import write_header
from PAOFLOW_QTpy.parsers.atmproj_tools import parse_atomic_proj


class Conductor:
    def __init__(
        self,
        name: str = "conductor.py",
        file_proj: str = "./al5.save/atomic_proj.xml",
        work_dir: str = ".",
        prefix: str = "al5",
        postfix: str = "_bulk",
        atmproj_sh: float = 3.5,
        atmproj_thr: float = 0.9,
        atmproj_nbnd: int = 60,
        do_orthoovp: bool = False,
    ):
        """
        Initialize the conductor simulation environment.

        Parameters
        ----------
        `name` : str
            The name of the main driver, used for startup logging.
        `file_proj` : str
            Path to atomic_proj.xml file.
        `work_dir` : str
            Working directory for outputs.
        `prefix` : str
            Prefix for outputs.
        `postfix` : str
            Suffix for outputs.
        `atmproj_sh` : float
            Energy shift for projection filtering.
        `atmproj_thr` : float
            Threshold for projectability filtering.
        `atmproj_nbnd` : int
            Maximum number of bands to use.
        `do_orthoovp` : bool
            Whether to orthogonalize overlaps.
        """
        self.name = name
        self.file_proj = file_proj
        self.work_dir = work_dir
        self.prefix = prefix
        self.postfix = postfix
        self.atmproj_sh = atmproj_sh
        self.atmproj_thr = atmproj_thr
        self.atmproj_nbnd = atmproj_nbnd
        self.do_orthoovp = do_orthoovp

    def run(self):
        """
        Run the startup routine and initialize the transport setup.
        """
        startup(self.name)
        write_header("Conductor Initialization")
        parse_atomic_proj(
            self.file_proj,
            self.work_dir,
            self.prefix,
            self.postfix,
            self.atmproj_sh,
            self.atmproj_thr,
            self.atmproj_nbnd,
            self.do_orthoovp,
            write_intermediate=True,
        )
