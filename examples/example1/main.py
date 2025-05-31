from PAOFLOW_QTpy.io.startup import startup
from PAOFLOW_QTpy.io.write_header import write_header
from PAOFLOW_QTpy.parsers.atmproj_tools import parse_atomic_proj
from PAOFLOW_QTpy.io.summary import print_summary


def main():
    file_proj = "./al5.save/atomic_proj.xml"
    work_dir = "./al5.save"
    prefix = "al5"
    postfix = "_bulk"
    atmproj_sh = 3.5
    atmproj_thr = 0.9
    atmproj_nbnd = 60
    atmproj_do_norm = False
    do_orthoovp = True

    startup("conductor.py")
    write_header("Conductor Initialization")
    parse_atomic_proj(
        file_proj=file_proj,
        work_dir=work_dir,
        prefix=prefix,
        postfix=postfix,
        atmproj_sh=atmproj_sh,
        atmproj_thr=atmproj_thr,
        atmproj_nbnd=atmproj_nbnd,
        atmproj_do_norm=atmproj_do_norm,
        do_orthoovp=do_orthoovp,
        write_intermediate=True,
    )
    print_summary()


if __name__ == "__main__":
    main()
