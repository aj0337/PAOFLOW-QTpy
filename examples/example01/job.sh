#!/bin/bash

PW_EXEC="/home/anooja/Work/tools/qe-5.3/PW/src/pw.x"
PP_EXEC="/home/anooja/Work/tools/qe-5.3/PP/src/projwfc.x"

# "$PW_EXEC" <scf.in >scf.out
# "$PW_EXEC" <nscf.in >nscf.out
# "$PP_EXEC" <proj.in >proj.out
rm -rf output/
mpirun -n 4 python main_conductor.py conductor.yaml > conductor.out
