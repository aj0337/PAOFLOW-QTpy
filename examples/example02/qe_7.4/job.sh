#!/bin/bash

PW_EXEC="/home/anooja/Work/software/qe-7.4.1/bin/pw.x"
PP_EXEC="/home/anooja/Work/software/qe-7.4.1/bin/projwfc.x"

# "$PW_EXEC" <scf.in >scf.out
# "$PW_EXEC" <nscf.in >nscf.out
# "$PP_EXEC" <proj.in >proj.out
# # rm -rf output/
# mpirun -n 4 python main_conductor.py conductor_bulk.yaml > conductor_bulk.out
mpirun -n 1 python main_conductor.py conductor_lcr.yaml > conductor_lcr.out
# mpirun -n 4 python main_conductor.py conductor_lead_Al.yaml > conductor_lead_Al.out
# mpirun -n 4 python main_current.py current.yaml > current.out
