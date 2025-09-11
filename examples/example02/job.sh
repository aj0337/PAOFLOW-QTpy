#!/bin/bash

PW_EXEC="/home/anooja/Work/tools/qe-5.3/PW/src/pw.x"
PP_EXEC="/home/anooja/Work/tools/qe-5.3/PP/src/projwfc.x"

# "$PW_EXEC" <scf.in >scf.out
# "$PW_EXEC" <nscf.in >nscf.out
# "$PP_EXEC" <proj.in >proj.out
# python main_conductor.py conductor_bulk.yaml > conductor_bulk.out
# python main_conductor.py conductor_lcr.yaml > conductor_lcr.out
# python main_conductor.py conductor_lead_Al.yaml > conductor_lead_Al.out
python main_current.py current.yaml > current.out
