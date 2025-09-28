#!/bin/bash

PW_EXEC="/home/anooja/Work/tools/qe-5.3/PW/src/pw.x"
PH_EXEC="/home/anooja/Work/tools/qe-5.3/bin/ph.x"
Q2T_EXEC="/home/anooja/Work/tools/qe-5.3/bin/q2trans.x"

# "$PW_EXEC" <cnhn.scf.in >scf.out
# "$PH_EXEC" <cnhn.ph.in >ph.out
# "$Q2T_EXEC" <cnhn.q2trans.in >q2t.out
rm -rf output/
python main_conductor.py conductor.yaml > conductor.out
