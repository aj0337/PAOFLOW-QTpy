#!/bin/bash

PW_EXEC="/home/anooja/Work/tools/qe-5.3/PW/src/pw.x"
PP_EXEC="/home/anooja/Work/tools/qe-5.3/PP/src/projwfc.x"
CONDUCTOR_EXEC="/home/anooja/Work/software/PAOFLOW/externals/transportPAO/bin/conductor.x"

# "$PW_EXEC" <scf.in >scf.out
# "$PW_EXEC" <nscf.in >nscf.out
# "$PP_EXEC" <proj.in >proj.out
"$CONDUCTOR_EXEC" <conductor_bulk.in >conductor_bulk.out
# "$CONDUCTOR_EXEC" <conductor_lcr.in >conductor_lcr.out
# "$CONDUCTOR_EXEC" <conductor_lead_Al.in >conductor_lead_Al.out
# /home/anooja/Work/software/PAOFLOW/externals/transportPAO/extlibs/iotk/bin/iotk convert sgmlead_L_bulk.sgm sgmlead_L_bulk.xml
