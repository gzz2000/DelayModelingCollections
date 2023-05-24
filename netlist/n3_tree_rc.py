# for debugging tau2015 simple FF CK->Q.
#
# units: ps, ff, kohm

import os
from .simple_spef_parser import parse_file

rc_path = os.path.join(os.path.dirname(__file__), 'tau2015-simple-n3.spef')
rc = parse_file(rc_path)

# we use rise as example.
# respectively: u2:I, u4:A2
rc.sink_cell_caps = [0.665639, 0.680152]
# fall: 0.701566 0.70871
