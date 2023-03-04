# this module exports a sample parasitics that is an RC tree.
# (from `tau2015-netcard_iccad-newNet_42040.spef`)
#
# units: ps, ff, kohm

import os
from .simple_spef_parser import parse_file

rc_path = os.path.join(os.path.dirname(__file__), 'tau2015-netcard_iccad-newNet_42040.spef')
rc = parse_file(rc_path)

# from pin 1 to pin 4
rc.sink_cell_caps = [1.0, 1.0, 1.0, 1.0]
