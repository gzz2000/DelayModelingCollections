# this module exports a sample parasitics that is a two-pin simple net.
# units: ps, ff, kohm

import os
from .simple_spef_parser import parse_file

rc_path = os.path.join(os.path.dirname(__file__), 'twopin.spef')
rc = parse_file(rc_path)

rc.sink_cell_caps = [1.0]
