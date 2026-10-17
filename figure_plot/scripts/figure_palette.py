#!/usr/bin/env python3
"""The one palette every ECHO figure imports.

Colour identifies a configuration and nothing else. The MTP family is grey and
the ECHO family a red ramp that deepens with prediction depth; the ramp is wide
enough to survive print and greyscale, with adjacent luminance ratios of 1.96
(ECHO-1 against ECHO-1.5) and 3.28 (ECHO-1.5 against ECHO-3), against 1.52 and
2.41 in the version it replaces.

Neutrals: INK for axes and text, CONN for the grey connectors that join points of
one family, GRID for gridlines.

Anything that is not a configuration of ours -- the post-hoc drafters in the
serving figures -- keeps its own hue and is listed in OTHER, so a reader never
has to ask whether a colour means a configuration or a drafter.
"""

# configurations: grey MTP, a deepening red for ECHO
MTP_1 = "#9ca3af"
MTP_3 = "#6b7280"
ECHO_1 = "#f9c9cc"
ECHO_15 = "#e8757e"
ECHO_3 = "#8c1026"

CONF = {"MTP-1": MTP_1, "MTP-3": MTP_3, "ECHO-1": ECHO_1,
        "ECHO-1.5": ECHO_15, "ECHO-1.5$^{(3)}$": ECHO_15, "ECHO-3": ECHO_3}

# neutrals
INK = "#404040"
AXIS = "#4d5656"
CONN = "#b3ada7"
GRID = "#e5e5e5"
ZONE = "#7f8c8d"

# drafters that are not ours, in the serving figures
OTHER = {"EAGLE3-cold": "#3478b8", "EAGLE3-warm": "#c34c98",
         "DSpark": "#2ca02c", "DFlash": "#8c564b"}
