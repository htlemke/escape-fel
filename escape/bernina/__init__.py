"""Bernina (SwissFEL) diffractometer and detector calculations on hklforge.

What is specific to the Bernina instrument, so that eco (on the beamline)
and the bernina-scripts prep apps (Binder, browser) compute the same thing:

- :mod:`.geometry` - You's six-circle geometry in the SwissFEL frame
  (``GEOMETRY``), frames, units, directions as people write them;
- :mod:`.diffraction` - a ``Sample`` from cell and orientations, solution
  tables, two reflections, reflection/reachability tables, what the detector
  sees, the 3D page's message;
- :mod:`.recspace` - the same in the form eco's ``RecSpace`` keeps a crystal
  (reflections with positions, UB fits, diffcalc-style constraint dicts);
- :mod:`.detectors` - JUNGFRAU faces and module layouts (JF01T03, JF07T32);
- :mod:`.detector_view` - a detector face mapped to hkl, Brillouin zones,
  powder rings, rods, exit angles (and a matplotlib drawing of it);
- :mod:`.otti` - the THC lid's detector arm;
- ``VIEWER_HTML`` - the three.js 3D page of the diffractometer and
  reciprocal space (self-contained, fed through ``diffraction.view_payload``).

General crystallography (space groups, CIF, structure factors, reflection
groups) and the angle solver are hklforge's (``hklforge.crystal``,
``hklforge.Diffractometer``). Needs numpy, pandas and hklforge only - not
the rest of escape - so it also runs in a browser.
"""
from pathlib import Path

VIEWER_HTML = Path(__file__).with_name("diffractometer_3d.html")
