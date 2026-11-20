"""escape.bernina.recspace (eco's RecSpace): the photon energy as a constraint."""
import numpy as np
import pytest

from escape.bernina import diffraction as dif
from escape.bernina import recspace as rc

if not hasattr(pytest.importorskip("hklforge.solve"), "_energy_limits"):
    pytest.skip("hklforge < 0.2.2: no free energy", allow_module_level=True)

CELL = dict(a=5.431, b=5.431, c=5.431, alpha=90, beta=90, gamma=90)
UB = dif.Sample.from_text(CELL, "(1,1,1) > (1,0,0)\n(1,1,-1) > (0,1,0)").UB


@pytest.mark.parametrize("hkl, cons, extra", [((1, 1, 1), {"mu": 0, "chi": 30, "phi": 45}, "eta"),
                                              ((1, 1, 1), {"mu": 0, "chi": 30, "phi": 45}, "delta"),
                                              ((2, 2, 0), {"mu": 0, "eta": 20, "chi": 90}, "gamma")])
def test_free_energy_finds_the_fixed_energy_solution(hkl, cons, extra):
    fixed = rc.solve(UB, 9000.0, hkl, {**cons, "energy": None})      # None: the energy is fixed
    assert fixed and "energy" not in fixed[0]
    sols = rc.solve(UB, None, hkl, {**cons, extra: fixed[0][extra], "energy": (5000, 20000)})
    energies = np.array([s["energy"] for s in sols])
    assert np.all((energies >= 5000) & (energies <= 20000)) and np.min(abs(energies - 9000)) < 1e-6
    for s in sols:
        assert np.allclose(rc.calc_hkl(UB, s["energy"], {n: s[n] for n in rc.POSITION_ORDER}), hkl, atol=1e-8)
    assert rc.make_mode({"eta": 1, "energy": (None, 12000)})[0]["energy"] == (None, 12000.0)
