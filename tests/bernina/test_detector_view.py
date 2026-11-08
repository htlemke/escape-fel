"""escape.bernina.detector_view: the face map against detector_map and brute force."""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from escape.bernina import detector_view as dv  # noqa: E402
from escape.bernina import diffraction as dif  # noqa: E402

CELL = dict(a=5.431, b=5.431, c=5.431, alpha=90, beta=90, gamma=90)
ORIENT = "(1,1,1) > (1,0,0)\n(1,1,-1) > (0,1,0)"
ENERGY = 9000.0


@pytest.fixture(scope="module")
def setup():
    s = dif.Sample.from_text(CELL, ORIENT, surface="(1,1,1)", reference="(1,0,0)")
    angles = dif.solution_angles(s.solve((1, 1, 3), ENERGY, {"mu": 0, "chi": 90, "betain": 2}), 1)
    return s, angles


@pytest.mark.parametrize("centring", ["P", "F"])
def test_grid_and_zones(setup, centring):
    s, angles = setup
    pose = dv.Pose(s, angles, ENERGY, 200.0, 153.6, 76.8, centring=centring)
    d = dv.face_grid(pose, 81)
    m = dif.detector_map(s, angles, ENERGY, 200.0, 153.6, 76.8, n=81)
    for c in "hkl":
        np.testing.assert_allclose(d[c], m[c], atol=1e-12)
    q = np.stack([d["h"], d["k"], d["l"]], -1) @ s.UB.T
    rng = np.random.default_rng(1)
    for _ in range(100):
        i, j = rng.integers(d["h"].shape[0]), rng.integers(d["h"].shape[1])
        base = np.round(np.linalg.solve(s.UB, q[i, j])).astype(int)
        box = np.stack(np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij"), -1).reshape(-1, 3) + base
        box = box[dv._CENTRING[centring](box[:, 0], box[:, 1], box[:, 2])]
        best = box[np.argmin(np.linalg.norm(q[i, j] - box @ s.UB.T, axis=1))]
        assert np.array_equal(best, d["node"][i, j])
    assert dv._CENTRING[centring](*d["node"].reshape(-1, 3).T).all()


def test_rods_lie_on_rods_and_rings_on_lattice(setup):
    s, angles = setup
    pose = dv.Pose(s, angles, ENERGY, 200.0, 153.6, 76.8)
    rods = dv.rod_crossings(pose)
    assert rods
    n_phi = pose.S.T @ pose.n_lab
    H = dv._lattice_points(pose, 2 * pose.k * 1.05) @ s.UB.T
    for u, v, _ in rods:
        e = dv.evaluate(pose, np.array([[u]]), np.array([[v]]))
        qq = np.array([e["h"][0, 0], e["k"][0, 0], e["l"][0, 0]]) @ s.UB.T
        dpar = (qq - H) - ((qq - H) @ n_phi)[:, None] * n_phi
        assert np.linalg.norm(dpar, axis=1).min() < 1e-9
    rings = dv.powder_rings(pose, 1.0, 4.0, allowed=lambda h: len({x % 2 for x in h}) == 1)
    for q, hkl in rings:
        assert q == pytest.approx(np.linalg.norm(s.UB @ np.asarray(hkl, float)))
        assert len({x % 2 for x in hkl}) == 1
    assert (1, 1, 1) in [h for _, h in rings]


def test_surface_quantities(setup):
    s, angles = setup
    pose = dv.Pose(s, angles, ENERGY, 200.0)
    d = dv.face_grid(pose, 41)
    n = pose.n_lab
    np.testing.assert_allclose(d["q_par"] ** 2 + d["q_perp"] ** 2, d["Q"] ** 2, rtol=1e-12)
    # betain 2 was the constraint: the incidence angle to the surface
    assert pose.alpha_in == pytest.approx(2.0, abs=1e-6)
    assert np.linalg.norm(n) == pytest.approx(1.0)


def test_draw_every_colour_and_overlay(setup):
    s, angles = setup
    pose = dv.Pose(s, angles, ENERGY, 200.0, 153.6, 76.8, centring="F", alpha_c=0.18)
    d = dv.face_grid(pose, 61)
    spots = dif.reflections_on_detector(s, angles, ENERGY, 200.0, 153.6, 76.8, 0.005)
    fig = Figure()
    for colour in dv.COLOURS:
        dv.draw(fig, pose, d, colour, list(dv.OVERLAYS), spots)
    assert "hkl" in dv.point_text(pose, 0.0, 0.0)


def test_without_surface_normal_notes_instead_of_failing():
    s = dif.Sample.from_text(CELL, ORIENT)
    pose = dv.Pose(s, {"delta": 30.0}, ENERGY, 300.0)
    d = dv.face_grid(pose, 41)
    notes = dv.draw(Figure(), pose, d, "alpha_f", ["horizon", "rods"])
    assert any("surface normal" in n for n in notes)


def test_jungfrau_layouts_and_gaps(setup):
    w, h, mods = dif.jungfrau_modules("JF07T32")
    assert len(mods) == 32 and (w, h) == pytest.approx((4215 * 0.075, 4432 * 0.075))
    w, h, mods = dif.jungfrau_modules("JF01T03")
    assert len(mods) == 3 and (w, h) == pytest.approx((1030 * 0.075, 1614 * 0.075))
    # the middle module is centred; the 36-pixel gap below it is off the modules
    assert dif.on_modules(mods, 0.0, 0.0)
    assert not dif.on_modules(mods, 0.0, -(257 + 18) * 0.075)
    for label, m in dif.DETECTOR_MODULES.items():
        rot = "turned" in label
        assert dif.DETECTOR_PRESETS[label] == pytest.approx(dif.jungfrau_modules(label.split()[0], rot)[:2])
    # turned by 90 degrees: the gaps run vertically, the face is wide
    wr, hr, rmods = dif.jungfrau_modules("JF01T03", rotated=True)
    assert (wr, hr) == pytest.approx((h, w))
    assert dif.on_modules(rmods, 0.0, 0.0) and not dif.on_modules(rmods, (257 + 18) * 0.075, 0.0)

    s, angles = setup
    pose = dv.Pose(s, angles, ENERGY, 300.0, w, h, modules=mods)
    spots = dif.reflections_on_detector(s, angles, ENERGY, 300.0, w, h, 0.02)
    notes = dv.draw(Figure(), pose, dv.face_grid(pose, 61), spots=spots)
    assert isinstance(notes, list)
