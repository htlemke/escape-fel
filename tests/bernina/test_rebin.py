"""escape.bernina.rebin against numpy, pyFAI, xrayutilities and itself
(numba vs array backend, dask vs serial, cached vs exact)."""
import numpy as np
import pytest

from escape.bernina import rebin

RNG = np.random.default_rng(1)
BACKENDS = ["numpy"] + (["numba"] if rebin.HAVE_NUMBA else [])
K = 4.5


def rot(axis, deg):
    """Right-handed rotation about 'x', 'y', 'z' ('+'/'-' for the sense)."""
    a = np.radians(deg) * (-1 if axis[1] == "-" else 1)
    c, s = np.cos(a), np.sin(a)
    i = "xyz".index(axis[0])
    j, k = (i + 1) % 3, (i + 2) % 3
    R = np.eye(3)
    R[j, j], R[j, k], R[k, j], R[k, k] = c, -s, s, c
    return R


def det_frame(delta, gamma=0.0):
    """Detector arm and face axes: delta about x-, then gamma about y+."""
    R = rot("y+", gamma) @ rot("x-", delta)
    return R[:, 2], R[:, 0], R[:, 1]


def numpy_q(uv, dist, arm, ex, ey, k=K):
    P = dist * arm + uv[:, :1] * ex + uv[:, 1:] * ey
    return k * (P / np.linalg.norm(P, axis=1, keepdims=True) - np.array([0, 0, 1.0]))


@pytest.fixture(scope="module")
def det():
    uv = rebin.detector_pixels(20.0, 15.0, 0.075)        # 266 x 200
    arm, ex, ey = det_frame(20.0, 3.0)
    return uv, (120.0, arm, ex, ey)


# ------------------------------------------------------------ Grid

def test_grid_constructors_and_index():
    g = rebin.Grid.from_voxel((0, -1, 2), (1, 1, 2.5), (0.1, 0.25, 0.05))
    assert g.shape == (10, 8, 10) and np.allclose(g.hi, (1, 1, 2.5))
    pts = RNG.uniform((-0.2, -1.2, 1.9), (1.2, 1.2, 2.6), (5000, 3))
    H, _ = np.histogramdd(pts, bins=g.edges)
    idx = g.index(pts)
    np.testing.assert_array_equal(np.bincount(idx[idx >= 0], minlength=g.size).reshape(g.shape), H)
    assert g.index([np.nan, 0, 2.1]) == -1
    # xrayutilities: centres at the given ends
    x = rebin.Grid.from_centres(0, 1, 11)
    np.testing.assert_allclose(x.centres[0], np.linspace(0, 1, 11))
    # covering, aligned to the voxel size
    c = rebin.Grid.covering(pts, 0.1)
    assert np.allclose(np.round(c.lo / 0.1), c.lo / 0.1) and np.all(c.index(pts) >= 0)
    assert rebin.Grid.from_bins(0, 1, 4).step[0] == 0.25


# ------------------------------------------------------------ fixed geometry

@pytest.mark.parametrize("backend", BACKENDS)
def test_fixed_index_nan_and_corrections(backend):
    npix, nbins = 5000, 37
    idx = RNG.integers(-1, nbins, npix)
    corr = RNG.uniform(0.5, 1.0, npix)
    stack = RNG.random((9, npix)).astype(np.float32)
    stack[3, ::7] = np.nan                                 # a per-frame mask
    fb = rebin.FixedBinner.from_index(idx, nbins, corrections=corr, backend=backend)
    sig, norm = fb.bin(stack)
    for f in range(len(stack)):
        ok = (idx >= 0) & np.isfinite(stack[f])
        np.testing.assert_allclose(sig[f], np.bincount(idx[ok], stack[f][ok], nbins), rtol=1e-5)
        np.testing.assert_allclose(norm[f], np.bincount(idx[ok], corr[ok], nbins), rtol=1e-5)
    s1, n1 = fb.bin(stack[3])                              # single frame path
    np.testing.assert_allclose(s1, sig[3], rtol=1e-5)
    np.testing.assert_allclose(n1, norm[3], rtol=1e-5)


@pytest.mark.parametrize("split,npt", [("no", 300), ("bbox", 300), ("full", 300), ("bbox", (120, 36))])
def test_fixed_from_pyfai_matches_pyfai(split, npt):
    pyFAI = pytest.importorskip("pyFAI")
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    shape = (300, 200)
    detector = pyFAI.detectors.Detector(75e-6, 75e-6, max_shape=shape)
    ai = AzimuthalIntegrator(dist=0.05, poni1=0.004, poni2=0.006, detector=detector, wavelength=1e-10)
    stack = RNG.random((10,) + shape).astype(np.float32)
    fb = rebin.FixedBinner.from_pyfai(ai, npt, shape, split=split, polarization=1.0)
    ours = fb.mean(stack)
    kw = dict(unit="2th_deg", method=(split, "csr", "cython"), correctSolidAngle=True, polarization_factor=1.0)
    for f in (0, 7):
        if np.ndim(npt):
            ref = ai.integrate2d_ng(stack[f], npt[0], npt[1], **kw).intensity.T     # pyFAI: (chi, 2theta)
        else:
            ref = ai.integrate1d_ng(stack[f], npt, **kw).intensity
        ok = np.isfinite(ours[f]) & (ref != 0)
        assert ok.mean() > 0.3
        np.testing.assert_allclose(ours[f][ok], ref[ok], rtol=1e-4)


def test_corrections_match_pyfai():
    pyFAI = pytest.importorskip("pyFAI")
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    shape = (200, 300)
    detector = pyFAI.detectors.Detector(75e-6, 75e-6, max_shape=shape)
    ai = AzimuthalIntegrator(dist=0.05, poni1=0.004, poni2=0.006, detector=detector, wavelength=1e-10)
    y = (np.arange(shape[0]) + 0.5) * 0.075 - 4.0
    x = (np.arange(shape[1]) + 0.5) * 0.075 - 6.0
    X, Y = np.meshgrid(x, y)
    uv = np.stack([X.ravel(), Y.ravel()], -1)
    for p in (1.0, 0.9, -0.5):
        c = rebin.pixel_corrections(uv, 50.0, [0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0], True, p)
        ref = ai.solidAngleArray(shape) * ai.polarization(shape, factor=p)
        np.testing.assert_allclose(c.reshape(shape), ref, rtol=1e-6)


def test_fixed_from_pose_matches_stream(det):
    uv, (dist, arm, ex, ey) = det
    grid = rebin.Grid.from_bins((-1, 0.5, -1), (1, 2.5, 1), (40, 50, 30))
    R = rot("y+", 12.0).T
    frames = RNG.random((10, len(uv))).astype(np.float32)
    fb = rebin.FixedBinner.from_pose(uv, K, grid, dist, arm, ex, ey, R)
    sb = rebin.StreamBinner(uv, K, grid)
    for f in frames:
        sb.add(f, dist, arm, ex, ey, R)
    sig, norm = fb.bin(frames)
    s2, n2 = sb.sums()
    np.testing.assert_allclose(sig.sum(0).reshape(grid.shape), s2, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(norm.sum(0).reshape(grid.shape), n2, rtol=1e-4, atol=1e-6)


# ------------------------------------------------------------ changing geometry

@pytest.mark.parametrize("backend", BACKENDS)
def test_stream_index_matches_numpy(det, backend):
    uv, (dist, arm, ex, ey) = det
    grid = rebin.Grid.from_bins((-1.0, 0.5, -1.0), (1.0, 2.5, 1.0), (60, 70, 80))
    R = rot("y+", 12.0).T
    sb = rebin.StreamBinner(uv, K, grid, backend=backend)
    sb._build_index((dist, arm, ex, ey) + grid.affine(R))
    ref = grid.index(numpy_q(uv, dist, arm, ex, ey) @ R.T)
    assert (ref >= 0).mean() > 0.5
    assert np.mean(rebin._to_numpy(sb.index[0]) != ref) < 1e-4       # bin-edge rounding only
    c = rebin.pixel_corrections(uv, dist, arm, ex, ey)
    np.testing.assert_allclose(rebin._to_numpy(sb.corr[0]), c, rtol=1e-5)


def test_q_matches_xrayutilities():
    xu = pytest.importorskip("xrayutilities")
    nv, nu, p, dist, en = 40, 60, 0.075, 150.0, 9000.0
    uv = rebin.detector_pixels(nu * p, nv * p, p)
    qc = xu.experiment.QConversion(["y+"], ["x-"], [0, 0, 1], en=en)
    qc.init_area("y+", "x+", cch1=nv / 2 - 0.5, cch2=nu / 2 - 0.5, Nch1=nv, Nch2=nu,
                 distance=dist, pwidth1=p, pwidth2=p)
    qx, qy, qz = qc.area(7.0, 25.0)
    q_xu = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], -1)
    k = 2 * np.pi / xu.en2lam(en)
    arm, ex, ey = det_frame(25.0)
    np.testing.assert_allclose(numpy_q(uv, dist, arm, ex, ey, k) @ rot("y+", 7.0), q_xu, atol=1e-6)


def test_stream_matches_xrayutilities_gridder():
    """Same q conversion and grid as xrayutilities' Gridder3D -> same mean."""
    xu = pytest.importorskip("xrayutilities")
    nv, nu, p, dist, en = 60, 80, 0.075, 150.0, 9000.0
    uv = rebin.detector_pixels(nu * p, nv * p, p)
    qc = xu.experiment.QConversion(["y+"], ["x-"], [0, 0, 1], en=en)
    qc.init_area("y+", "x+", cch1=nv / 2 - 0.5, cch2=nu / 2 - 0.5, Nch1=nv, Nch2=nu,
                 distance=dist, pwidth1=p, pwidth2=p)
    k = 2 * np.pi / xu.en2lam(en)
    etas = np.linspace(5, 6, 15)
    frames = RNG.random((len(etas), nv * nu)).astype(np.float32)
    n = (20, 25, 15)
    allq = np.concatenate([np.stack([a.ravel() for a in qc.area(e, 25.0)], -1) for e in etas])
    lo, hi = allq.min(0), allq.max(0)
    g = xu.Gridder3D(*n)
    g.KeepData(True)
    g.dataRange(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], fixed=True)
    for e, f in zip(etas, frames):
        qx, qy, qz = qc.area(e, 25.0)
        g(qx, qy, qz, f.reshape(nv, nu))
    grid = rebin.Grid.from_centres(lo, hi, n)
    np.testing.assert_allclose(grid.centres[0], g.xaxis)
    sb = rebin.StreamBinner(uv, k, grid, solid_angle=False, polarization=None)
    arm, ex, ey = det_frame(25.0)
    for e, f in zip(etas, frames):
        sb.add(f, dist, arm, ex, ey, rot("y+", e).T)
    ours, ref = sb.result(), g.data
    ok = np.isfinite(ours) & (g._gnorm > 0)
    assert ok.sum() > 100
    assert np.mean(np.abs(ours[ok] - ref[ok]) > 1e-5) < 0.01         # bin-edge points only


@pytest.mark.parametrize("backend", BACKENDS)
def test_tolerance_group_and_nan(det, backend):
    uv, (dist, arm, ex, ey) = det
    grid = rebin.Grid.from_bins((-1, 0.5, -1), (1, 2.5, 1), (40, 40, 40))
    frames = RNG.random((20, len(uv))).astype(np.float32)
    frames[:, :500] = np.nan                               # a fixed gap
    frames[5, 1000:3000] = np.nan                          # one shot's mask
    etas = 10 + np.arange(20) * 1e-5                       # drifts far below a voxel
    out = {}
    for name, kw in dict(exact={}, cached=dict(tolerance=0.05), grouped=dict(tolerance=0.05, group=True)).items():
        sb = rebin.StreamBinner(uv, K, grid, backend=backend, **kw)
        for f, e in zip(frames, etas):
            sb.add(f, dist, arm, ex, ey, rot("y+", e).T)
        out[name] = (sb.sums(), dict(sb.stats))
    assert out["exact"][1]["index_builds"] == 20
    assert out["cached"][1]["index_builds"] == 1 and out["grouped"][1]["bin_passes"] == 1
    (sc, nc), (sg, ng) = out["cached"][0], out["grouped"][0]
    np.testing.assert_allclose(sg, sc, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(ng, nc, rtol=1e-4, atol=1e-5)
    assert np.isfinite(sc).all() and (nc > 0).sum() > 100
    (se, ne) = out["exact"][0]
    assert abs(se.sum() - sc.sum()) / se.sum() < 1e-3


def test_backends_agree(det):
    if not rebin.HAVE_NUMBA:
        pytest.skip("needs numba")
    uv, (dist, arm, ex, ey) = det
    grid = rebin.Grid.from_bins((-1, 0.5, -1), (1, 2.5, 1), (30, 30, 30))
    frames = RNG.random((5, len(uv))).astype(np.float32)
    res = []
    for backend in ("numba", "numpy"):
        sb = rebin.StreamBinner(uv, K, grid, backend=backend, split=2, pixel=0.075)
        for i, f in enumerate(frames):
            sb.add(f, dist, arm, ex, ey, rot("y+", 10 + i).T)
        res.append(sb.sums())
    for a, b in zip(*res):
        assert np.mean(~np.isclose(a, b, rtol=1e-4, atol=1e-6)) < 1e-3


def test_split_conserves_and_smooths(det):
    uv, (dist, arm, ex, ey) = det
    grid = rebin.Grid.from_bins((-0.3, 1.2, -0.4), (0.3, 1.9, 0.0), (90, 90, 30))
    frame = np.ones(len(uv), np.float32)
    sums = {}
    for s in (1, 3):
        sb = rebin.StreamBinner(uv, K, grid, split=s, pixel=0.075, solid_angle=False, polarization=None)
        sb.add(frame, dist, arm, ex, ey, np.eye(3))
        sums[s] = sb.sums()[0]
    inside = (grid.index(numpy_q(uv, dist, arm, ex, ey)) >= 0).sum()
    assert sums[1].sum() == pytest.approx(inside)
    assert sums[3].sum() == pytest.approx(inside, rel=2e-3)       # subpixels near the edge may leave
    # finer voxels than pixels: splitting fills more voxels
    assert (sums[3] > 0).sum() > (sums[1] > 0).sum()


def test_scan_grid_covers_every_pixel(det):
    uv, (dist, arm, ex, ey) = det
    poses = [(dist, arm, ex, ey, rot("y+", e).T) for e in (10, 11, 12)]
    grid = rebin.scan_grid(uv, K, poses, voxel=0.01)
    for p in poses:
        assert np.all(grid.index(numpy_q(uv, *p[:4]) @ p[4].T) >= 0)


def test_hkl_from_angles():
    """At a solved reflection the detector centre sees integer hkl."""
    dif = pytest.importorskip("escape.bernina.diffraction")
    pytest.importorskip("hklforge")
    s = dif.Sample.from_text(dict(a=5.431, b=5.431, c=5.431, alpha=90, beta=90, gamma=90),
                             "(1,1,1) > (1,0,0)\n(1,1,-1) > (0,1,0)")
    energy = 9000.0
    angles = dif.solution_angles(s.solve((1, 1, 3), energy, {"mu": 0, "chi": 90, "phi": 0}), 1)
    arm, ex, ey, R = rebin.shot_geometry(angles, s, "hkl")
    grid = rebin.Grid.from_centres((0.5, 0.5, 2.5), (1.5, 1.5, 3.5), (11, 11, 11))
    sb = rebin.StreamBinner(np.zeros((1, 2)), dif.wavevector(energy), grid)
    sb.add_angles(np.ones(1, np.float32), angles, 200.0, s, "hkl")
    assert np.unravel_index(int(rebin._to_numpy(sb.index[0, 0])), grid.shape) == (5, 5, 5)
    assert np.allclose(R @ numpy_q(np.zeros((1, 2)), 200.0, arm, ex, ey, dif.wavevector(energy))[0], (1, 1, 3))


# ------------------------------------------------------------ dask and escape

def test_stream_reduce_dask_equals_serial(det):
    da = pytest.importorskip("dask.array")
    uv, (dist, arm, ex, ey) = det
    grid = rebin.Grid.from_bins((-1, 0.5, -1), (1, 2.5, 1), (30, 30, 30))
    frames = RNG.random((23, len(uv))).astype(np.float32)
    R = np.array([rot("y+", 10 + 0.1 * i).T for i in range(len(frames))])
    serial = rebin.StreamBinner(uv, K, grid).reduce(frames, dist, arm, ex, ey, R)
    lazy = rebin.StreamBinner(uv, K, grid).reduce(da.from_array(frames, chunks=(5, -1)), dist, arm, ex, ey, R,
                                                  scheduler="threads")
    for a, b in zip(serial.sums(), lazy.sums()):
        np.testing.assert_allclose(a, b, rtol=1e-10)


def test_fixed_dask_and_escape():
    da = pytest.importorskip("dask.array")
    escape = pytest.importorskip("escape")
    shape, nbins = (30, 40), 17
    idx = RNG.integers(-1, nbins, shape).ravel()
    stack = RNG.random((12,) + shape).astype(np.float32)
    fb = rebin.FixedBinner.from_index(idx, nbins, shape=shape)
    ref = fb.mean(stack)
    np.testing.assert_allclose(fb.mean_dask(da.from_array(stack, chunks=(5, 10, 40))).compute(), ref)
    arr = escape.Array(data=da.from_array(stack, chunks=(5, 30, 40)), index=np.arange(100, 112), step_lengths=[12])
    out = fb.apply(arr)
    assert out.shape == (12, nbins) and np.array_equal(out.index, arr.index)
    np.testing.assert_allclose(np.asarray(out.data.compute()), ref)


def test_stream_reduce_escape(det):
    da = pytest.importorskip("dask.array")
    escape = pytest.importorskip("escape")
    uv, (dist, arm, ex, ey) = det
    grid = rebin.Grid.from_bins((-1, 0.5, -1), (1, 2.5, 1), (20, 20, 20))
    frames = RNG.random((10, len(uv))).astype(np.float32)
    imgs = escape.Array(data=da.from_array(frames, chunks=(4, -1)), index=np.arange(10), step_lengths=[10])
    eta = escape.Array(data=np.linspace(10, 11, 8), index=np.arange(2, 10), step_lengths=[8])
    fake = {}

    def geom(angles, sample=None, space="hkl"):
        fake[angles["eta"]] = True
        return arm, ex, ey, rot("y+", angles["eta"]).T
    orig = rebin.shot_geometry
    rebin.shot_geometry = geom
    try:
        sb = rebin.StreamBinner(uv, K, grid).reduce_escape(imgs, {"eta": eta, "delta": 20.0}, dist)
    finally:
        rebin.shot_geometry = orig
    ref = rebin.StreamBinner(uv, K, grid)
    for i, e in zip(range(2, 10), np.linspace(10, 11, 8)):
        ref.add(frames[i], dist, arm, ex, ey, rot("y+", e).T)
    np.testing.assert_allclose(sb.sums()[0], ref.sums()[0], rtol=1e-10)
    assert sb.stats["shots"] == 8 and len(fake) == 8


# ------------------------------------------------------------ pyFAI PONI <-> detector arm

def _pyfai(orientation=3, rots=(0.05, -0.12, 0.3), shape=(120, 90)):
    pyFAI = pytest.importorskip("pyFAI")
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    det = pyFAI.detectors.Detector(75e-6, 100e-6, max_shape=shape, orientation=orientation)
    return AzimuthalIntegrator(dist=0.08, poni1=0.004, poni2=0.006, rot1=rots[0], rot2=rots[1], rot3=rots[2],
                               detector=det, wavelength=1.3e-10)


@pytest.mark.parametrize("orientation", [3, 1, 2, 4])
@pytest.mark.parametrize("rots", [(0, 0, 0), (0.05, -0.12, 0.3), (-0.4, 0.3, -1.2)])
def test_poni_positions_match_pyfai(orientation, rots):
    ai = _pyfai(orientation, rots)
    g = rebin.poni_geometry(ai)
    P = g["distance"] * g["arm"] + g["uv"][:, :1] * g["ex"] + g["uv"][:, 1:] * g["ey"]
    ref = ai.position_array(g["shape"]).reshape(-1, 3)[:, ::-1] * 1e3      # pyFAI: (z, y, x) in m
    np.testing.assert_allclose(P, ref, atol=1e-6)
    c = rebin.pixel_coordinates(g["uv"], g["distance"], g["arm"], g["ex"], g["ey"], g["k"])
    np.testing.assert_allclose(c["tth"], ai.center_array(g["shape"], unit="2th_deg").ravel(), atol=1e-6)
    dchi = (c["chi"] - ai.center_array(g["shape"], unit="chi_deg").ravel() + 180) % 360 - 180
    assert np.abs(dchi).max() < 1e-5
    np.testing.assert_allclose(c["q"] * 10, ai.center_array(g["shape"], unit="q_nm^-1").ravel(), rtol=1e-6)


def test_frame_poni_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(20):
        rots = rng.uniform(-1.2, 1.2, 3)
        d, arm, ex, ey, foot = rebin.poni_to_frame(0.1, 0.02, 0.03, *rots)
        back = rebin.frame_to_poni(d, arm, ex, ey, foot)
        np.testing.assert_allclose([back[k] for k in ("rot1", "rot2", "rot3")], rots, atol=1e-12)
        assert back["dist"] == pytest.approx(0.1) and back["poni2"] == pytest.approx(0.03)
    # the Bernina detector at a pose: face centred on the arm
    arm, ex, ey = det_frame(25.0, 8.0)
    p = rebin.frame_to_poni(300.0, arm, ex, ey, foot=(7.5, 10.0))
    np.testing.assert_allclose(rebin.poni_to_frame(**p)[1], arm, atol=1e-12)
    with pytest.raises(ValueError):
        rebin.frame_to_poni(300.0, arm, -ex, ey, foot=(0, 0))              # mirrored
    for o in (1, 2, 4):                                                    # mirrored frames
        d, arm, ex, ey, foot = rebin.poni_to_frame(0.1, 0.02, 0.03, 0.1, -0.2, 0.3, orientation=o)
        back = rebin.frame_to_poni(d, arm, ex, ey, foot, orientation=o)
        np.testing.assert_allclose([back[k] for k in ("rot1", "rot2", "rot3")], (0.1, -0.2, 0.3), atol=1e-12)


def test_read_poni_file(tmp_path):
    ai = _pyfai(4, (0.1, 0.2, -0.3))
    ai.save(str(tmp_path / "x.poni"))
    p = rebin.read_poni(tmp_path / "x.poni")
    assert p["orientation"] == 4 and p["shape"] == (120, 90)
    for k in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"):
        assert p[k] == pytest.approx(getattr(ai, k))
    g1, g2 = rebin.poni_geometry(tmp_path / "x.poni"), rebin.poni_geometry(ai)
    np.testing.assert_allclose(g1["uv"], g2["uv"])
    np.testing.assert_allclose(g1["arm"], g2["arm"])


def test_azimuthal_without_pyfai_matches_pyfai():
    """pixel_coordinates + FixedBinner.from_coords + corrections = pyFAI's
    histogram integration (no splitting) of the same PONI."""
    ai = _pyfai(3, (0.05, -0.12, 0.3), shape=(300, 200))
    g = rebin.poni_geometry(ai)
    c = rebin.pixel_coordinates(g["uv"], g["distance"], g["arm"], g["ex"], g["ey"])
    corr = rebin.pixel_corrections(g["uv"], g["distance"], g["arm"], g["ex"], g["ey"], polarization=0.95)
    frame = RNG.random(g["shape"]).astype(np.float32)
    r = ai.integrate1d_ng(frame, 200, unit="2th_deg", method=("no", "histogram", "cython"),
                          correctSolidAngle=True, polarization_factor=0.95)
    edges = np.concatenate([r.radial - np.diff(r.radial)[0] / 2, [r.radial[-1] + np.diff(r.radial)[0] / 2]])
    grid = rebin.Grid(edges[0], edges[1] - edges[0], len(r.radial))
    ours = rebin.FixedBinner.from_coords(c["tth"], grid, g["shape"], corr).mean(frame)
    ok = np.isfinite(ours) & (r.intensity != 0)
    assert ok.mean() > 0.9
    assert np.mean(~np.isclose(ours[ok], r.intensity[ok], rtol=1e-4)) < 0.03     # bin-edge pixels only


# ------------------------------------------------------------ sf-daq/dap radial profile

def test_matches_dap_radial_profile():
    """escape reproduces dap's radint_I exactly (set DAP_SRC to a sf-daq/dap checkout)."""
    import os
    if not os.environ.get("DAP_SRC"):
        pytest.skip("set DAP_SRC to a sf-daq/dap checkout")
    from pathlib import Path
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench_dap", Path(__file__).with_name("bench_dap_radprof.py"))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    rp = bench.load_dap_radprof(os.environ["DAP_SRC"])
    shape = (300, 200)
    mask = RNG.random(shape) > 0.05
    img = RNG.random(shape).astype(np.float32)
    x0, y0 = 80.3, 150.7
    res = dict(do_radial_integration=1, beam_center_x=x0, beam_center_y=y0)
    rp.calc_radial_integration(res, img, mask)
    r_min, n = res["radint_q"][0], len(res["radint_I"])
    rad = np.sqrt(np.add.outer((np.arange(shape[0]) - y0) ** 2, (np.arange(shape[1]) - x0) ** 2)).astype(int)
    fb = rebin.FixedBinner.from_index(np.where(mask, rad, -1).ravel(), int(rad.max()) + 1, shape=shape)
    np.testing.assert_allclose(fb.mean(img)[r_min:r_min + n], res["radint_I"], rtol=1e-12)
