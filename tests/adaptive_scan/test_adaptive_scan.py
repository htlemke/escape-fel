"""Tests for the impulsive-response model, the generalised test-data generator,
the model fit and the adaptive delay-scan planner (escape.adaptive_scan).

Run just this file:
    python3 -m pytest tests/adaptive_scan/test_adaptive_scan.py
"""
import numpy as np
import pytest

from escape import adaptive_scan as ads
from escape.adaptive_scan import ImpulsiveResponse, Oscillation
from escape.storage import get_test_data


# --------------------------------------------------------------------- model
def test_default_response_is_classic_test_signal():
    t = np.linspace(-1, 5, 61)
    expected = np.where(t >= 0, 1 - np.cos(2 * np.pi / 0.7 * t) * np.exp(-t / 2), 0)
    np.testing.assert_allclose(ImpulsiveResponse.default()(t), expected, atol=1e-12)


@pytest.mark.parametrize("sigma", [0.05, 0.3])
def test_smearing_matches_numerical_convolution(sigma):
    resp = ImpulsiveResponse(
        [
            Oscillation(1.0, decay=3.0),
            Oscillation(-0.6, period=0.7, decay=2.0),
            Oscillation(0.3, period=0.25, decay=1.0, rise=0.5, onset=0.3, phase=0.4),
        ],
        t0=0.1,
        offset=0.2,
    )
    t = np.linspace(-1, 4, 21)
    x = np.linspace(-8 * sigma, 8 * sigma, 20001)
    w = np.exp(-(x**2) / (2 * sigma**2))
    w /= w.sum()
    numeric = np.array([np.sum(w * resp(tt - x)) for tt in t])
    np.testing.assert_allclose(resp(t, sigma=sigma), numeric, atol=2e-4)


def test_smearing_is_finite_far_from_time_zero():
    y = ImpulsiveResponse.default()(np.array([-1e3, -40.0, 40.0, 1e3]), sigma=0.2)
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y, [0, 0, 1, 1], atol=1e-6)


def test_grow_in_and_onset():
    osc = Oscillation(1.0, period=0.5, rise=0.4, onset=0.5)
    resp = ImpulsiveResponse([osc])
    assert np.all(resp(np.linspace(-1, 0.5, 30)) == 0)
    # envelope grows: max |y| over successive periods increases
    t = np.linspace(0.5, 3.0, 2001)
    y = np.abs(resp(t))
    blocks = [y[(t >= a) & (t < a + 0.5)].max() for a in (0.5, 1.0, 2.0)]
    assert blocks[0] < blocks[1] < blocks[2] <= 1.0


# ------------------------------------------------------------------ test data
def test_get_test_data_seeded_and_default_scan():
    a = get_test_data(600, as_da=False, seed=3)
    b = get_test_data(600, as_da=False, seed=3)
    c = get_test_data(600, as_da=False, seed=4)
    np.testing.assert_array_equal(np.asarray(a["i"].data), np.asarray(b["i"].data))
    assert not np.array_equal(np.asarray(a["i"].data), np.asarray(c["i"].data))
    assert set(a) == {"i0", "i", "t", "timetool", "i_pump", "pump_on", "drift"}
    np.testing.assert_allclose(a["i"].scan.par_steps["t"], [-1.0, -0.8, -0.6])


def test_get_test_data_custom_delays_and_response():
    delays = [2.0, -0.5, 2.0, 0.3]
    resp = ImpulsiveResponse(
        [Oscillation(1.0), Oscillation(0.5, period=0.3, decay=1.0, rise=0.2)]
    )
    d = get_test_data(delays=delays, step_length=50, response=resp, seed=0)
    assert len(d["i"]) == 200
    np.testing.assert_allclose(d["i"].scan.par_steps["t"], delays)
    # nominal delay + timetool jitter = true delay
    t_true = np.asarray(d["t"].data)
    tt = np.asarray(d["timetool"].data)
    np.testing.assert_allclose(t_true - tt, np.repeat(delays, 50))


def test_pump_probe_points_recovers_step():
    m = ads.PumpProbeSimulator(
        ImpulsiveResponse([Oscillation(1.0)]), shots_per_point=2000, seed=1
    )
    y, yerr = m([-0.8, 3.0])
    assert abs(y[0]) < 4 * yerr[0]
    assert abs(y[1] - 0.05) < 4 * yerr[1]


# ------------------------------------------------------------------------ fit
def test_fit_recovers_damped_oscillation():
    truth = ImpulsiveResponse.default().scaled(0.05)
    m = ads.GaussianSimulator(truth, noise=0.001, sigma=0.1, seed=2)
    t = np.linspace(-1, 5, 120)
    res = ads.fit_response(t, *m(t), max_osc=2)
    best = res[0]
    assert best.spec.n_osc == 1
    e = best.errors()
    assert abs(e["osc0_period"][0] - 0.7) < 4 * e["osc0_period"][1] + 0.01
    assert abs(e["osc0_decay"][0] - 2.0) < 4 * e["osc0_decay"][1] + 0.1
    assert abs(e["sigma"][0] - 0.1) < 0.05
    assert 0.5 < best.chi2_red < 2.0
    fitted = best.response(t, sigma=best.sigma)
    np.testing.assert_allclose(fitted, best.predict(t), atol=1e-9)


def test_fit_with_fixed_sigma_and_two_modes():
    truth = ImpulsiveResponse(
        [
            Oscillation(1.0),
            Oscillation(-0.5, period=0.7, decay=3.0),
            Oscillation(0.3, period=0.23, decay=1.5),
        ]
    )
    m = ads.GaussianSimulator(truth, noise=0.01, sigma=0.02, seed=5)
    t = np.linspace(-0.5, 4, 300)
    best = ads.fit_response(t, *m(t), max_osc=2, sigma=0.02)[0]
    assert best.spec.n_osc == 2
    periods = sorted(c.period for c in best.response.oscillations)
    np.testing.assert_allclose(periods, [0.23, 0.7], rtol=0.02)


# --------------------------------------------------------------------- design
def test_information_gain_rank_one():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    cov = A @ A.T + np.eye(3)
    J = rng.standard_normal((5, 3))
    g = ads.information_gain(J, cov, 0.5, np.arange(3))
    for j, gj in zip(J, g):
        post = np.linalg.inv(np.linalg.inv(cov) + np.outer(j, j) / 0.25)
        expected = 0.5 * np.log(np.linalg.det(cov) / np.linalg.det(post))
        assert gj == pytest.approx(expected, rel=1e-8)


def test_suggest_without_data_is_survey():
    scan = ads.AdaptiveScan((-1, 5), candidates=np.arange(-1, 5.01, 0.1))
    t = scan.suggest(7)
    np.testing.assert_allclose(t, np.linspace(-1, 5, 7), atol=1e-9)


def test_adaptive_loop_runs_and_stays_on_candidates():
    truth = ImpulsiveResponse.default().scaled(0.05)
    m = ads.GaussianSimulator(truth, noise=0.004, sigma=0.15, seed=0)
    cand = np.round(np.arange(-1, 5.0001, 0.02), 6)
    scan = ads.AdaptiveScan((-1, 5), candidates=cand, max_osc=1)
    ads.run_scan(scan, m, n_initial=20, n_rounds=4, points_per_round=5)
    assert scan.n_points == 40
    assert np.all(np.isin(np.round(scan.t, 6), cand))
    assert scan.best is not None
    w = scan.model_weights()
    assert w.shape == (len(scan.results),) and w.sum() == pytest.approx(1.0)
    # batches spread out instead of repeating one delay five times
    assert len(np.unique(scan.history[-1][0])) > 1


def test_estimates_combine_competing_solutions():
    truth = ImpulsiveResponse.default().scaled(0.05)
    m = ads.GaussianSimulator(truth, noise=0.0035, sigma=0.2, seed=4)
    scan = ads.AdaptiveScan((-1, 5))
    t = np.linspace(-1, 5, 300)
    scan.add(t, *m(t))
    est = scan.estimates()
    single = scan.best.errors()
    assert {"t0", "sigma", "osc0_period", "osc0_decay"} <= set(est)
    # mixture spread is never smaller than one hypothesis' own error
    assert est["t0"][1] >= single["t0"][1] * 0.999
    assert abs(est["osc0_period"][0] - 0.7) < 4 * est["osc0_period"][1]
    assert "competing solution" in scan.summary()


# ----------------------------------------------------------- costs / planning
def test_axis_move_times_and_backlash():
    ax = ads.Axis(speed=2.0, acceleration=4.0, overhead=0.3, backlash=0.1)
    fwd = ax.move_time(0.0, [1.0])[0]
    back = ax.move_time(1.0, [0.0])[0]
    assert fwd == pytest.approx(0.3 + 1.0 / 2.0 + 2.0 / 4.0)
    assert back > fwd  # overshoot + second approach
    assert ax.move_time(0.0, [0.0])[0] == 0.0
    small, large = ax.move_time(0.0, [0.01, 1.0])
    assert small / 0.01 > large / 1.0  # small moves: overhead dominated, but far cheaper


def test_axis_calibrate_recovers_parameters():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0, 5, 200), rng.uniform(0, 5, 200)
    true = ads.Axis(speed=1.5, overhead=0.4, reverse_penalty=0.8)
    dur = np.array([true.move_time(x, [z])[0] for x, z in zip(a, b)])
    fit = ads.Axis.calibrate(a, b, dur)
    assert fit.speed == pytest.approx(1.5, rel=1e-6)
    assert fit.overhead == pytest.approx(0.4, abs=1e-6)
    assert fit.reverse_penalty == pytest.approx(0.8, abs=1e-6)


def test_order_route_exact_and_heuristic():
    import itertools

    m = ads.Motion(ads.Axis(speed=2.0, overhead=0.3, backlash=0.1))
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 5, (7, 1))
    _o, t = ads.order_route(pts, [2.5], m)
    brute = min(
        ads.route_time(pts[list(p)], [2.5], m) for p in itertools.permutations(range(7))
    )
    assert t == pytest.approx(brute)
    pts = rng.uniform(0, 5, (25, 1))
    _o, t = ads.order_route(pts, [2.5], m)
    assert t <= ads.route_time(np.sort(pts, 0), [2.5], m) + 1e-9


def test_exposure_model_calibration():
    em = ads.ExposureModel("auto", base=1.0)
    e = np.r_[np.ones(20), 4 * np.ones(20)]
    err = np.sqrt(0.1**2 / e + 0.03**2)
    em.calibrate(err, e)
    assert em.floor == pytest.approx(0.03, rel=1e-6)
    assert em.coeff == pytest.approx(0.1, rel=1e-6)
    np.testing.assert_allclose(em.options, [0.5, 1, 2, 4, 8])


def test_grid_step_uses_roundto_grid():
    from escape.utilities import roundto

    scan = ads.ModelFreeScan((-1.03, 2.0), grid_step=0.05)
    np.testing.assert_allclose(scan.candidates, roundto(scan.candidates, 0.05))
    assert scan.candidates[0] == pytest.approx(-1.0)
    assert scan.candidates[-1] == pytest.approx(2.0)


def test_plan_prefers_cheap_moves_and_is_ordered():
    m = ads.Axis(speed=0.5, overhead=1.0, backlash=0.2)
    truth = ImpulsiveResponse([Oscillation(1.0)], t0=2.0)
    sim = ads.GaussianSimulator(truth, noise=0.05, sigma=0.1, seed=0)
    scan = ads.ModelFreeScan((0, 5), grid_step=0.05, motion=m)
    p = scan.initial_points(11)
    scan.add(p, *sim(p))
    plan = scan.plan(6, position=[2.0])
    assert len(plan) == 6 and plan.total_time > 0
    # visiting order is (near) monotone in the preferred direction
    assert ads.route_time(plan.positions[:, None], [2.0], scan.motion) == pytest.approx(
        plan.move_times.sum()
    )
    assert plan.move_times.sum() <= ads.route_time(
        np.sort(plan.positions)[:, None], [2.0], scan.motion
    ) + 1e-9


# ------------------------------------------------------------- model free
def test_model_free_refines_edge_and_beats_uniform():
    truth = ImpulsiveResponse([Oscillation(1.0)], t0=1.0)
    errs = {}
    for name in ("uniform", "modelfree"):
        e = []
        for seed in range(3):
            sim = ads.GaussianSimulator(truth, noise=0.05, sigma=0.05, seed=seed)
            kw = dict(grid_step=0.01)
            s = ads.UniformScan((-1, 5), **kw) if name == "uniform" else \
                ads.ModelFreeScan((-1, 5), **kw)
            instr = ads.Instrument(sim)
            ads.run_plan(s, instr, n_batches=15, batch=5)
            g = ads.ModelFreeScan((-1, 5))
            g.add(s.x, s.y, s.yerr)
            _i, x, y, _e = g.averaged()
            tt = np.linspace(-1, 5, 3000)
            e.append(np.sqrt(np.mean((np.interp(tt, x, y) - sim.truth(tt)) ** 2)))
        errs[name] = np.mean(e)
    assert errs["modelfree"] < errs["uniform"]


def test_model_free_adds_statistics_at_resolution_limit():
    truth = ImpulsiveResponse([Oscillation(1.0)], t0=1.0)
    sim = ads.GaussianSimulator(truth, noise=0.1, sigma=0.05, seed=0)
    s = ads.GradientScan((0, 2), grid_step=0.1, exposure=ads.ExposureModel([1, 4]))
    ads.run_plan(s, ads.Instrument(sim), n_batches=12, batch=4, n_initial=5)
    idx = s._nearest(s._X)
    assert np.bincount(idx).max() > 1  # grid filled -> re-measured with more statistics


# ------------------------------------------------------------- step model
def test_step_scan_finds_step_in_wide_range():
    truth = ImpulsiveResponse([Oscillation(1.0)], t0=7.3)
    sim = ads.GaussianSimulator(truth, noise=0.15, sigma=0.08, seed=1)
    m = ads.Axis(speed=1.0, overhead=0.5, backlash=0.05)
    s = ads.StepScan((0, 20), grid_step=0.01, motion=m)
    ads.run_plan(s, ads.Instrument(sim, motion=m), n_batches=8, batch=4)
    e = s.estimates()
    assert abs(e["t0"][0] - 7.3) < max(4 * e["t0"][1], 0.05)
    assert "exp0_rate" not in e  # pure step
    assert s.summary().startswith("step")


# ------------------------------------------------------ oscillation models
def test_three_oscillations_with_rise_times():
    truth = ImpulsiveResponse([
        Oscillation(1.0, decay=8.0),
        Oscillation(-0.5, period=0.7, decay=2.0),
        Oscillation(0.35, period=0.31, decay=1.2, rise=0.4),
        Oscillation(0.25, period=1.9, decay=4.0, rise=1.0),
    ])
    sim = ads.GaussianSimulator(truth, noise=0.02, sigma=0.03, seed=0)
    t = np.linspace(-1, 8, 500)
    best = ads.fit_response(t, *sim(t), max_osc=3, fit_rise=True, t_range=(-1, 8))[0]
    assert best.spec.n_osc == 3
    osc = sorted(best.response.oscillations, key=lambda c: c.period)
    np.testing.assert_allclose([c.period for c in osc], [0.31, 0.7, 1.9], rtol=0.03)
    np.testing.assert_allclose([c.decay for c in osc], [1.2, 2.0, 4.0], rtol=0.15)
    assert osc[2].rise == pytest.approx(1.0, abs=0.25)


# ------------------------------------------------------------------ 2-D
def test_peak2d_canonical_form_is_equivalent():
    from escape.adaptive_scan.peak2d import Peak2DSpec, _canonical

    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (300, 2))
    spec = Peak2DSpec(True)
    for _ in range(100):
        th = np.r_[rng.normal(size=2) * 0.3, rng.uniform(-10, 10), rng.uniform(0.2, 2, 4)]
        c = _canonical(spec, th)
        np.testing.assert_allclose(spec.design(X, th), spec.design(X, c), atol=1e-12)
        assert -np.pi / 2 <= c[2] < np.pi / 2
        assert c[3] + c[4] >= c[5] + c[6]


def test_peak_scan_2d_recovers_tilted_asymmetric_peak():
    truth = ads.Peak2D(0.7, -0.4, (0.35, 0.6), 0.2, np.deg2rad(30), 1.0, 0.05)
    m = ads.Motion([ads.Axis(speed=1.0, overhead=0.3), ads.Axis(speed=0.5, overhead=0.3)])
    sim = ads.GaussianSimulator(truth, noise=0.05, seed=3)
    s = ads.PeakScan2D(((-3, 3), (-3, 3)), grid_step=0.05, motion=m)
    ads.run_plan(s, ads.Instrument(sim, motion=m, ndim=2), n_batches=8, batch=6)
    e = s.estimates()
    assert abs(e["angle_deg"][0] - 30) < 6
    assert abs(e["fwhm_u"][0] - truth.fwhm[0]) < 0.2
    assert abs(e["fwhm_v"][0] - truth.fwhm[1]) < 0.1
    assert np.hypot(e["x0"][0] - 0.7, e["y0"][0] + 0.4) < 0.2
