"""Regenerate the adaptive-scan gallery figures (docs/_static/adaptive_scan/*.png).

Usage (from the repository root)::

    python examples/adaptive_scan_gallery.py                 # all figures
    python examples/adaptive_scan_gallery.py models,step     # a subset

Figure keys: models, costs, modelfree, step, osc, osc3, 2d, bench.  All
simulations are seeded; "osc" and "osc3" take several minutes (one model fit
per batch).  The benchmark bar chart uses the medians reported in
docs/user_guide/adaptive_scan.md.
"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from escape import adaptive_scan as ads
from escape.adaptive_scan import ImpulsiveResponse, Oscillation
ALL = "models,costs,modelfree,step,osc,osc3,2d,bench"
WHICH = (sys.argv[1] if len(sys.argv) > 1 else ALL).split(",")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "_static", "adaptive_scan")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#a3a29c", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "lines.linewidth": 2, "legend.frameon": False})
stage = ads.Axis(speed=1.0, overhead=0.5, backlash=0.05, direction=+1)
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=110, bbox_inches="tight"); plt.close(fig); print("saved", name, flush=True)
def truth_line(ax, x, y, label="truth"):
    ax.plot(x, y, color=MUTED, lw=1.5, label=label, zorder=1)
def pts(ax, s, color=BLUE, label="measured", order=False):
    sz = 10 + 12 * np.log2(s.exposure / s.exposure_model.base + 1e-9).clip(-1, 3) + 12
    if order:
        sc = ax.scatter(s.x, s.y, c=np.arange(s.n_points), cmap="Blues", s=sz, edgecolors=INK2, linewidths=0.3, zorder=3, label=label)
        return sc
    ax.errorbar(s.x, s.y, s.yerr, fmt="none", ecolor=color, alpha=0.4, lw=0.8, zorder=2)
    ax.scatter(s.x, s.y, s=sz, color=color, edgecolors=SURF, linewidths=0.5, zorder=3, label=label)

# 1 ---------------------------------------------------------------- models
if "models" in WHICH:
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    t = np.linspace(-1, 6, 2000)
    r = ImpulsiveResponse.default()
    axs[0].plot(t, r(t), color=BLUE, label="ideal")
    axs[0].plot(t, r(t, sigma=0.1), color=ORANGE, label="0.1 timing jitter")
    axs[0].plot(t, r(t, sigma=0.2), color=AQUA, label="0.2 timing jitter")
    axs[0].set_title("Default: 1 − cos(2πt/0.7)·exp(−t/2)"); axs[0].legend(loc="lower right")
    three = ImpulsiveResponse([Oscillation(1.0, decay=8.0), Oscillation(-0.5, period=0.7, decay=2.0),
        Oscillation(0.35, period=0.31, decay=1.2, rise=0.4), Oscillation(0.25, period=1.9, decay=4.0, rise=1.0)])
    axs[1].plot(t, three(t), color=BLUE, label="sum (ideal)")
    comps = three.components
    axs[1].plot(t, comps[2].evaluate(t) + 1.6, color=ORANGE, lw=1.3, label="mode T=0.31, rise 0.4 (+1.6)")
    axs[1].plot(t, comps[3].evaluate(t) + 2.1, color=AQUA, lw=1.3, label="mode T=1.9, rise 1.0 (+2.1)")
    axs[1].set_title("3 overlapping modes, own decay and grow-in times"); axs[1].legend(loc="lower right", fontsize=8)
    for a in axs: a.set_xlabel("delay")
    axs[0].set_ylabel("signal"); save(fig, "1_signal_models")

# 2 ---------------------------------------------------------------- costs
if "costs" in WHICH:
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    x = np.linspace(0, 5, 1001)
    axs[0].plot(x, stage.move_time(2.5, x), color=BLUE, label="0.5 s overhead, 1 unit/s, backlash 0.05")
    acc = ads.Axis(speed=1.0, acceleration=2.0, overhead=0.5, backlash=0.05)
    axs[0].plot(x, acc.move_time(2.5, x), color=ORANGE, label="same, with acceleration 2 unit/s²")
    axs[0].axvline(2.5, color=MUTED, lw=1); axs[0].text(2.55, 0.1, "current position", color=INK2, fontsize=8)
    axs[0].set_xlabel("target position"); axs[0].set_ylabel("move time (s)")
    axs[0].set_title("Moving down costs an overshoot + second approach"); axs[0].legend(fontsize=8, loc="upper center")
    rng = np.random.default_rng(1); P = rng.uniform(0, 5, (12, 1)); m = ads.Motion(stage)
    order, topt = ads.order_route(P, [2.5], m)
    t_given = ads.route_time(P, [2.5], m)
    axs[1].plot(np.r_[2.5, P[:, 0]], np.arange(13), "-o", color=ORANGE, ms=5, lw=1.3, label=f"planned order: {t_given:.1f} s")
    axs[1].plot(np.r_[2.5, P[order, 0]], np.arange(13), "-o", color=BLUE, ms=5, lw=2, label=f"optimised route: {topt:.1f} s")
    axs[1].set_xlabel("position"); axs[1].set_ylabel("visit #"); axs[1].invert_yaxis()
    axs[1].set_title("Visiting order of one batch of 12 points"); axs[1].legend(fontsize=8, loc="lower left")
    save(fig, "2_motion_costs_and_route")

# 3 ---------------------------------------------------------------- model free
if "modelfree" in WHICH:
    truth = ImpulsiveResponse.default()
    fig, axs = plt.subplots(2, 3, figsize=(15, 5.5), sharex=True, height_ratios=(3, 1))
    for col, (name, make) in enumerate([("UniformScan", ads.UniformScan), ("ModelFreeScan", ads.ModelFreeScan), ("GradientScan", ads.GradientScan)]):
        sim = ads.GaussianSimulator(truth, noise=0.08, sigma=0.05, seed=4)
        instr = ads.Instrument(sim, motion=stage)
        s = make((-2, 15), grid_step=0.01, motion=stage, exposure=ads.ExposureModel("auto"))
        ads.run_plan(s, instr, budget=300, batch=5)
        tt = np.linspace(-2, 15, 3000); truth_line(axs[0, col], tt, sim.truth(tt))
        g = ads.ModelFreeScan((-2, 15)); g.add(s.x, s.y, s.yerr); _i, x, y, e = g.averaged()
        axs[0, col].plot(x, y, color=BLUE, lw=0.8, alpha=0.7, zorder=2)
        pts(axs[0, col], s)
        rms = np.sqrt(np.mean((np.interp(tt, x, y) - sim.truth(tt)) ** 2))
        axs[0, col].set_title(f"{name}: {s.n_points} pts, RMS {rms:.3f}\n{instr.move_time:.0f} s moving, {instr.n_reversals} reversals (300 s total)", fontsize=10)
        axs[1, col].hist(s.x, bins=85, range=(-2, 15), color=BLUE, rwidth=0.85)
        axs[1, col].set_xlabel("delay")
    axs[0, 0].set_ylabel("signal"); axs[1, 0].set_ylabel("points / bin"); axs[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Model-free strategies, unknown signal over a wide range (marker size = exposure)", y=1.02, fontweight="bold")
    save(fig, "3_model_free")

# 4 ---------------------------------------------------------------- step
if "step" in WHICH:
    truth = ImpulsiveResponse([Oscillation(1.0)], t0=12.34)
    fig, axs = plt.subplots(3, 2, figsize=(13, 9), width_ratios=(2.2, 1)); fig.subplots_adjust(hspace=0.45)
    for row, (name, make) in enumerate([("UniformScan", ads.UniformScan), ("GradientScan", ads.GradientScan), ("StepScan", ads.StepScan)]):
        sim = ads.GaussianSimulator(truth, noise=0.3, sigma=0.08, seed=2)
        instr = ads.Instrument(sim, motion=stage)
        s = make((0, 20), grid_step=0.01, motion=stage, exposure=ads.ExposureModel("auto"))
        ads.run_plan(s, instr, budget=150, batch=4)
        an = ads.StepScan((0, 20)); an.add(s.x, s.y, s.yerr, exposure=s.exposure); e = an.estimates()
        for ax, lim in ((axs[row, 0], (0, 20)), (axs[row, 1], (11.6, 13.1))):
            tt = np.linspace(*lim, 2000); truth_line(ax, tt, sim.truth(tt))
            sc = pts(ax, s, order=True, label="measured (darker = later)")
            ax.plot(tt, an.best.predict(tt), color=ORANGE, lw=1.5, label="step fit")
            ax.set_xlim(*lim)
        axs[row, 0].set_title(f"{name}: {s.n_points} points in 150 s — fitted edge {e['t0'][0]:.3f} ± {e['t0'][1]:.3f} (true 12.340)", fontsize=10, loc="left")
        axs[row, 1].set_title("zoom on the edge", fontsize=10, loc="left")
    axs[0, 0].legend(fontsize=8, loc="upper left"); axs[2, 0].set_xlabel("position"); axs[2, 1].set_xlabel("position")
    save(fig, "4_step_search")

# 5 ---------------------------------------------------------------- oscillation
if "osc" in WHICH:
    measure = ads.PumpProbeSimulator(shots_per_point=100, seed=3)
    scan = ads.OscillationScan((-2, 15), max_osc=2)
    ads.run_scan(scan, measure, n_initial=15, n_rounds=30, points_per_round=5)
    tt = scan.candidates; e = scan.estimates()
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True, height_ratios=(3, 1.2, 1))
    truth_line(axs[0], tt, measure.truth(tt))
    mu, sd = scan.best.predict(tt), scan.best.predict_std(tt)
    axs[0].fill_between(tt, mu - sd, mu + sd, color=ORANGE, alpha=0.25, lw=0)
    axs[0].plot(tt, mu, color=ORANGE, lw=1.5, label="fit ±1σ")
    pts(axs[0], scan)
    axs[0].set_title(f"OscillationScan on shot-by-shot pump-probe data (100 shots/point, {scan.n_points} points): "
                     f"period {e['osc0_period'][0]:.3f} ± {e['osc0_period'][1]:.3f} (true 0.7), decay {e['osc0_decay'][0]:.2f} ± {e['osc0_decay'][1]:.2f} (true 2.0)", fontsize=10, loc="left")
    axs[0].legend(loc="lower right", fontsize=8); axs[0].set_ylabel("ΔI/I")
    score, parts = scan.acquisition()
    axs[1].plot(tt, parts["gain"], color=BLUE, lw=1.5, label="information gain")
    axs[1].plot(tt, parts["discrimination"], color=AQUA, lw=1.5, label="model discrimination")
    axs[1].set_ylabel("value (nats)"); axs[1].legend(fontsize=8, loc="upper right")
    axs[2].hist(scan.x[15:], bins=85, range=(-2, 15), color=BLUE, rwidth=0.85); axs[2].set_ylabel("chosen pts"); axs[2].set_xlabel("delay")
    save(fig, "5a_oscillation_scan")
if "osc3" in WHICH:
    three = ImpulsiveResponse([Oscillation(1.0, decay=8.0), Oscillation(-0.5, period=0.7, decay=2.0),
        Oscillation(0.35, period=0.31, decay=1.2, rise=0.4), Oscillation(0.25, period=1.9, decay=4.0, rise=1.0)])
    sim = ads.GaussianSimulator(three, noise=0.03, sigma=0.03, seed=0)
    instr = ads.Instrument(sim, motion=stage)
    s = ads.OscillationScan((-1, 8), max_osc=3, fit_rise=True, grid_step=0.01, motion=stage)
    ads.run_plan(s, instr, budget=700, batch=8, n_initial=40)
    e = s.estimates(); tt = s.candidates
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=(3, 1))
    truth_line(axs[0], tt, sim.truth(tt)); axs[0].plot(tt, s.best.predict(tt), color=ORANGE, lw=1.5, label="fit")
    pts(axs[0], s)
    txt = "   ".join(f"T={e[f'osc{k}_period'][0]:.3f}, τ={e[f'osc{k}_decay'][0]:.2f}, rise={e[f'osc{k}_rise'][0]:.2f}" for k in range(3))
    axs[0].set_title(f"3 overlapping modes: {s.n_points} points in {instr.clock:.0f} s\nfitted  {txt}\ntrue     T=1.9, τ=4, rise=1.0   T=0.7, τ=2, rise=0   T=0.31, τ=1.2, rise=0.4", fontsize=9, loc="left")
    axs[0].legend(fontsize=8, loc="lower right"); axs[0].set_ylabel("signal")
    axs[1].hist(s.x, bins=90, range=(-1, 8), color=BLUE, rwidth=0.85); axs[1].set_ylabel("points"); axs[1].set_xlabel("delay")
    save(fig, "5b_three_modes")

# 6 ---------------------------------------------------------------- 2-D
if "2d" in WHICH:
    peak = ads.Peak2D(x0=0.7, y0=-0.4, sigma_u=(0.35, 0.6), sigma_v=0.2, angle=np.deg2rad(30), amplitude=1.0, offset=0.05)
    xy = ads.Motion([ads.Axis(speed=1.0, overhead=0.3, backlash=0.05), ads.Axis(speed=0.5, overhead=0.3, backlash=0.05)])
    B = ((-3, 3), (-3, 3))
    sim = ads.GaussianSimulator(peak, noise=0.08, seed=3); ir = ads.Instrument(sim, motion=xy, ndim=2)
    xs = np.arange(-3, 3.001, 0.5); P = []
    for i, x in enumerate(xs): P += [(x, y) for y in (xs if i % 2 == 0 else xs[::-1])]
    P = np.array(P); y, err = ir(P); rast = ads.PeakScan2D(B, grid_step=0.05); rast.add(P, y, err)
    sim2 = ads.GaussianSimulator(peak, noise=0.08, seed=3); ia = ads.Instrument(sim2, motion=xy, ndim=2)
    ad = ads.PeakScan2D(B, grid_step=0.05, motion=xy); ads.run_plan(ad, ia, budget=ir.clock, batch=6)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    xs2 = np.unique(ad._C[:, 0]); XX, YY = np.meshgrid(xs2, xs2, indexing="ij"); G = np.column_stack([XX.ravel(), YY.ravel()])
    for ax, s, inst, name in ((axs[0], rast, ir, "snake raster, step 0.5"), (axs[1], ad, ia, "PeakScan2D")):
        Z = s.best.predict(G).reshape(XX.shape).T
        ax.pcolormesh(xs2, xs2, Z, cmap="Blues", shading="auto", vmin=0, vmax=1)
        ax.contour(xs2, xs2, peak(G).reshape(XX.shape).T, levels=[0.25, 0.5, 0.75], colors=INK, linewidths=0.8)
        ax.scatter(s._X[:, 0], s._X[:, 1], s=9, color=ORANGE, edgecolors=SURF, linewidths=0.3)
        e = s.estimates()
        ax.set_title(f"{name}: {s.n_points} pts, {inst.clock:.0f} s\nFWHM {e['fwhm_u'][0]:.2f} ± {e['fwhm_u'][1]:.2f} / {e['fwhm_v'][0]:.2f} ± {e['fwhm_v'][1]:.2f} (true 1.12 / 0.47)\n"
                     f"tilt {e['angle_deg'][0]:.1f} ± {e['angle_deg'][1]:.1f}° (true 30°)", fontsize=9, loc="left")
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    score = ad.acquisition()[0][ad._nearest(G)].reshape(XX.shape).T
    im = axs[2].pcolormesh(xs2, xs2, score, cmap="Blues", shading="auto")
    axs[2].contour(xs2, xs2, peak(G).reshape(XX.shape).T, levels=[0.25, 0.5, 0.75], colors=INK, linewidths=0.8)
    plt.colorbar(im, ax=axs[2], label="information (nats)")
    axs[2].set_title("where the next point is worth most\n(flanks along both axes, tails for asymmetry)", fontsize=9, loc="left")
    axs[2].set_aspect("equal"); axs[2].set_xlabel("x")
    fig.suptitle("2-D peak, same beamtime (contours = truth; dots = measured)", fontweight="bold", y=1.04)
    save(fig, "6_peak2d")

# 7 ---------------------------------------------------------------- benchmark
if "bench" in WHICH:
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
    names = ["Uniform", "ModelFree", "Gradient", "StepScan"]; v = [0.22, 0.17, 0.063, 0.007]
    axs[0].bar(names, v, color=[MUTED, MUTED, MUTED, BLUE], width=0.6); axs[0].set_yscale("log"); axs[0].set_ylim(0.004, 0.6)
    for i, x in enumerate(v): axs[0].text(i, x * 1.15, f"{x:g}", ha="center", color=INK2, fontsize=9)
    axs[0].set_title("Step search: median error of edge\nposition (150 s, log scale)", fontsize=10, loc="left")
    names = ["Uniform", "ModelFree", "Oscillation\n(no reorder)", "Oscillation"]; v = [1, 1, 2, 5]
    axs[1].bar(names, v, color=[MUTED, MUTED, MUTED, BLUE], width=0.6); axs[1].set_ylim(0, 8)
    for i, x in enumerate(v): axs[1].text(i, x + 0.15, f"{x}/8", ha="center", color=INK2, fontsize=9)
    axs[1].set_title("Wide-range oscillation: runs that found\nthe right period (400 s)", fontsize=10, loc="left")
    cats = ["FWHM long", "FWHM short", "tilt / 10°", "centre"]; r = [0.067, 0.067, 0.52, 0.13]; a = [0.040, 0.016, 0.11, 0.047]
    xx = np.arange(4); axs[2].bar(xx - 0.18, r, 0.34, color=MUTED, label="raster (417 s)"); axs[2].bar(xx + 0.18, a, 0.34, color=BLUE, label="PeakScan2D (427 s)")
    axs[2].set_xticks(xx, cats); axs[2].legend(fontsize=8)
    axs[2].set_title("2-D peak: median errors at equal time", fontsize=10, loc="left")
    fig.suptitle("Benchmarks (equal simulated beamtime incl. moves and backlash; 8–12 random repeats)", fontweight="bold", y=1.05)
    save(fig, "7_benchmarks")
