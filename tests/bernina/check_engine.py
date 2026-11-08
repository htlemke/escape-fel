"""Check the angle engine (hklforge, as escape.bernina uses it) against independent ones.

    python tests/bernina/check_engine.py diffcalc [N]          # needs diffcalc-core
    python tests/bernina/check_engine.py cases FILE [N]        # positions for the next step
    python tests/bernina/check_engine.py xrayutilities FILE    # needs xrayutilities (no hklforge)

diffcalc (Diamond's implementation of You's six-circle geometry) is compared
for the inverse problem: random cells of every crystal system, random
orientations, surface and reference vectors, energies and reachable
reflections, under many constraint combinations - UB, the complete set of
solutions, and the virtual angles of every solution.

xrayutilities has its own goniometer code (QConversion); it is compared for
the forward problem, angles -> hkl: random positions, and the solver's own
solutions, which must give back the requested reflection.
"""
from __future__ import annotations

import json
import sys

import numpy as np

CIRCLES = ("gamma", "delta", "mu", "eta", "chi", "phi")
VIRTUAL = ("theta", "qaz", "naz", "alpha", "beta", "tau", "psi", "betain", "betaout")

# constraint combinations (three each; booleans True); values drawn per case
MODES = [
    ("eta", "chi", "phi"), ("mu", "chi", "phi"), ("mu", "eta", "phi"), ("mu", "eta", "chi"),
    ("delta", "eta", "chi"), ("gamma", "mu", "chi"), ("delta", "mu", "phi"), ("gamma", "eta", "phi"),
    ("delta", "mu", "eta"), ("gamma", "chi", "phi"),
    ("qaz", "mu", "chi"), ("qaz", "eta", "phi"), ("qaz", "alpha", "phi"), ("qaz", "betain", "mu"),
    ("alpha", "mu", "chi"), ("beta", "mu", "phi"), ("psi", "mu", "eta"), ("naz", "alpha", "mu"),
    ("a_eq_b", "mu", "phi"), ("betain", "mu", "chi"), ("betaout", "eta", "phi"), ("bin_eq_bout", "mu", "eta"),
    ("delta", "alpha", "mu"), ("gamma", "betain", "eta"), ("delta", "psi", "chi"), ("gamma", "alpha", "phi"),
]
SYSTEMS = ("cubic", "tetragonal", "orthorhombic", "hexagonal", "rhombohedral", "monoclinic", "triclinic")


def wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def random_cell(rng):
    s = rng.choice(SYSTEMS)
    a, b, c = rng.uniform(3, 12, 3)
    al = be = ga = 90.0
    if s == "cubic":
        b = c = a
    elif s == "tetragonal":
        b = a
    elif s == "hexagonal":
        b, ga = a, 120.0
    elif s == "rhombohedral":
        b = c = a
        al = be = ga = rng.uniform(50, 110)
    elif s == "monoclinic":
        be = rng.uniform(92, 120)
    elif s == "triclinic":
        while True:
            al, be, ga = rng.uniform(70, 110, 3)
            ca, cb, cg = np.cos(np.radians([al, be, ga]))
            if 1 - ca ** 2 - cb ** 2 - cg ** 2 + 2 * ca * cb * cg > 0.1:
                break
    return s, dict(a=float(a), b=float(b), c=float(c), alpha=float(al), beta=float(be), gamma=float(ga))


def vec(v):
    return "(" + ",".join(f"{x:.6g}" for x in v) + ")"


def random_sample(rng):
    from escape.bernina import diffraction as dif
    system, cell = random_cell(rng)
    h1, h2 = rng.integers(-3, 4, 3), rng.integers(-3, 4, 3)
    while not h1.any() or np.linalg.norm(np.cross(h1, h2)) < 1e-9:
        h1, h2 = rng.integers(-3, 4, 3), rng.integers(-3, 4, 3)
    x1, x2 = rng.normal(size=3), rng.normal(size=3)
    orient = f"{vec(h1)} > {vec(x1)}\n{vec(h2)} > {vec(x2)}"
    surface, reference = rng.integers(-2, 3, 3), rng.integers(-2, 3, 3)
    surface = surface if surface.any() else np.array([0, 0, 1])
    reference = reference if reference.any() else np.array([1, 0, 0])
    sample = dif.Sample.from_text(cell, orient, vec(surface), vec(reference))
    return system, cell, orient, surface, reference, sample


def random_value(name, rng):
    if name in ("a_eq_b", "bin_eq_bout", "bisect"):
        return True
    if name in ("alpha", "beta", "betain", "betaout"):
        return float(rng.uniform(0, 30))
    if name in ("mu", "eta"):
        return float(rng.choice([0.0, rng.uniform(-30, 30)]))
    if name == "chi":
        return float(rng.choice([0.0, 90.0, rng.uniform(-180, 180)]))
    if name == "qaz":
        return float(rng.choice([90.0, 0.0, rng.uniform(-180, 180)]))
    return float(rng.uniform(-180, 180))


def reachable_hkl(sample, energy, rng):
    from escape.bernina import diffraction as dif
    kmax = 2 * dif.k_magnitude(energy) * 0.97
    for _ in range(200):
        h = rng.integers(-4, 5, 3)
        q = np.linalg.norm(sample.UB @ h)
        if h.any() and 0.2 * kmax < q < kmax:
            return tuple(int(x) for x in h)
    return None


def run_diffcalc(n=300, seed=1):
    from diffcalc.hkl.calc import HklCalculation
    from diffcalc.hkl.constraints import Constraints
    from diffcalc.ub.calc import UBCalculation
    from escape.bernina import diffraction as dif

    rng = np.random.default_rng(seed)
    stats = dict(cases=0, both_none=0, agree=0, solutions=0, continuum=0, unsupported=0, ub_max_dev=0.0,
                 angle_max_dev=0.0, virtual_compared=0, virtual_max_dev=0.0)
    mismatches = []
    for i in range(n):
        system, cell, orient, surface, reference, sample = random_sample(rng)
        ub = UBCalculation("you")
        ub.set_lattice("x", **cell)
        for hkl, xyz in dif.parse_orientations(orient.splitlines()):
            ub.add_orientation(hkl, tuple(dif.to_you(xyz)), None, None)
        ub.n_hkl, ub.surf_nhkl = tuple(float(x) for x in reference), tuple(float(x) for x in surface)
        ub.calc_ub()
        stats["ub_max_dev"] = max(stats["ub_max_dev"], float(np.abs(dif.to_you(sample.UB) - ub.UB).max()))
        energy = float(rng.uniform(5000, 20000))
        for mode in rng.permutation(len(MODES))[:4]:
            names = MODES[mode]
            cons = {nm: random_value(nm, rng) for nm in names}
            hkl = reachable_hkl(sample, energy, rng)
            if hkl is None:
                continue
            stats["cases"] += 1
            dc = {("nu" if k == "gamma" else k): v for k, v in cons.items()}
            try:
                cc = Constraints(dc)
                if not cc.is_current_mode_implemented():
                    stats["unsupported"] += 1
                    continue
                theirs = HklCalculation(ub, cc).get_position(*hkl, dif.en2lam(energy))
            except Exception as exc:
                theirs = exc
            try:
                df = sample.solve(hkl, energy, cons)
                ours = [{n: float(df.loc[n, c]) for n in df.index} for c in df.columns]
            except Exception as exc:
                ours = exc
            if isinstance(ours, Exception) and "continuum" in str(ours):
                stats["continuum"] += 1
                continue
            t_list = [] if isinstance(theirs, Exception) else theirs
            o_list = [] if isinstance(ours, Exception) else ours
            if not t_list and not o_list:
                stats["both_none"] += 1
                continue
            t_pos = sorted(tuple(np.round(wrap([getattr(p, "nu" if n == "gamma" else n) for n in CIRCLES]), 3))
                           for p, _ in t_list)
            o_pos = sorted(tuple(np.round(wrap([o[n] for n in CIRCLES]), 3)) for o in o_list)
            same = len(t_pos) == len(o_pos) and all(
                np.all(np.abs(wrap(np.subtract(a, b))) < 2e-3) for a, b in zip(t_pos, o_pos))
            if not same:
                mismatches.append(dict(system=system, cell=cell, orient=orient, surface=surface.tolist(),
                                       reference=reference.tolist(), energy=energy, hkl=hkl, cons=cons,
                                       ours=o_pos if not isinstance(ours, Exception) else str(ours),
                                       theirs=t_pos if not isinstance(theirs, Exception) else str(theirs)))
                continue
            stats["agree"] += 1
            stats["solutions"] += len(o_pos)
            for p, virt in t_list:
                pos = np.array([getattr(p, "nu" if n == "gamma" else n) for n in CIRCLES])
                match = min(o_list, key=lambda o: np.abs(wrap(pos - [o[n] for n in CIRCLES])).max())
                stats["angle_max_dev"] = max(stats["angle_max_dev"],
                                             float(np.abs(wrap(pos - [match[n] for n in CIRCLES])).max()))
                for name in VIRTUAL:
                    if name in virt and name in match and np.isfinite(match[name]) and np.isfinite(virt[name]):
                        stats["virtual_compared"] += 1
                        stats["virtual_max_dev"] = max(stats["virtual_max_dev"],
                                                       float(abs(wrap(virt[name] - match[name]))))
    print(json.dumps(stats, indent=1))
    print(f"{len(mismatches)} mismatches")
    for m in mismatches[:10]:
        print(json.dumps(m, default=str))
    return stats, mismatches


def write_cases(path, n=200, seed=2):
    from escape.bernina import diffraction as dif
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        _, cell, _, _, _, sample = random_sample(rng)
        energy = float(rng.uniform(5000, 20000))
        # a random position, with the hkl hklforge puts in Bragg condition there
        angles = {c: float(rng.uniform(-180, 180)) for c in CIRCLES}
        out.append(dict(kind="random", ub=sample.UB.tolist(), energy=energy, angles=angles,
                        hkl=list(map(float, sample.hkl_at(angles, energy)))))
        # the solver's solutions for a reflection: must give it back
        hkl = reachable_hkl(sample, energy, rng)
        names = MODES[int(rng.integers(len(MODES)))]
        try:
            df = sample.solve(hkl, energy, {nm: random_value(nm, rng) for nm in names})
        except Exception:
            continue
        for c in df.columns:
            out.append(dict(kind="solution", ub=sample.UB.tolist(), energy=energy,
                            angles={n: float(df.loc[n, c]) for n in CIRCLES}, hkl=list(map(float, hkl))))
    geometry = dif.GEOMETRY
    spec = dict(beam=list(map(float, geometry.beam)),
                sample=[(c.name, list(map(float, c.axis))) for c in geometry.sample],
                detector=[(c.name, list(map(float, c.axis))) for c in geometry.detector])
    with open(path, "w") as f:
        json.dump(dict(geometry=spec, cases=out), f)
    print(f"{len(out)} positions written to {path}")


def axis_string(v):
    i = int(np.argmax(np.abs(v)))
    if abs(abs(v[i]) - 1) > 1e-12 or np.count_nonzero(v) != 1:
        raise ValueError(f"xrayutilities needs principal axes, not {v}")
    return "xyz"[i] + ("+" if v[i] > 0 else "-")


def run_xrayutilities(path):
    import xrayutilities as xu
    with open(path) as f:
        data = json.load(f)
    g = data["geometry"]
    qc = xu.QConversion([axis_string(a) for _, a in g["sample"]], [axis_string(a) for _, a in g["detector"]],
                        g["beam"])
    order = [n for n, _ in g["sample"]] + [n for n, _ in g["detector"]]
    dev = {"random": 0.0, "solution": 0.0}
    count = {"random": 0, "solution": 0}
    for case in data["cases"]:
        angles = [case["angles"][n] for n in order]
        hkl = qc.point(*angles, UB=np.array(case["ub"]), en=case["energy"])
        d = float(np.abs(np.ravel(hkl) - case["hkl"]).max())
        dev[case["kind"]] = max(dev[case["kind"]], d)
        count[case["kind"]] += 1
    for kind in dev:
        print(f"{kind}: {count[kind]} positions, max |hkl(xrayutilities) - expected| = {dev[kind]:.2e}")
    return dev


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "diffcalc":
        run_diffcalc(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    elif what == "cases":
        write_cases(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200)
    elif what == "xrayutilities":
        run_xrayutilities(sys.argv[2])
    else:
        raise SystemExit(__doc__)
