#!/usr/bin/env python3
"""Read and summarize the output of the MolDStruct strong-field module (simulation_output/ of mdrun -ionize,
gromacs-sf), with or without the trapped-electron cloud.  No GROMACS needed.

    sfmd_output.py summary <simulation_output>
        the run (from the log), final charges per element, events (ADK, electron impact, three-body
        recombination) per element and in 5 fs bins, the electron cloud, the autostop
    sfmd_output.py charges <simulation_output> [--every 10]
        mean charge per element over time, from charges_over_time.bin
    sfmd_output.py transitions <simulation_output> [--type adk,impact,recomb] [--atom 12] [--tmin 20 --tmax 40]
        one line per event:  t[fs]  atom  element  type  q_before -> q_after
    sfmd_output.py plot <simulation_output> [-o overview.png] [--gif [--every N] [--no-h]] [--extract]
        one overview figure of what the run wrote (panels for missing files are left out); default
        <run dir>/overview.png.  --gif: explosion.gif, three views along the principal axes of the initial
        heavy-atom gyration tensor, coloured by charge (reads the run's .trr).  --extract: <run>/extracted/ with
        transitions.txt and charges_over_time.txt

<simulation_output> may be omitted (default: the current directory) or be the run directory holding it.

Times.  Every file is put on a time axis in fs from what the run wrote, never as steps x dt: under
sfmd-schedule the MD step changes during the run (gromacs-mc's output script had that bug).
  - files with a time column (pulse_profile, mean_charge_vs_time, the event logs, pb_log, laser_log, pbg_box,
    energy_ledger) are used as they are;
  - screening_log.txt counts steps: its times are interpolated in energy_ledger.txt, which has both columns;
  - charges_over_time.bin: frame k is at tinit + k * sfmd-charge-output-stride * dt with the tpr's dt and stride
    (from the run log; the schedule rescales the stride to keep that spacing).

In Python:
    import sfmd_output as so
    ev = so.read_events("run/simulation_output")       # structured array: t[fs], atom (0-based), elem, kind, q0, q1
    el = so.elements("run/simulation_output")          # element per atom (from masses.bin)
    t, q = so.read_charges("run/simulation_output")    # frame times [fs], (frames, atoms) float32
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

STD_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305,
            "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ca": 40.078, "Fe": 55.845, "Ni": 58.693,
            "I": 126.904}
KINDS = ["adk", "impact", "recomb"]
# reference categorical palette, fixed order (dataviz palette.md, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"


# ------------------------------------------------------------------ reading

def _dir(p):
    """simulation_output, given it or the run directory holding it."""
    p = p.rstrip("/") or "."
    return os.path.join(p, "simulation_output") if os.path.isdir(os.path.join(p, "simulation_output")) else p


def _run_dir(d):
    return os.path.dirname(os.path.abspath(d))


def _f(d, name):
    f = os.path.join(d, name)
    return f if os.path.exists(f) and os.path.getsize(f) > 0 else None


def masses(d):
    return np.fromfile(os.path.join(d, "masses.bin"), "<f4")


def element_of(m):
    if 0.5 <= m < 2.5:
        return "H"
    el, dm = min(((e, abs(m - s)) for e, s in STD_MASS.items()), key=lambda t: t[1])
    return el if dm < 0.5 else f"m{m:.0f}"


def elements(d):
    return np.array([element_of(float(m)) for m in masses(d)])


def table(path, names=None):
    """A text log as {column: array}; the names from its '#' header (or given)."""
    if path is None or not os.path.exists(path):
        return None
    head = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                if head is None:
                    head = line[1:].split()
            else:
                break
    try:
        a = np.loadtxt(path, ndmin=2, comments="#")
    except ValueError:
        return None
    if a.size == 0:
        return None
    names = names or head or [f"c{i}" for i in range(a.shape[1])]
    return {n: a[:, i] for i, n in enumerate(names[:a.shape[1]])}


def run_log(d):
    """The mdrun log next to simulation_output (the one with the sfmd parameters), as text, or ''."""
    for f in sorted(glob.glob(os.path.join(_run_dir(d), "*.log")), key=os.path.getmtime, reverse=True):
        txt = open(f, errors="ignore").read()
        if "sfmd-charge-output-stride" in txt:
            return txt
    return ""


def run_params(d):
    """tinit [fs], the tpr's dt [fs], sfmd-charge-output-stride, peak and FWHM [fs] from the log's parameter dump
    (written before do_md applies a schedule)."""
    txt = run_log(d)

    def g(name, default=None):
        m = re.search(r"^\s*" + re.escape(name) + r"\s*=\s*([-+0-9.eE]+)", txt, re.M)
        return float(m.group(1)) if m else default
    p = {"tinit": g("tinit", 0.0) * 1e3, "dt": g("delta_t", None), "stride": g("sfmd-charge-output-stride", None),
         "peak": g("sfmd-pulse-peak-time", None), "fwhm": g("sfmd-pulse-fwhm", None)}
    if p["dt"] is not None:
        p["dt"] *= 1e3
    return p


def read_charges(d):
    """charges_over_time.bin: frame times [fs] and (frames, atoms) float32."""
    n = len(masses(d))
    q = np.fromfile(os.path.join(d, "charges_over_time.bin"), "<f4")
    q = q[:len(q) // n * n].reshape(-1, n)
    p = run_params(d)
    if p["dt"] and p["stride"]:
        t = p["tinit"] + np.arange(len(q)) * p["stride"] * p["dt"]
    else:
        # no log: spread the frames evenly up to the last step (exact without a schedule only by luck)
        mq = table(_f(d, "mean_charge_vs_time.txt"), ["t", "q", "n", "on"])
        t_end = mq["t"][-1] * 1e3 if mq else 0.0
        t = np.linspace(0, t_end, len(q)) if len(q) > 1 else np.zeros(1)
        print("note: no run log with the tpr's dt and stride: charge frames spread evenly", file=sys.stderr)
    return t, q


def read_events(d):
    """ADK events (transition_log.txt) and electron-impact ionizations / three-body recombinations
    (collisional_log.txt), merged and sorted: t [fs], atom (0-based), elem, kind, q0, q1."""
    rows = []
    for name, kind in (("transition_log.txt", None), ("collisional_log.txt", None)):
        f = _f(d, name)
        if f is None:
            continue
        for line in open(f):
            w = line.split()
            if len(w) < 5 or line.startswith("#"):
                continue
            q0, q1 = int(w[3]), int(w[4])
            k = 0 if name == "transition_log.txt" else (1 if q1 > q0 else 2)
            rows.append((float(w[0]) * 1e3, int(w[1]), w[2].capitalize(), k, q0, q1))
    ev = np.array(rows, dtype=[("t", "f8"), ("atom", "i4"), ("elem", "U3"), ("kind", "i1"), ("q0", "i2"), ("q1", "i2")])
    return np.sort(ev, order="t") if len(ev) else ev


def screening_times(d, steps):
    """Times [fs] of screening_log.txt's steps, interpolated in energy_ledger.txt (step and t columns): not
    steps x dt, which is wrong under sfmd-schedule."""
    led = table(_f(d, "energy_ledger.txt"))
    if led is None or "step" not in led or len(led["step"]) < 2:
        return None
    return np.interp(steps, led["step"], led["t[ps]"] * 1e3)


def pulse_band(d):
    """(start, end) [fs] of the pulse above half its peak intensity, from pulse_profile.txt."""
    pp = table(_f(d, "pulse_profile.txt"), ["t", "I", "E", "on"])
    if pp is None or pp["I"].max() <= 0:
        return None
    half = pp["t"][pp["I"] >= 0.5 * pp["I"].max()] * 1e3
    return half.min(), half.max()


# ------------------------------------------------------------------ commands

def cmd_summary(a):
    d = _dir(a.dir)
    log = run_log(d)
    el = elements(d)
    n = len(el)
    print(f"# {os.path.abspath(_run_dir(d))}")
    m = re.search(r"^ Pulse .*\n.*$", log, re.M)
    if m:
        print(m.group(0))
    m = re.search(r"^ Run .*$", log, re.M)
    if m:
        print(m.group(0))
    m = re.search(r"MolDStruct SF: end of the run at .*", log)
    if m:
        print(" End" + m.group(0).split("end of the run", 1)[1])

    if _f(d, "charges.txt"):
        q = np.loadtxt(os.path.join(d, "charges.txt"), ndmin=2)[:, 1]
        print(f"\nfinal charge: total {q.sum():+.0f}, {q.mean():.3f} per atom, heavy atoms "
              f"{q[el != 'H'].mean():.3f}")
        for e in sorted(set(el), key=lambda x: -STD_MASS.get(x, 0)):
            s = el == e
            hist = np.bincount(q[s].astype(int).clip(0))
            print(f"  {e:2s} {s.sum():6d} atoms  mean {q[s].mean():5.2f}   charge states "
                  + " ".join(f"{k}:{c}" for k, c in enumerate(hist) if c))

    ev = read_events(d)
    if len(ev):
        print("\nevents (from transition_log.txt / collisional_log.txt; need sfmd-detailed-output = 1)")
        els = sorted(set(ev["elem"]), key=lambda x: -STD_MASS.get(x, 0))
        print("  kind     total  " + " ".join(f"{e:>7s}" for e in els))
        for k, kn in enumerate(KINDS):
            s = ev["kind"] == k
            if s.any():
                print(f"  {kn:7s} {s.sum():6d}  " + " ".join(f"{(s & (ev['elem'] == e)).sum():7d}" for e in els))
        t0, t1 = 5 * np.floor(ev["t"].min() / 5), 5 * np.ceil(ev["t"].max() / 5)
        edges = np.arange(t0, t1 + 5, 5)
        print("  per 5 fs bin:  t[fs]  " + " ".join(f"{k:>7s}" for k in KINDS))
        for i in range(len(edges) - 1):
            s = (ev["t"] >= edges[i]) & (ev["t"] < edges[i + 1])
            if s.any():
                print(f"             {edges[i]:6.0f}  " + " ".join(f"{(s & (ev['kind'] == k)).sum():7d}"
                                                          for k in range(3)))
    lz = table(_f(d, "laser_log.txt"))
    pb = table(_f(d, "pb_log.txt"))
    if pb is not None:
        Nmax = pb["N"].max()
        print(f"\nelectron cloud: {len(pb['N'])} PB solves; trapped at most {Nmax:.0f} at "
              f"{pb['t[ps]'][pb['N'].argmax()] * 1e3:.1f} fs, at the end {pb['N'][-1]:.0f}; T_e at most "
              f"{pb['T_e[eV]'].max():.1f} eV")
        if lz is not None:
            last = {k: v[-1] for k, v in lz.items()}
            print(f"  escaped at birth {last.get('escaped_at_birth_total', 0):.0f}, evaporated "
                  f"{last.get('evaporated_total', 0):.0f}, spilled {last.get('spilled_total', 0):.0f}; "
                  f"laser heating {last.get('laser_energy_total[eV]', 0) / 1e3:.1f} keV")
        for line in re.findall(r"^sfelectrons: (?:electron-impact|three-body|\d+ steps).*$", log, re.M):
            print("  " + line.replace("sfelectrons: ", ""))
    ad = _f(d, "autostop_dirs.txt")
    if ad:
        a2 = table(ad, ["t", "turn", "pred", "n"])
        if a2 is not None:
            print(f"\nautostop 2: last check at {a2['t'][-1]:.1f} fs, median turn {a2['turn'][-1]:.3f} deg, predicted "
                  f"rest {a2['pred'][-1]:.2f} deg")
    m = re.search(r"^Simulation terminated:.*$", log, re.M)
    if m:
        print(m.group(0))


def cmd_charges(a):
    d = _dir(a.dir)
    el = elements(d)
    t, q = read_charges(d)
    els = sorted(set(el), key=lambda x: -STD_MASS.get(x, 0))
    print("# t[fs]    mean  heavy " + " ".join(f"{e:>6s}" for e in els) + "   (charges_over_time.bin)")
    for i in range(0, len(q), a.every):
        print(f"{t[i]:7.2f} {q[i].mean():7.3f} {q[i][el != 'H'].mean():6.3f} "
              + " ".join(f"{q[i][el == e].mean():6.3f}" for e in els))


def cmd_transitions(a):
    d = _dir(a.dir)
    ev = read_events(d)
    if a.type:
        ks = [KINDS.index(k) for k in a.type.split(",")]
        ev = ev[np.isin(ev["kind"], ks)]
    if a.atom is not None:
        ev = ev[ev["atom"] == a.atom - 1]
    if a.tmin is not None:
        ev = ev[ev["t"] >= a.tmin]
    if a.tmax is not None:
        ev = ev[ev["t"] <= a.tmax]
    print("# t[fs] atom(1-based) element type q_before -> q_after")
    for e in ev:
        print(f"{e['t']:9.4f} {e['atom'] + 1:6d} {e['elem']:2s}  {KINDS[e['kind']]:7s} {e['q0']} -> {e['q1']}")


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _dir(a.dir)
    el = elements(d)
    n = len(el)
    band = pulse_band(d)
    mq = table(_f(d, "mean_charge_vs_time.txt"), ["t", "q", "n", "on"])
    pp = table(_f(d, "pulse_profile.txt"), ["t", "I", "E", "on"])
    pb = table(_f(d, "pb_log.txt"))
    lz = table(_f(d, "laser_log.txt"))
    scr = table(_f(d, "screening_log.txt"))
    ad = table(_f(d, "autostop_dirs.txt"), ["t", "turn", "pred", "n"])
    ev = read_events(d)
    t_end = mq["t"][-1] * 1e3 if mq is not None else None
    panels = []

    def panel(f):
        panels.append(f)
        return f

    def series(ax, x, y, k, label, ls="-"):
        ax.plot(x, y, color=SERIES[k], lw=2, ls=ls, label=label)

    if pp is not None:
        @panel
        def _(ax):
            series(ax, pp["t"] * 1e3, pp["I"], 0, "intensity")
            ax.set_ylabel("W/cm²"); ax.set_title("laser intensity")

    if os.path.exists(os.path.join(d, "charges_over_time.bin")) and os.path.getsize(os.path.join(d, "charges_over_time.bin")):
        t, q = read_charges(d)
        els = sorted(set(el), key=lambda x: -STD_MASS.get(x, 0))[:7]

        @panel
        def _(ax):
            series(ax, t, q.mean(1), 0, "all atoms")
            for i, e in enumerate(els):
                series(ax, t, q[:, el == e].mean(1), i + 1, e)
            ax.set_ylabel("mean charge"); ax.set_title("ion charge per element")
            ax.legend(fontsize=7, frameon=False, ncol=2)
    elif mq is not None:
        @panel
        def _(ax):
            series(ax, mq["t"] * 1e3, mq["q"], 0, "mean")
            ax.set_ylabel("mean charge"); ax.set_title("mean ion charge")

    if len(ev):
        @panel
        def _(ax):
            for k, kn in enumerate(KINDS):
                s = ev["kind"] == k
                if s.any():
                    series(ax, ev["t"][s], np.arange(1, s.sum() + 1) / n, k, f"{kn} ({s.sum()})")
            ax.set_ylabel("events per atom"); ax.set_title("events, cumulative (legend: total)")
            ax.legend(fontsize=7, frameon=False)

    if pb is not None and mq is not None:
        @panel
        def _(ax):
            tp = pb["t[ps]"] * 1e3
            series(ax, tp, pb["N"], 0, "trapped (N)")
            series(ax, tp, pb["Q"], 1, "escaped = net charge (Q)")
            qion = np.interp(tp, mq["t"] * 1e3, mq["q"]) * n
            series(ax, tp, qion, 2, "ion charge (Q + N)", ls="--")
            ax.set_ylabel("electrons / e"); ax.set_title("trapped, escaped, ion charge")
            ax.legend(fontsize=7, frameon=False)

    if pb is not None:
        @panel
        def _(ax):
            series(ax, pb["t[ps]"] * 1e3, pb["T_e[eV]"], 0, "T_e at the solves")
            if scr is not None and "T_e[eV]" in scr:
                ts = screening_times(d, scr["step"])
                if ts is not None:
                    series(ax, ts, scr["T_e[eV]"], 1, "T_e (screening_log)", ls="--")
            ax.set_yscale("symlog", linthresh=1); ax.set_ylabel("eV"); ax.set_title("cloud temperature")
            ax.legend(fontsize=7, frameon=False)

    if lz is not None and "R[nm]" in lz:
        @panel
        def _(ax):
            series(ax, lz["t[ps]"] * 1e3, lz["R[nm]"], 0, "R")
            ax.set_ylabel("nm"); ax.set_title("sample radius (volume-equivalent, heavy atoms)")

        if pb is not None:
            @panel
            def _(ax):
                R = np.interp(pb["t[ps]"], lz["t[ps]"], lz["R[nm]"])
                ne = pb["N"] / (4 / 3 * np.pi * (R * 1e-7) ** 3)
                series(ax, pb["t[ps]"] * 1e3, np.where(ne > 0, ne, np.nan), 0, "n_e")
                ax.set_yscale("log"); ax.set_ylabel("cm⁻³"); ax.set_title("mean free-electron density N / V")

    if lz is not None and "heating[eV/fs]" in lz and lz["heating[eV/fs]"].max() > 0:
        @panel
        def _(ax):
            series(ax, lz["t[ps]"] * 1e3, lz["heating[eV/fs]"], 0, "heating")
            ax.set_ylabel("eV/fs"); ax.set_title("laser heating of the cloud")

    if ad is not None:
        @panel
        def _(ax):
            s = ad["pred"] >= 0
            series(ax, ad["t"], ad["turn"], 0, "median turn in the interval")
            series(ax, ad["t"][s], ad["pred"][s], 1, "predicted rest")
            ax.set_yscale("log"); ax.set_ylabel("deg"); ax.set_title("autostop 2: heavy-ion direction turn")
            ax.legend(fontsize=7, frameon=False)

    if not panels:
        print("nothing to plot"); return
    nc = 2 if len(panels) > 1 else 1
    nr = (len(panels) + nc - 1) // nc
    fig, axs = plt.subplots(nr, nc, figsize=(6.2 * nc, 2.9 * nr), squeeze=False)
    for ax, f in zip(axs.flat, panels):
        if band:
            ax.axvspan(*band, color=BAND, lw=0, zorder=0)
        f(ax)
        ax.set_xlabel("t [fs]", color=MUTED, fontsize=8)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c8c6c0")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.title.set_color(INK); ax.title.set_fontsize(9)
        if t_end:
            ax.set_xlim(left=0, right=t_end)
    for ax in axs.flat[len(panels):]:
        ax.axis("off")
    name = "/".join(_run_dir(d).split(os.sep)[-2:])
    fig.suptitle(f"{name}: {n} atoms" + ("   (grey band: pulse FWHM)" if band else ""), x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    out = a.o or os.path.join(_run_dir(d), "overview.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)
    if a.gif:
        make_gif(d, os.path.join(_run_dir(d), "explosion_noH.gif" if a.no_h else "explosion.gif"), a.every, no_h=a.no_h)
    if a.extract:
        extract(d, os.path.join(_run_dir(d), "extracted"))


# ------------------------------------------------------------------ gif, extract (from gromacs-mc's mcmd_output.py)

def read_trr(path):
    """Frames of a GROMACS .trr (XDR, single or double precision) without GROMACS: dicts with step, t [ps], x, v."""
    import struct
    with open(path, "rb") as fh:
        data = fh.read()
    o, n = 0, len(data)
    while o + 8 <= n:
        magic, = struct.unpack_from(">i", data, o)
        if magic != 1993:
            raise ValueError(f"{path}: bad frame magic {magic} at byte {o}")
        slen, sl = struct.unpack_from(">ii", data, o + 4)
        o += 12 + (sl + 3) // 4 * 4
        (ir, e, box, vir, pres, top, sym, xs, vs, fs, nat, step, nre) = struct.unpack_from(">13i", data, o)
        o += 52
        rs = box // 9 if box else (xs // (3 * nat) if xs else 4)
        rf = ">d" if rs == 8 else ">f"
        t, = struct.unpack_from(rf, data, o)
        o += 2 * rs + ir + e + box + vir + pres + top + sym
        arr = {}
        for key, size in (("x", xs), ("v", vs), ("f", fs)):
            arr[key] = (np.frombuffer(data, ">f8" if rs == 8 else ">f4", nat * 3, o).reshape(nat, 3).astype(float)
                        if size else None)
            o += size
        yield {"step": step, "t": t, "x": arr["x"], "v": arr["v"]}


def _find_trr(d):
    cands = sorted(glob.glob(os.path.join(_run_dir(d), "*.trr")))
    return cands[0] if cands else None


def _charges_at(d, times_fs):
    """Per-atom charges at the given times: the latest charges_over_time.bin frame at or before each."""
    tq, q = read_charges(d)
    idx = np.searchsorted(tq, np.asarray(times_fs) + 1e-6, side="right") - 1
    return q[np.clip(idx, 0, len(q) - 1)]


def make_gif(d, out, every=1, fps=10, no_h=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    trr = _find_trr(d)
    if trr is None:
        print(f"no .trr next to {d}: no gif"); return
    frames = [f for i, f in enumerate(read_trr(trr)) if i % every == 0 and f["x"] is not None]
    t = np.array([f["t"] for f in frames]) * 1e3          # the trr's own times: right under a schedule too
    X = np.array([f["x"] for f in frames])
    m, el = masses(d), elements(d)
    h = el == "H"
    Q = _charges_at(d, t)
    if no_h:
        X, Q, m, el = X[:, ~h], Q[:, ~h], m[~h], el[~h]
        h = el == "H"
    w = m * ~h
    c0 = (X[0] * w[:, None]).sum(0) / w.sum()
    G = ((X[0] - c0).T * w) @ (X[0] - c0) / w.sum()
    ev, V = np.linalg.eigh(G)
    V = V[:, ::-1]
    rg = np.sqrt(ev[::-1])
    P = np.array([(x - (x * w[:, None]).sum(0) / w.sum()) @ V for x in X])
    lims = np.maximum.accumulate([np.percentile(np.abs(p[~h]), 99.5) * 1.15 for p in P])
    order = np.argsort(~h)
    size = np.maximum(np.where(h, 0.8, 3.0)[order] * min(1.0, (2000 / len(m)) ** 0.5), 0.15)
    vmax = max(1.0, float(np.percentile(Q[-1], 99.5)))
    on = pulse_band(d)
    views = [((1, 2), "view along axis 1 (longest)"), ((0, 2), "view along axis 2"), ((0, 1), "view along axis 3")]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.9), dpi=90)
    fig.patch.set_facecolor("white")
    sc, bars = [], []
    for ax, ((i, j), title) in zip(axs, views):
        sc.append(ax.scatter(P[0][order, i], P[0][order, j], s=size, c=Q[0][order], cmap="plasma", vmin=0,
                             vmax=vmax, lw=0, rasterized=True))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c8c6c0")
        ax.set_title(title, fontsize=9, color=MUTED)
        bars.append((ax.plot([], [], color=INK, lw=2)[0], ax.text(0, 0, "", ha="center", fontsize=8, color=INK)))
    cb = fig.colorbar(sc[0], cax=fig.add_axes([0.3, 0.12, 0.4, 0.03]), orientation="horizontal")
    cb.set_label("ion charge", fontsize=9)
    head = fig.suptitle("", x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.2, wspace=0.04)
    name = "/".join(_run_dir(d).split(os.sep)[-2:])
    axes_txt = f"R_g axes {rg[0]:.2f} / {rg[1]:.2f} / {rg[2]:.2f} nm"

    def draw(k):
        lim = lims[k]
        bar = next(v for v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if v >= 0.3 * lim)
        for s_, ((i, j), _), ax, (ln, tx) in zip(sc, views, axs, bars):
            s_.set_offsets(np.c_[P[k][order, i], P[k][order, j]])
            s_.set_array(Q[k][order])
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ln.set_data([-0.9 * lim, -0.9 * lim + bar], [-0.9 * lim] * 2)
            tx.set_position((-0.9 * lim + bar / 2, -0.85 * lim)); tx.set_text(f"{bar:g} nm")
        pulse = "  pulse on (FWHM)" if on and on[0] <= t[k] <= on[1] else ""
        head.set_text(f"{name}: {len(m)} {'heavy ' if no_h else ''}atoms ({axes_txt})    t = {t[k]:5.1f} fs    "
                      f"mean charge {Q[k].mean():.2f}{pulse}")
        return sc

    FuncAnimation(fig, draw, frames=len(t), blit=False).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(out)


def extract(d, outdir):
    """Human-readable copies: transitions.txt (all events) and charges_over_time.txt (time, then one column per
    atom)."""
    os.makedirs(outdir, exist_ok=True)
    ev = read_events(d)
    if len(ev):
        with open(os.path.join(outdir, "transitions.txt"), "w") as fo:
            fo.write("# t[fs] atom(1-based) element type q_before -> q_after\n")
            for e in ev:
                fo.write(f"{e['t']:9.4f} {e['atom'] + 1:6d} {e['elem']:2s}  {KINDS[e['kind']]:7s} {e['q0']} -> {e['q1']}\n")
        print(os.path.join(outdir, "transitions.txt"))
    if _f(d, "charges_over_time.bin"):
        t, q = read_charges(d)
        np.savetxt(os.path.join(outdir, "charges_over_time.txt"), np.c_[t, q], fmt="%.4g",
                   header=f"t[fs] then the charge of atoms 1..{q.shape[1]} (one row per charges_over_time.bin frame)")
        print(os.path.join(outdir, "charges_over_time.txt"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = "simulation_output or the run directory holding it (default: .)"
    p = sub.add_parser("summary"); p.add_argument("dir", nargs="?", default=".", help=h); p.set_defaults(f=cmd_summary)
    p = sub.add_parser("charges"); p.add_argument("dir", nargs="?", default=".", help=h)
    p.add_argument("--every", type=int, default=10); p.set_defaults(f=cmd_charges)
    p = sub.add_parser("transitions"); p.add_argument("dir", nargs="?", default=".", help=h)
    p.add_argument("--type", help="comma list of adk, impact, recomb"); p.add_argument("--atom", type=int, help="1-based")
    p.add_argument("--tmin", type=float); p.add_argument("--tmax", type=float); p.set_defaults(f=cmd_transitions)
    p = sub.add_parser("plot"); p.add_argument("dir", nargs="?", default=".", help=h)
    p.add_argument("-o", help="output png (default <run>/overview.png)")
    p.add_argument("--gif", action="store_true"); p.add_argument("--every", type=int, default=1, help="gif: every Nth frame")
    p.add_argument("--no-h", action="store_true"); p.add_argument("--extract", action="store_true")
    p.set_defaults(f=cmd_plot)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
