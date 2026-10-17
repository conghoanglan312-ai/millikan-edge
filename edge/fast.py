"""The hottest pieces of the live path, made cheaper without changing an answer.

    from edge import fast; fast.install(P)

Profiled on the board (spray_labelled_0907, window on, INT8 gate): Track.recentSpeed
took 21% of the live path's time.  It is called ~50 times a frame -- the hold check,
the hunt aid, the pick -- often for the same track in the same frame, and each call
tests every one of its 30 points against every grid line in a Python loop.  The hunt
aid's Track.windowFit scans a track's whole history to find its last 3 s.  With those
two fixed, assign() is the largest: in the dense part of the spray (~90 live tracks,
~40 blobs) it fills its cost matrix one pair at a time, ~15 ms a frame.

Same arithmetic, same answers:
  recentSpeed   the grid-line test as one numpy expression (the same float64
                subtraction, abs and comparison), the same np.polyfit; remembered
                per track until the track gains a point -- points are only ever
                appended, so their count is the track's state
  windowFit     the same predicate, now - t <= seconds, found by bisection
                (times only grow along a track, so the predicate is monotonic in
                the index) instead of a scan; the same polyfit on the same points
  assign        the cost matrix as elementwise numpy over all pairs at once: the
                same float64 operations on the same operands, so the same matrix to
                the bit, then the same linear_sum_assignment
Installed from outside cv/, which stays untouched, like the classifier gate.

    python edge/fast.py check <fixture> <frames>

replays the first <frames> of a recording through edge/suggestions.py with and
without it and compares every fall, ring, pick and sighting.

    python edge/fast.py exact <source> [frames]

replays a recording (a fixture name, or frames:<folder>) with both assign()s on every
call and stops at the first cost matrix that differs in a single bit.
"""
import json
import os
import sys

import numpy as np


def install(P):
    Track = P.Track
    grid = P.GRID_BIAS_PX
    fitS = P.CANDIDATE_FIT_S

    def speed(self, spacing, window, rows):
        pts = self.points[-window:]
        if len(pts) < 8:
            return None
        a = np.array(pts, dtype = float)
        if rows:
            far = np.abs(a[:, 1, None] - np.array(rows, dtype = float)[None, :]) \
                .min(axis = 1) > grid
            if int(far.sum()) >= 8:
                a = a[far]
        if a[-1, 0] == a[0, 0]:
            return None
        return float(np.polyfit(a[:, 0], a[:, 1], 1)[0]) / spacing

    def recentSpeed(self, spacing, window = 30):
        rows = getattr(self, "gridRows", None)
        key = (len(self.points), spacing, window, tuple(rows) if rows else None)
        memo = self.__dict__.get("_speedMemo")
        if memo is not None and memo[0] == key:
            return memo[1]
        v = speed(self, spacing, window, rows)
        self._speedMemo = (key, v)
        return v

    def windowFit(self, spacing, now, seconds = None):
        seconds = fitS if seconds is None else seconds
        pts = self.points
        lo, hi = 0, len(pts)
        while lo < hi:                       # first point with now - t <= seconds
            mid = (lo + hi) // 2
            if now - pts[mid][0] <= seconds:
                hi = mid
            else:
                lo = mid + 1
        pts = pts[lo:]
        if len(pts) < 20 or spacing <= 0:
            return None
        a = np.array(pts, dtype = float)
        ts, ys = a[:, 0], a[:, 1]
        if ts[-1] - ts[0] < seconds * 0.6:
            return None
        m, c = np.polyfit(ts, ys, 1)
        resid = float(np.sqrt(np.mean((ys - (m * ts + c)) ** 2)))
        return float(m) / spacing, resid, float(ys[-1] - ys[0]) / spacing

    Track.recentSpeed = recentSpeed
    Track.windowFit = windowFit
    P.assign = assignFor(P)


def assignFor(P):
    """cv/pipeline.py's assign() with its cost matrix built in one go.

    The original fills the matrix pair by pair -- for every track, every blob, a
    scalar np.hypot and a few comparisons -- which in a dense spray (90 live tracks,
    40 blobs) is ~3,600 trips through Python a frame.  Here every entry is the same
    float64 expression on the same operands, evaluated elementwise: the same np.hypot,
    abs, division, product and sum, the same comparisons in the same sense, so the
    matrix is the same to the bit and so is linear_sum_assignment's answer.  The
    measuring-first split and the greedy fallback are copied as they are."""

    def assign(tracks, centres, maxStep, areaFloor, now, xGate):
        if not tracks or not centres:
            return {}
        measuring = [i for i, t in enumerate(tracks)
                     if t.tTop is not None and t.tBottom is None]
        if measuring and len(measuring) < len(tracks):
            first = assign([tracks[i] for i in measuring], centres, maxStep, areaFloor,
                           now, xGate)
            pairs = {measuring[r]: c for r, c in first.items()}
            taken = set(pairs.values())
            restI = [i for i in range(len(tracks)) if i not in set(measuring)]
            restC = [j for j in range(len(centres)) if j not in taken]
            if restI and restC:
                second = assign([tracks[i] for i in restI], [centres[j] for j in restC],
                                maxStep, areaFloor, now, xGate)
                for r, c in second.items():
                    pairs[restI[r]] = restC[c]
            return pairs
        BIG = 1e6
        n = len(tracks)
        ax, ay, px, denom, tol = (np.empty(n) for _ in range(5))
        wide = np.zeros(n, bool)
        yLo, yHi = np.full(n, np.nan), np.full(n, np.nan)
        for i, t in enumerate(tracks):
            aim = t.predict(now)
            ax[i], ay[i], px[i] = aim[0], aim[1], t.pos[0]
            if t.miss > 0 and len(t.points) >= P.CONFIRM_POINTS and t.vel[1] != 0:
                wide[i] = True
                slack = max(4.0, P.REID_Y_SLACK * abs(aim[1] - t.pos[1]))
                if t.vel[1] > 0:
                    yLo[i], yHi[i] = t.pos[1] - 4.0, aim[1] + slack
                else:
                    yLo[i], yHi[i] = aim[1] - slack, t.pos[1] + 4.0
            denom[i] = max(t.area, areaFloor)
            tol[i] = (P.MEASURING_AREA_TOLERANCE
                      if (t.miss > 0 and t.tTop is not None and t.tBottom is None)
                      else P.AREA_TOLERANCE)
        c = np.array([(q[0], q[1], q[2]) for q in centres], dtype = float)
        cx, cy, ca = c[None, :, 0], c[None, :, 1], c[None, :, 2]
        d = np.hypot(cx - ax[:, None], cy - ay[:, None])
        far = d > maxStep
        outside = ((np.abs(cx - px[:, None]) > xGate)
                   | ~((yLo[:, None] <= cy) & (cy <= yHi[:, None])))
        own = np.array([t.area for t in tracks], dtype = float)[:, None]
        areaDiff = np.abs(ca - own) / denom[:, None]
        skip = (far & (~wide[:, None] | outside)) | (areaDiff > tol[:, None])
        cost = np.where(skip, BIG, d + areaDiff * maxStep)
        pairs = {}
        if P.HAVE_SCIPY:
            rows, cols = P.linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < BIG:
                    pairs[int(r)] = int(c)
        else:
            order = np.argsort(cost, axis = None)
            takenT, takenC = set(), set()
            for flat in order:
                r, c = int(flat // cost.shape[1]), int(flat % cost.shape[1])
                if cost[r, c] >= BIG:
                    break
                if r in takenT or c in takenC:
                    continue
                takenT.add(r); takenC.add(c); pairs[r] = c
        return pairs

    return assign


def child(fixture, frames, out, useFast):
    """One replay of edge/suggestions.py (classifier after the gates), cut short."""
    from cv import pipeline as P
    from edge import suggestions as SG
    if useFast:
        install(P)
    SG.OUT = out
    orig = P.trackAll
    P.trackAll = lambda *a, **k: orig(*a, toFrame = frames, **k)
    SG.run("after", fixture)


def check(fixture, frames):
    """Replay with and without, then compare every ring, pick, sighting and fall."""
    import glob
    import subprocess
    import tempfile
    root = tempfile.mkdtemp(prefix = "fast_check_")
    runs = []
    for useFast in (0, 1):
        out = os.path.join(root, str(useFast))
        runs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "child",
                                      fixture, str(frames), out, str(useFast)],
                                     stdout = subprocess.DEVNULL))
    for r in runs:
        r.wait()
    same = True
    a, b = [np.load(os.path.join(root, str(k), "%s_after.npz" % fixture)) for k in (0, 1)]
    for key in ("rings", "picks", "sightings"):
        eq = a[key].shape == b[key].shape and np.array_equal(a[key], b[key])
        print("%-10s %s  (%d rows)" % (key, "same" if eq else "DIFFERENT", len(a[key])))
        same &= eq
    fa, fb = [sorted(glob.glob(os.path.join(root, str(k), "runs", fixture, "after", "*falls.csv")))
              for k in (0, 1)]
    for x, y in zip(fa, fb):
        with open(x, encoding = "utf-8") as f, open(y, encoding = "utf-8") as g:
            eq = f.read() == g.read()
        print("%-10s %s" % (os.path.basename(x), "same" if eq else "DIFFERENT"))
        same &= eq
    sa, sb = [json.load(open(os.path.join(root, str(k), "%s_after.json" % fixture)))
              for k in (0, 1)]
    print("replay minutes: %.1f without, %.1f with" % (sa["replayMinutes"], sb["replayMinutes"]))
    print("ALL SAME" if same and fa else "NOT THE SAME", flush = True)


def exact(source, frames = None):
    """Both assign()s on every call of a replay: the same cost matrices, bit for bit,
    and the same pairs.  linear_sum_assignment is watched to see each matrix."""
    import time
    from cv import pipeline as P
    from edge import common as C
    from edge import gate as G
    if not source.startswith("frames:"):
        source = "frames:" + C.fixture(source)
    try:                                 # on the board, the live path's own gate
        import rknnlite                  # noqa: F401
        from edge import live_board as LB
        gate = G.BatchedNpuGate({b: os.path.join(C.REPO, "results", "int8", f)
                                 for b, f in LB.RKNN.items()})
    except ImportError:
        gate = G.Gate(os.path.join(C.REPO, "results", "fp32", "model.onnx"))
    G.install(P, gate, "after")
    orig = P.assign
    install(P)
    mine = P.assign
    seen = []
    lsa = P.linear_sum_assignment

    def watched(cost):
        seen.append(cost.copy())
        return lsa(cost)
    P.linear_sum_assignment = watched
    n = {"calls": 0, "matrices": 0, "cells": 0, "pairs": 0, "orig_s": 0.0, "mine_s": 0.0}

    def both(tracks, centres, *a):
        del seen[:]
        P.assign = orig                  # the original recurses through the global
        t0 = time.perf_counter()
        try:
            want = orig(tracks, centres, *a)
        finally:
            P.assign = both
        t1 = time.perf_counter()
        wantM, seen[:] = list(seen), []
        got = mine(tracks, centres, *a)
        t2 = time.perf_counter()
        gotM = list(seen)
        if want != got or len(wantM) != len(gotM) or any(
                x.shape != y.shape or x.tobytes() != y.tobytes() for x, y in zip(wantM, gotM)):
            sys.exit("DIFFERENT at call %d: %d tracks, %d blobs" % (n["calls"] + 1,
                                                                   len(tracks), len(centres)))
        n["calls"] += 1
        n["matrices"] += len(gotM)
        n["cells"] += sum(x.size for x in gotM)
        n["pairs"] += len(got)
        n["orig_s"] += t1 - t0
        n["mine_s"] += t2 - t1
        return got
    P.assign = both
    P.trackAll(source, show = False, saveVideo = False,
               outDir = os.path.join(C.DATA, "fast_exact"), toFrame = frames)
    print("ALL SAME: %(calls)d calls, %(matrices)d cost matrices (%(cells)d cells), "
          "%(pairs)d pairs; assign %(orig_s).1f s -> %(mine_s).1f s" % n, flush = True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if sys.argv[1] == "child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == "1")
    elif sys.argv[1] == "check":
        check(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "exact":
        exact(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
