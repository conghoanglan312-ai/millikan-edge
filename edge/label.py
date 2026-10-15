"""Hand-label droplets on frames of any raw recording, blind to the pipeline.

    python edge/label.py sample <recording>          choose the frames, once (seeded)
    python edge/label.py label  <recording>          label them; resumes where you stopped
    python edge/label.py review <recording>          blind second look, after scoring

    -> labels/<recording>/sample.json, labels.json, review.json   (tracked in git)

The parent's test/tools/detacc_label.py -- the tool behind the 1,863 marks on
bench_0908 -- made general: same screens, same keys, same snapping, same magnifier,
same rule for what a droplet is.  What changed is only what was tied to bench_0908:
its phases came from that session's spray and measuring times, which a new
recording does not have, so frames are drawn from four equal slices of the session
instead.  Blind as before: the screen shows neither the frame number nor where in
the session a frame came from; frames come in random order, and REPEATS of them come
round again unannounced, so the labeller's own consistency is measured too.

A droplet is a small bright dot that is still there, slightly moved, a frame or two
either side.  Static speckle is not.  Clicks snap to the dot's brightness centre, and
labels are stored in full-frame pixels.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge import common as C                                         # noqa: E402

FRAMES = 100            # labelled per recording: ~2 h; bench_0908 had 150 over 4 phases
SLICES = 4              # equal slices of the session, FRAMES // SLICES drawn from each
REPEATS = 10            # frames shown a second time, unannounced
REPEAT_GAP = 30         # ...no sooner than this many screens later
MIN_GAP_S = 3.0         # two sampled frames at least this far apart
SEED = 20261014
STRETCH_GAIN = 2.5
STRETCH_BG = 45            # where the background lands after the stretch
STRETCH = None             # per recording, from its background (use())

CONTEXT = 12            # frames either side, for seeing what moves
SCALE = 1.2             # main view; fits a 1646 x 926 desktop
BAR = 30
MAG_OUT = 384           # magnifier, screen px
ZOOMS = (6, 8, 12)
FONT = cv2.FONT_HERSHEY_SIMPLEX
CLAHE = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))
CONTRASTS = ("raw", "stretch", "clahe")
SURE, UNSURE = (0, 220, 0), (0, 220, 255)

HELP = ("left click     droplet   (click again = remove)",
        "right click    unsure: faint or ambiguous",
        "space          next frame",
        "w / backspace  previous frame",
        "a / d          step -1 / +1 frame to see motion",
        "[ ]            step -5 / +5 frames",
        "s              back to the frame being labelled",
        "f              play -12..+12 frames, looping",
        "v              contrast       h   hide marks",
        "z              undo last mark",
        "c              magnifier zoom",
        "q / esc        save and quit",
        "",
        "Mark every droplet in the picture, on lines too.",
        "A droplet is still there, slightly moved, a frame",
        "or two either side. Speckle that blinks is not.",
        "Unsure costs nothing either way -- use it.")



FIX = {"name": None}
SAMPLE = None


def use(recording):
    """Point every path at one recording."""
    global SAMPLE
    FIX["name"] = recording
    FIX["folder"] = C.fixture(recording)
    FIX["labels"] = os.path.join(C.REPO, "labels", recording)
    SAMPLE = os.path.join(FIX["labels"], "sample.json")
    if not os.path.isdir(FIX["folder"]):
        sys.exit("no recording at %s" % FIX["folder"])
    setStretch()


def setStretch():
    """x2.5 around the recording's own background gray, not a fixed level.

    The parent's table (2.5 v - 30) suited bench_0908, background ~30; units differ
    (25 to 122 on 2026-10-15) and on a bright one the fixed table saturates to white.
    Same table as before at a background of 30.
    """
    global STRETCH
    level = 30.0
    bg = cv2.imread(os.path.join(FIX["folder"], "background.png"), cv2.IMREAD_GRAYSCALE)
    if bg is not None and os.path.exists(os.path.join(FIX["folder"], "roi.txt")):
        x, y, w, h = fixtureRoi()
        level = float(np.median(bg[y:y + h, x:x + w]))
    STRETCH = np.clip((np.arange(256) - level) * STRETCH_GAIN + STRETCH_BG,
                      0, 255).astype(np.uint8)


def framePath(i):
    return os.path.join(FIX["folder"], "f%06d.png" % i)


def fixtureRoi():
    info = dict(l.split() for l in open(os.path.join(FIX["folder"], "roi.txt")))
    return int(info["x"]), int(info["y"]), int(info["w"]), int(info["h"])


def labelsPath(name):
    return os.path.join(FIX["labels"], "labels.json" if name == "owner" else
                        "labels_%s.json" % name)


def todoPath(name):
    return os.path.join(FIX["labels"], "review_todo.json")


def reviewPath(name):
    return os.path.join(FIX["labels"], "review.json")


def makeSample(force):
    if os.path.exists(SAMPLE) and not force:
        sys.exit("%s already exists. The frames are chosen once, before anything is "
                 "scored; --force only if nothing has been labelled yet." % SAMPLE)
    t = np.load(os.path.join(FIX["folder"], "times.npy"))
    edges = np.linspace(t[0], t[-1], SLICES + 1)
    rng = np.random.default_rng(SEED)
    chosen = []
    for k in range(SLICES):
        name = "slice%d" % (k + 1)
        idx = np.where((t >= edges[k]) & (t <= edges[k + 1]))[0]
        idx = idx[(idx >= CONTEXT) & (idx < len(t) - CONTEXT)]
        acc = []
        for i in rng.permutation(idx):
            if all(abs(t[i] - t[j]) >= MIN_GAP_S for j in acc):
                acc.append(int(i))
                if len(acc) == FRAMES // SLICES:
                    break
        for i in acc:
            chosen.append({"id": len(chosen), "frame": i, "t": round(float(t[i]), 3),
                           "stratum": name, "repeatOf": None})
    seq = [chosen[o] for o in rng.permutation(len(chosen))]
    for rid in rng.choice(len(chosen), REPEATS, replace = False):
        at = next(k for k, it in enumerate(seq)
                  if it["id"] == rid and it["repeatOf"] is None)
        lo = min(at + REPEAT_GAP, len(seq))
        seq.insert(int(rng.integers(lo, len(seq) + 1)),
                   dict(chosen[rid], id = 1000 + int(rid), repeatOf = int(rid)))
    items = [dict(it, pos = k) for k, it in enumerate(seq)]
    C.saveJson(SAMPLE, {"seed": SEED, "fixture": FIX["name"], "slices": SLICES,
                        "repeats": REPEATS, "minGapS": MIN_GAP_S, "items": items})
    print("%d screens (%d frames + %d repeats) -> %s"
          % (len(items), len(chosen), REPEATS, SAMPLE))


class Frames(object):
    def __init__(self):
        self.cache = {}

    def get(self, i):
        g = self.cache.get(i)
        if g is None:
            if len(self.cache) > 90:
                self.cache.clear()
            g = cv2.imread(framePath(i), cv2.IMREAD_GRAYSCALE)
            self.cache[i] = g
        return g


def contrast(g, mode):
    if mode == 1:
        return cv2.LUT(g, STRETCH)
    if mode == 2:
        return CLAHE.apply(g)
    return g


def snap(g, fx, fy):
    """Brightness-weighted centre of the dot nearest the click, or the click itself."""
    x0, y0 = int(round(fx)), int(round(fy))
    ys, ye = max(0, y0 - 10), min(g.shape[0], y0 + 11)
    xs, xe = max(0, x0 - 10), min(g.shape[1], x0 + 11)
    win = g[ys:ye, xs:xe].astype(np.float32)
    if win.size == 0:
        return fx, fy
    base = float(np.median(win))
    blur = cv2.GaussianBlur(win, (3, 3), 0)
    yy, xx = np.mgrid[ys:ye, xs:xe]
    near = (yy - fy) ** 2 + (xx - fx) ** 2 <= 16.0
    if not near.any():
        return fx, fy
    k = int(np.argmax(np.where(near, blur, -1.0)))
    py, px = int(yy.flat[k]), int(xx.flat[k])
    if blur.flat[k] - base < 8.0:
        return fx, fy                   # nothing dot-like there; keep the click
    sy, sx = max(0, py - 2), max(0, px - 2)
    patch = np.clip(g[sy:py + 3, sx:px + 3].astype(np.float32) - base, 0, None)
    if patch.sum() <= 0:
        return float(px), float(py)
    gy, gx = np.mgrid[sy:sy + patch.shape[0], sx:sx + patch.shape[1]]
    return float((gx * patch).sum() / patch.sum()), float((gy * patch).sum() / patch.sum())


def magnifier(g, cx, cy, mode, zoom, marks):
    half = MAG_OUT // (2 * zoom)
    cx, cy = int(round(cx)), int(round(cy))
    x0, y0 = cx - half, cy - half
    patch = np.zeros((2 * half, 2 * half), np.uint8)
    ys, ye = max(0, y0), min(g.shape[0], y0 + 2 * half)
    xs, xe = max(0, x0), min(g.shape[1], x0 + 2 * half)
    if ys < ye and xs < xe:
        patch[ys - y0:ye - y0, xs - x0:xe - x0] = g[ys:ye, xs:xe]
    big = cv2.resize(contrast(patch, mode), (2 * half * zoom, 2 * half * zoom),
                     interpolation = cv2.INTER_NEAREST)
    big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    for px, py, colour, radius in marks:
        p = (int(round((px - x0) * zoom)), int(round((py - y0) * zoom)))
        if 0 <= p[0] < big.shape[1] and 0 <= p[1] < big.shape[0]:
            cv2.circle(big, p, int(radius * zoom), colour, 1)
    c = big.shape[0] // 2
    for a, b in (((c - 14, c), (c - 5, c)), ((c + 5, c), (c + 14, c)),
                 ((c, c - 14), (c, c - 5)), ((c, c + 5), (c, c + 14))):
        cv2.line(big, a, b, (0, 0, 255), 1)
    return big


def label(name):
    sample = C.loadJson(SAMPLE)
    if sample is None:
        sys.exit("no sample.json -- run:  python edge/label.py sample %s" % FIX["name"])
    items = sample["items"]
    path = labelsPath(name)
    data = C.loadJson(path, {"name": name, "seed": sample["seed"], "items": {}})
    if data.get("seed") != sample["seed"]:
        sys.exit("%s belongs to a different sample" % path)
    X, Y, W, H = fixtureRoi()
    frames = Frames()

    def rec(pos):
        return data["items"].setdefault(str(pos), {"frame": items[pos]["frame"],
                                                   "points": [], "done": False,
                                                   "seconds": 0.0})

    start = next((k for k in range(len(items)) if not rec(k)["done"]), len(items) - 1)
    st = {"pos": start, "off": 0, "play": False, "dir": 1, "tick": 0.0, "mode": 1,
          "hide": False, "zoom": 1, "cur": (X + W / 2.0, Y + H / 2.0), "msg": "",
          "entered": time.time()}

    def leave():
        r = rec(st["pos"])
        r["seconds"] = round(r["seconds"] + time.time() - st["entered"], 1)
        st["entered"] = time.time()
        C.saveJson(path, data)

    def onMouse(event, x, y, flags, _param):
        if y >= int(H * SCALE):
            return
        fx, fy = x / SCALE + X, y / SCALE + Y
        st["cur"] = (fx, fy)
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        if st["off"] or st["play"]:
            st["off"], st["play"] = 0, False
            st["msg"] = "back on the frame -- click again to mark"
            return
        r = rec(st["pos"])
        near = [k for k, (px, py, _s) in enumerate(r["points"])
                if (px - fx) ** 2 + (py - fy) ** 2 <= 36.0]
        if near:
            r["points"].pop(near[0])
            st["msg"] = "removed"
        else:
            sure = int(event == cv2.EVENT_LBUTTONDOWN
                       and not flags & cv2.EVENT_FLAG_SHIFTKEY)
            sx, sy = snap(frames.get(items[st["pos"]]["frame"]), fx, fy)
            r["points"].append([round(sx, 2), round(sy, 2), sure])
            st["msg"] = "droplet" if sure else "unsure"
        C.saveJson(path, data)

    def mainView():
        it, r = items[st["pos"]], rec(st["pos"])
        g = frames.get(it["frame"] + st["off"])
        crop = contrast(g[Y:Y + H, X:X + W], st["mode"])
        img = cv2.resize(crop, (int(W * SCALE), int(H * SCALE)),
                         interpolation = cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not st["hide"]:
            for px, py, sure in r["points"]:
                p = (int(round((px - X) * SCALE)), int(round((py - Y) * SCALE)))
                cv2.circle(img, p, 9, SURE if sure else UNSURE, 1)
                if not sure:
                    cv2.putText(img, "?", (p[0] + 10, p[1] - 6), FONT, 0.45, UNSURE, 1)
        if st["off"] or st["play"]:
            cv2.rectangle(img, (1, 1), (img.shape[1] - 2, img.shape[0] - 2),
                          (0, 140, 255), 3)
        bar = np.zeros((BAR, img.shape[1], 3), np.uint8)
        nSure = sum(1 for p in r["points"] if p[2])
        where = ("playing -- clicks return to the frame" if st["play"] else
                 "offset %+d -- s to return" % st["off"] if st["off"] else
                 "labelling this frame")
        cv2.putText(bar, "%d / %d   %s   droplets %d  unsure %d   contrast %s%s"
                    % (st["pos"] + 1, len(items), where, nSure, len(r["points"]) - nSure,
                       CONTRASTS[st["mode"]], "   [done]" if r["done"] else ""),
                    (8, 20), FONT, 0.5, (230, 230, 230), 1)
        if st["msg"]:
            (tw, _th), _ = cv2.getTextSize(st["msg"], FONT, 0.5, 1)
            cv2.putText(bar, st["msg"], (img.shape[1] - tw - 10, 20), FONT, 0.5,
                        (0, 170, 255), 1)
        return np.vstack([img, bar])

    def zoomView():
        it, r = items[st["pos"]], rec(st["pos"])
        g = frames.get(it["frame"] + st["off"])
        marks = [] if st["hide"] else [(px, py, SURE if s else UNSURE, 4)
                                       for px, py, s in r["points"]]
        big = magnifier(g, st["cur"][0], st["cur"][1], st["mode"], ZOOMS[st["zoom"]], marks)
        canvas = np.full((MAG_OUT + 30 + 22 * len(HELP), 470, 3), 24, np.uint8)
        canvas[:big.shape[0], :big.shape[1]] = big
        for k, line in enumerate(HELP):
            cv2.putText(canvas, line, (10, big.shape[0] + 30 + 22 * k), FONT, 0.47,
                        (215, 215, 215), 1)
        return canvas

    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("zoom", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("label", 0, 0)
    cv2.moveWindow("zoom", int(W * SCALE) + 14, 0)
    cv2.setMouseCallback("label", onMouse)
    while True:
        now = time.time()
        if st["play"] and now - st["tick"] >= 0.06:
            st["tick"] = now
            if abs(st["off"] + st["dir"]) > CONTEXT:
                st["dir"] = -st["dir"]
            st["off"] += st["dir"]
        cv2.imshow("label", mainView())
        cv2.imshow("zoom", zoomView())
        k = cv2.waitKey(15) & 0xFF
        if cv2.getWindowProperty("label", cv2.WND_PROP_VISIBLE) < 1:
            leave()
            break
        if k == 255:
            continue
        if 65 <= k <= 90:
            k += 32                     # caps lock on
        st["msg"] = ""
        if k in (ord("q"), 27):
            leave()
            break
        elif k in (32, 13):
            rec(st["pos"])["done"] = True
            leave()
            if st["pos"] + 1 < len(items):
                st["pos"] += 1
                st["off"], st["play"] = 0, False
            else:
                st["msg"] = "all %d screens labelled -- q to quit" % len(items)
        elif k in (8, ord("w")):
            leave()
            st["pos"] = max(0, st["pos"] - 1)
            st["off"], st["play"] = 0, False
        elif k in (ord("d"), ord("a"), ord("["), ord("]")):
            step = {ord("a"): -1, ord("d"): 1, ord("["): -5, ord("]"): 5}[k]
            st["off"] = int(np.clip(st["off"] + step, -CONTEXT, CONTEXT))
            st["play"] = False
        elif k == ord("s"):
            st["off"], st["play"] = 0, False
        elif k == ord("f"):
            st["play"] = not st["play"]
            if not st["play"]:
                st["off"] = 0
        elif k == ord("v"):
            st["mode"] = (st["mode"] + 1) % len(CONTRASTS)
        elif k == ord("h"):
            st["hide"] = not st["hide"]
        elif k == ord("z"):
            r = rec(st["pos"])
            if r["points"]:
                r["points"].pop()
                C.saveJson(path, data)
                st["msg"] = "undone"
        elif k == ord("c"):
            st["zoom"] = (st["zoom"] + 1) % len(ZOOMS)
    cv2.destroyAllWindows()
    done = sum(1 for k in range(len(items)) if rec(k)["done"])
    spent = sum(rec(k)["seconds"] for k in range(len(items)))
    print("%d of %d screens labelled, %.0f min so far -> %s"
          % (done, len(items), spent / 60.0, path))


def review(name):
    todo = C.loadJson(todoPath(name))
    if not todo or not todo.get("items"):
        sys.exit("nothing to review yet -- the scorer writes the disagreement spots to "
                 "%s once this unit has been labelled and replayed" % todoPath(name))
    items = todo["items"]
    path = reviewPath(name)
    data = C.loadJson(path, {"name": name, "answers": {}})
    X, Y, W, H = fixtureRoi()
    frames = Frames()
    pos = next((k for k, it in enumerate(items) if it["rid"] not in data["answers"]), None)
    if pos is None:
        print("all %d review items already answered" % len(items))
        return
    st = {"pos": pos, "off": 0, "dir": 1, "tick": 0.0, "mode": 1, "play": True,
          "msg": "", "zoom": 1}
    help2 = ("y              a droplet is here",
             "n              no droplet here",
             "u              can't tell",
             "w / backspace  previous item",
             "space          pause / play",
             "v              contrast     c  zoom",
             "q / esc        save and quit",
             "",
             "Judge only what is inside the red cross-hair",
             "ring. It plays -12..+12 frames around the frame.")
    cv2.namedWindow("review", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("zoom", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("review", 0, 0)
    cv2.moveWindow("zoom", int(W * SCALE) + 14, 0)
    while True:
        it = items[st["pos"]]
        now = time.time()
        if st["play"] and now - st["tick"] >= 0.07:
            st["tick"] = now
            if abs(st["off"] + st["dir"]) > CONTEXT:
                st["dir"] = -st["dir"]
            st["off"] += st["dir"]
        g0 = frames.get(it["frame"])
        img = cv2.resize(contrast(g0[Y:Y + H, X:X + W], st["mode"]),
                         (int(W * SCALE), int(H * SCALE)), interpolation = cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        p = (int(round((it["x"] - X) * SCALE)), int(round((it["y"] - Y) * SCALE)))
        cv2.rectangle(img, (p[0] - 24, p[1] - 24), (p[0] + 24, p[1] + 24), (0, 0, 255), 1)
        bar = np.zeros((BAR, img.shape[1], 3), np.uint8)
        prev = data["answers"].get(it["rid"], {}).get("a")
        cv2.putText(bar, "item %d / %d%s   y droplet   n none   u unsure   %s"
                    % (st["pos"] + 1, len(items),
                       "   answered: %s" % prev if prev else "", st["msg"]),
                    (8, 20), FONT, 0.5, (230, 230, 230), 1)
        cv2.imshow("review", np.vstack([img, bar]))
        g = frames.get(it["frame"] + st["off"])
        zoom = ZOOMS[st["zoom"]]
        big = magnifier(g, it["x"], it["y"], st["mode"], zoom,
                        [(it["x"], it["y"], (0, 0, 255), C.MATCH_PX)])
        canvas = np.full((MAG_OUT + 30 + 22 * len(help2), 470, 3), 24, np.uint8)
        canvas[:big.shape[0], :big.shape[1]] = big
        cv2.putText(canvas, "offset %+d" % st["off"], (10, big.shape[0] - 10), FONT, 0.5,
                    (0, 170, 255), 1)
        for k, line in enumerate(help2):
            cv2.putText(canvas, line, (10, big.shape[0] + 30 + 22 * k), FONT, 0.47,
                        (215, 215, 215), 1)
        cv2.imshow("zoom", canvas)
        k = cv2.waitKey(15) & 0xFF
        if cv2.getWindowProperty("review", cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == 255:
            continue
        if 65 <= k <= 90:
            k += 32                     # caps lock on
        st["msg"] = ""
        if k in (ord("q"), 27):
            break
        elif k in (ord("y"), ord("n"), ord("u")):
            data["answers"][it["rid"]] = {"pos": it["pos"], "x": it["x"], "y": it["y"],
                                          "a": chr(k)}
            C.saveJson(path, data)
            if st["pos"] + 1 < len(items):
                st["pos"] += 1
                st["off"], st["play"] = 0, True
            else:
                st["msg"] = "all done -- q to quit"
        elif k in (8, ord("w")):
            st["pos"] = max(0, st["pos"] - 1)
            st["off"], st["play"] = 0, True
        elif k == 32:
            st["play"] = not st["play"]
            if not st["play"]:
                st["off"] = 0
        elif k == ord("v"):
            st["mode"] = (st["mode"] + 1) % len(CONTRASTS)
        elif k == ord("c"):
            st["zoom"] = (st["zoom"] + 1) % len(ZOOMS)
    cv2.destroyAllWindows()
    print("%d of %d review items answered -> %s"
          % (sum(1 for it in items if it["rid"] in data["answers"]), len(items), path))


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices = ("sample", "label", "review"))
    ap.add_argument("recording")
    ap.add_argument("--name", default = "owner")
    ap.add_argument("--force", action = "store_true")
    a = ap.parse_args()
    use(a.recording)
    if a.mode == "sample":
        makeSample(a.force)
    elif a.mode == "label":
        label(a.name)
    else:
        review(a.name)


if __name__ == "__main__":
    main()
