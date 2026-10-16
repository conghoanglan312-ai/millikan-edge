"""The live pipeline on the board, with its window on the laptop or on its own screen.

    python3 edge/live_board.py --port 8765 [--source cam:0] [--gate npu|none]
    PYTHONPATH=/opt/cvgui python3 edge/live_board.py --local [--source ...]

Started by edge/live_viewer.py over adb; connects to 127.0.0.1:PORT, which
`adb reverse` carries to the viewer on the laptop over the USB cable.

--local: the window opens on the board's own desktop (HDMI monitor, USB mouse), no
laptop; closing it ends the session as q does.  The board's pip OpenCV is headless,
so this needs one with a GUI first on the path: opencv-python 4.10.0.84 (Qt),
installed with `pip3 install --no-deps --target <dir>`, kept apart from the
system's.  edge/board_local.sh sets the board up for it: desktop launchers, no
laptop needed at any step.

The pipeline is unchanged.  Its window calls are taken over from outside cv/:

  imshow(img)          the picture the operator would have seen -- HUD, rings,
                       banners and all -- goes to the laptop as JPEG (newest wins)
  setMouseCallback     the pipeline's own click handler is kept, and laptop clicks
                       are fed to it as EVENT_LBUTTONDOWN at image coordinates, so
                       click-to-pick works exactly as at the parent's laptop
  waitKey              returns the key the laptop sent (q stops the session)
  namedWindow, destroyAllWindows    nothing: the board has no screen

Its own --publish stream was not enough: it is sent before the HUD is drawn (the
app drew its own), and its remote clicks need a real window to exist.

Wire format.  Board -> laptop: 4-byte big-endian length, then a JPEG.  Laptop ->
board: one JSON object per line, {"click": [x, y]} or {"key": "q"}.
"""
import argparse
import json
import os
import queue
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2                                                           # noqa: E402
import numpy as np                                                   # noqa: E402
from cv import pipeline as P                                         # noqa: E402
from edge import fast as F                                           # noqa: E402
from edge import gate as G                                           # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RKNN = {1: "model_int8.rknn", 8: "model_int8_b8.rknn", 16: "model_int8_b16.rknn",
        32: "model_int8_b32.rknn"}
JPEG_QUALITY = 80
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"   # on the board's image


class Screen(object):
    """Stands in for cv2 inside cv/pipeline.py: Chinese text drawn in a CJK font,
    everything else passed through -- the board's own screen (--local)."""

    def __init__(self):
        self.shown = []
        self.name = None

    def __getattr__(self, name):
        return getattr(cv2, name)

    def imshow(self, name, img):
        self.shown.append(time.perf_counter())
        self.name = name
        return cv2.imshow(name, img)

    def waitKey(self, delay = 0):
        """q, or closing the window with the mouse -- the board may have no keyboard."""
        key = cv2.waitKey(delay)
        if self.name and cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
            return ord("q")
        return key

    def smoothness(self):
        """Updates a second and the gaps between them -- the viewer's own line."""
        t = np.array(self.shown)
        if len(t) < 2:
            return "no frames shown"
        gap = np.diff(t) * 1000.0
        return ("%.1f updates a second, gap median %.0f ms, p90 %.0f ms, %.0f%% over "
                "150 ms" % ((len(t) - 1) / (t[-1] - t[0]), np.median(gap),
                            np.percentile(gap, 90), 100.0 * np.mean(gap > 150)))

    def putText(self, img, text, org, fontFace, fontScale, color, thickness = 1,
                *a, **k):
        """cv2.putText draws ? for every Chinese character -- the HUD's 平衡/提升
        button names among them.  The parent's own window had the same ?s; the browser
        app drew its own banner.  Here this window is the operator's only screen, so
        non-ASCII text is drawn with a CJK font instead, in a strip around the line."""
        if text.isascii():
            return cv2.putText(img, text, org, fontFace, fontScale, color, thickness, *a, **k)
        mask = self._glyphs(text, max(10, int(round(fontScale * 30))))
        if mask is None:
            return cv2.putText(img, text, org, fontFace, fontScale, color, thickness, *a, **k)
        # The mask's bottom sits a quarter of its height under the baseline, like a
        # descender; clip it to the image and blend the colour in by coverage.
        h, w = mask.shape
        x0, y0 = int(org[0]), int(org[1]) - int(h * 0.78)
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
        if xb <= xa or yb <= ya:
            return img
        m = mask[ya - y0:yb - y0, xa - x0:xb - x0, None]
        roi = img[ya:yb, xa:xb].astype(np.float32)
        img[ya:yb, xa:xb] = (roi * (1 - m) + np.array(color[:3], np.float32) * m).astype(np.uint8)
        return img

    def _glyphs(self, text, px, cache = {}):
        """Coverage mask of `text` at `px`, rendered once: the banner repeats itself
        frame after frame, and rendering it every time cost the view a quarter of its
        frame rate (12 -> 9 a second)."""
        key = (text, px)
        if key not in cache:
            try:
                from PIL import Image, ImageDraw, ImageFont
                font = ImageFont.truetype(CJK_FONT, px)
                left, top, right, bottom = font.getbbox(text)
                im = Image.new("L", (right + 2, int(px * 1.3)), 0)
                ImageDraw.Draw(im).text((0, 0), text, font = font, fill = 255)
                cache[key] = np.asarray(im, np.float32) / 255.0
            except Exception:
                cache[key] = None
            if len(cache) > 512:
                cache.clear()
        return cache.get(key)


class Window(Screen):
    """The laptop's window: imshow streams, clicks and keys come back."""

    def __init__(self, sock):
        Screen.__init__(self)
        self.sock = sock
        self.callback = None
        self.inbox = queue.Queue()
        self.latest = None
        self.wake = threading.Event()
        self.closed = False
        threading.Thread(target = self._send, daemon = True).start()
        threading.Thread(target = self._receive, daemon = True).start()

    def namedWindow(self, *a, **k):
        pass

    def destroyAllWindows(self, *a, **k):
        pass

    def setMouseCallback(self, name, callback, *a):
        self.callback = callback

    def imshow(self, name, img):
        self.latest = img
        self.wake.set()

    def waitKey(self, delay = 0):
        key = -1
        while True:
            try:
                msg = self.inbox.get_nowait()
            except queue.Empty:
                break
            if "click" in msg and self.callback is not None:
                x, y = msg["click"]
                self.callback(cv2.EVENT_LBUTTONDOWN, int(x), int(y), 0, None)
            elif "key" in msg and key == -1:
                key = ord(msg["key"][0])
        if self.closed and key == -1:
            key = ord("q")                   # the viewer went away: end cleanly
        return key

    def _send(self):
        try:
            while not self.closed:
                self.wake.wait(1.0)
                self.wake.clear()
                img, self.latest = self.latest, None
                if img is None:
                    continue
                ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                                     JPEG_QUALITY])
                if ok:
                    data = buf.tobytes()
                    self.sock.sendall(len(data).to_bytes(4, "big") + data)
        except OSError:
            self.closed = True

    def _receive(self):
        f = self.sock.makefile("r", encoding = "utf-8")
        try:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.inbox.put(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        self.closed = True


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type = int, default = 8765)
    ap.add_argument("--source", default = "cam:0")
    ap.add_argument("--gate", choices = ("npu", "none"), default = "npu")
    ap.add_argument("--out", default = "/root/live")
    ap.add_argument("--local", action = "store_true",
                    help = "window on the board's own screen, no laptop")
    a = ap.parse_args()

    sock = None
    if a.local:
        if "GUI:                           NONE" in cv2.getBuildInformation():
            sys.exit("this OpenCV has no windows -- run with PYTHONPATH=/opt/cvgui")
        P.cv2 = Screen()
    else:
        for _ in range(50):                  # the viewer may still be starting
            try:
                sock = socket.create_connection(("127.0.0.1", a.port), timeout = 5.0)
                break
            except OSError:
                time.sleep(0.2)
        if sock is None:
            sys.exit("no viewer on 127.0.0.1:%d -- start edge/live_viewer.py on the "
                     "laptop" % a.port)
        sock.settimeout(None)
        P.cv2 = Window(sock)
    if a.gate == "npu":
        gate = G.BatchedNpuGate({b: os.path.join(REPO, "results", "int8", f)
                                 for b, f in RKNN.items()})
        G.install(P, gate, "after")
        print("classifier: INT8 on the NPU, after the gates, threshold %.2f"
              % gate.threshold, flush = True)
    else:
        print("classifier: none (classical detector)", flush = True)
    P.writeLiveStatus = lambda **k: None
    F.install(P)                 # same answers, ~6 ms a frame less (edge/fast.py)
    # The pipeline shows every 2nd frame because on the parent's laptop imshow +
    # waitKey cost 14 ms of a 42 ms frame (SHOW_EVERY, pipeline.py).  Here "showing"
    # only hands the frame to the sender thread, so every frame goes: measured
    # 2026-10-12, every 2nd gave the operator ~8 updates a second, unevenly.
    P.SHOW_EVERY = 1
    outDir = os.path.join(a.out, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outDir, exist_ok = True)
    print("session -> %s" % outDir, flush = True)
    falls = P.trackAll(a.source, show = True, saveVideo = False, outDir = outDir)
    P.report(falls, outDir, a.source)
    if a.local:
        print("window: %s" % P.cv2.smoothness(), flush = True)
    else:
        try:
            sock.close()
        except OSError:
            pass
    os.sync()


if __name__ == "__main__":
    main()
