"""The board's live window, on the laptop.  One command at the bench:

    python edge/live_viewer.py                      the camera on the board
    python edge/live_viewer.py --record             ...and save what you see as MP4
    python edge/live_viewer.py --source frames:/root/test/fixtures/spray_labelled_0907
                                                    a recording, for a rehearsal

Click a droplet to pick it, exactly as in the parent's window; q stops the session.
The board writes falls.csv and its report; the viewer then copies the session
folder to data/live/ on the laptop.

What it does: `adb reverse` so the board's 127.0.0.1:PORT reaches this laptop, a
listening socket, then edge/live_board.py started on the board through `adb shell`
(its log printed here).  Frames in, clicks and keys out -- see live_board.py.

--headless N: no window; run N seconds, save every 10th frame to data/live/check/,
and click at --click X,Y once, after --click-at seconds.  For testing without a
person at the keyboard.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = "Board - millikan-edge"


def adbPath():
    for p in (os.environ.get("ADB"),
              os.path.join(REPO, "data", "tools", "platform-tools", "adb.exe"),
              shutil.which("adb")):
        if p and os.path.exists(p):
            return p
    sys.exit("adb not found: set ADB, or put platform-tools in data/tools/")


def recvExact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("board closed the connection")
        buf += chunk
    return buf


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type = int, default = 8765)
    ap.add_argument("--source", default = "cam:0")
    ap.add_argument("--gate", choices = ("npu", "none"), default = "npu")
    ap.add_argument("--record", action = "store_true")
    ap.add_argument("--headless", type = float, default = 0)
    ap.add_argument("--click", default = None)
    ap.add_argument("--click-at", type = float, default = 10.0)
    a = ap.parse_args()

    adb = adbPath()
    subprocess.run([adb, "reverse", "tcp:%d" % a.port, "tcp:%d" % a.port], check = True)
    subprocess.run([adb, "shell", "date -s @%d >/dev/null" % int(time.time())])
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", a.port))
    srv.listen(1)
    cmd = ("cd /root/millikan-edge && MILLIKAN_PARENT=/root python3 -u edge/live_board.py "
           "--port %d --source %s --gate %s" % (a.port, a.source, a.gate))
    board = subprocess.Popen([adb, "shell", cmd], stdout = subprocess.PIPE,
                             stderr = subprocess.STDOUT, text = True, errors = "replace")
    log, session = [], {"dir": None}

    def relay():
        for line in board.stdout:
            line = line.rstrip()
            log.append(line)
            if line.startswith("session -> "):
                session["dir"] = line.split("-> ", 1)[1].strip()
            if not line.startswith(("I RKNN", "W RKNN")):
                print("[board] " + line, flush = True)
    threading.Thread(target = relay, daemon = True).start()

    srv.settimeout(60)
    conn, _ = srv.accept()
    conn.settimeout(None)
    print("board connected; click a droplet to pick it, q to stop", flush = True)
    send = lambda obj: conn.sendall((json.dumps(obj) + "\n").encode())

    if not a.headless:
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, lambda ev, x, y, fl, pa:
                             send({"click": [x, y]}) if ev == cv2.EVENT_LBUTTONDOWN else None)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(REPO, "data", "live"), exist_ok = True)
    writer, frames, t0, clicked = None, 0, time.time(), False
    arrivals = []
    check = os.path.join(REPO, "data", "live", "check")
    if a.headless:
        os.makedirs(check, exist_ok = True)
    try:
        while True:
            n = int.from_bytes(recvExact(conn, 4), "big")
            img = cv2.imdecode(np.frombuffer(recvExact(conn, n), np.uint8), cv2.IMREAD_COLOR)
            frames += 1
            arrivals.append(time.time())
            if a.record:
                if writer is None:
                    writer = cv2.VideoWriter(
                        os.path.join(REPO, "data", "live", "demo_%s.mp4" % stamp),
                        cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (img.shape[1], img.shape[0]))
                writer.write(img)
            if a.headless:
                el = time.time() - t0
                if frames % 10 == 0:
                    cv2.imwrite(os.path.join(check, "f%05d.jpg" % frames), img)
                if a.click and not clicked and el >= a.click_at:
                    x, y = (int(v) for v in a.click.split(","))
                    send({"click": [x, y]})
                    clicked = True
                    cv2.imwrite(os.path.join(check, "click_at_f%05d.jpg" % frames), img)
                    print("clicked (%d, %d) at frame %d" % (x, y, frames), flush = True)
                if el >= a.headless:
                    send({"key": "q"})
                    break
                continue
            cv2.imshow(WINDOW, img)
            k = cv2.waitKey(1) & 0xFF
            if k != 0xFF:
                send({"key": chr(k)})
                if k == ord("q"):
                    break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                send({"key": "q"})
                break
    except ConnectionError:
        print("the board ended the session (q, or a recording ran out)", flush = True)
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    try:
        board.wait(timeout = 120)
    except subprocess.TimeoutExpired:
        board.kill()
    print("%d frames shown in %.0f s" % (frames, time.time() - t0), flush = True)
    if len(arrivals) > 10:
        gaps = np.diff(np.array(arrivals[5:])) * 1000.0      # skip the start-up
        print("smoothness: %.1f updates a second; gap median %.0f ms, p90 %.0f ms, "
              "max %.0f ms" % (1000.0 / gaps.mean(), np.median(gaps),
                               np.percentile(gaps, 90), gaps.max()), flush = True)
    if session["dir"]:
        dest = os.path.join(REPO, "data", "live")
        # adb prints its progress to stderr, which PowerShell shows as a red error
        subprocess.run([adb, "pull", session["dir"], dest], capture_output = True)
        print("session copied to %s" % os.path.join(dest, os.path.basename(session["dir"])))


if __name__ == "__main__":
    main()
