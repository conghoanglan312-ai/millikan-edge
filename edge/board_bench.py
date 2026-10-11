"""The benchmark table's numbers, measured on the RK3588 board itself.

    python3 edge/board_bench.py model        per-call latency: FP32 on CPU vs INT8 on NPU
    python3 edge/board_bench.py accuracy     INT8 on the real NPU vs FP32, validation crops
    python3 edge/board_bench.py pipeline V   the whole pipeline over a fixture, V one of
                                             classical / cpu-fp32 / npu-int8

Each prints one JSON line; run each variant in its own process so peak memory
(ru_maxrss) belongs to that variant alone.

Paths: the board copy lives in /root/millikan-edge (cv/, edge/, results/) with
fixtures in /root/fixtures; set MILLIKAN_PARENT=/root so edge.common finds them.
"""
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge import gate as G                                           # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(REPO, "results", "fp32", "model.onnx")
RKNN = os.path.join(REPO, "results", "int8", "model_int8.rknn")
ONNX_DYN = os.path.join(REPO, "results", "fp32", "model_dynamic.onnx")
RKNN_BATCHED = {1: RKNN, **{b: os.path.join(REPO, "results", "int8", "model_int8_b%d.rknn" % b)
                            for b in (8, 16, 32)}}
FIXTURE = "spray_labelled_0907"
WARMUP, CALLS = 50, 2000


def peakMB():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def model():
    x = (np.random.default_rng(0).standard_normal((1, 1, 24, 24)) * 0.5).astype(np.float32)
    out = {}
    import onnxruntime as ort
    for threads in (1, 4):
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        sess = ort.InferenceSession(ONNX, o, providers = ["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        for _ in range(WARMUP):
            sess.run(None, {name: x})
        t = []
        for _ in range(CALLS):
            t0 = time.perf_counter()
            sess.run(None, {name: x})
            t.append(time.perf_counter() - t0)
        out["fp32_cpu_%dthread" % threads] = stats(t)
    from rknnlite.api import RKNNLite
    rk = RKNNLite(verbose = False)
    assert rk.load_rknn(RKNN) == 0
    assert rk.init_runtime(core_mask = RKNNLite.NPU_CORE_0) == 0
    for _ in range(WARMUP):
        rk.inference(inputs = [x], data_format = "nchw")
    t = []
    for _ in range(CALLS):
        t0 = time.perf_counter()
        rk.inference(inputs = [x], data_format = "nchw")
        t.append(time.perf_counter() - t0)
    out["int8_npu_core0"] = stats(t)
    rk.release()
    out["peakMB"] = peakMB()
    return out


def stats(t):
    a = np.array(t) * 1000.0
    return {"calls": len(a), "median_ms": round(float(np.median(a)), 4),
            "p90_ms": round(float(np.percentile(a, 90)), 4),
            "mean_ms": round(float(a.mean()), 4)}


def accuracy():
    z = np.load(os.path.join(REPO, "data", "val.npz"))
    diff = (z["gray"].astype(np.float32) - z["bg"].astype(np.float32)) / G.SCALE
    x = diff[:, 4:28, 4:28][:, None].astype(np.float32)
    y, kind, kinds = z["y"], z["kind"], [str(k) for k in z["kinds"]]
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX, providers = ["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    fp32 = np.array([sess.run(None, {name: x[k:k + 1]})[0].ravel()[0] for k in range(len(x))])
    from rknnlite.api import RKNNLite
    rk = RKNNLite(verbose = False)
    assert rk.load_rknn(RKNN) == 0
    assert rk.init_runtime(core_mask = RKNNLite.NPU_CORE_0) == 0
    int8 = np.array([rk.inference(inputs = [x[k:k + 1]], data_format = "nchw")[0].ravel()[0]
                     for k in range(len(x))])
    rk.release()
    p32, p8 = 1 / (1 + np.exp(-fp32)), 1 / (1 + np.exp(-int8))
    out = {"crops": int(len(y)), "agreement@0.5": float(((p32 >= .5) == (p8 >= .5)).mean()),
           "mean_abs_dp": float(np.abs(p8 - p32).mean())}
    for tag, p in (("fp32_cpu", p32), ("int8_npu", p8)):
        out[tag] = {"pass rate, %s" % n: float((p[kind == k] >= .5).mean())
                    for k, n in enumerate(kinds) if (kind == k).any()}
    return out


def batchedAgreement():
    """The batched NPU path against the one-crop-a-call one, on every validation crop.

    Fed in ragged groups of 29 so every chunk size and the zero padding are exercised.
    """
    z = np.load(os.path.join(REPO, "data", "val.npz"))
    diff = (z["gray"].astype(np.float32) - z["bg"].astype(np.float32)) / G.SCALE
    x = np.ascontiguousarray(diff[:, 4:28, 4:28][:, None])
    one, many = G.NpuGate(RKNN), G.BatchedNpuGate(RKNN_BATCHED)
    a = np.concatenate([one.infer(x[k:k + 29]) for k in range(0, len(x), 29)])
    b = np.concatenate([many.infer(x[k:k + 29]) for k in range(0, len(x), 29)])
    return {"crops": int(len(a)), "npuCallsBatched": many.npuCalls,
            "same decision": float(((a >= .5) == (b >= .5)).mean()),
            "max_abs_dp": float(np.abs(a - b).max())}


def pipeline(variant):
    from cv import pipeline as P
    from edge import common as C
    detectT = []
    gate = None
    if variant == "cpu-fp32":
        gate = G.Gate(ONNX)
        G.install(P, gate, "after")
    elif variant == "npu-int8":
        gate = G.NpuGate(RKNN)
        G.install(P, gate, "after")
    elif variant == "cpu-fp32-batched":
        gate = G.Gate(ONNX_DYN)
        G.install(P, gate, "after")
    elif variant == "npu-int8-batched":
        gate = G.BatchedNpuGate(RKNN_BATCHED)
        G.install(P, gate, "after")
    inner = P.detectDroplets

    def timed(*a, **k):
        t0 = time.perf_counter()
        out = inner(*a, **k)
        detectT.append(time.perf_counter() - t0)
        return out
    P.detectDroplets = timed
    P._beep = lambda: None
    P.writeLiveStatus = lambda **k: None
    folder = C.fixture(FIXTURE)
    n = len(np.load(os.path.join(folder, "times.npy")))
    t0 = time.perf_counter()
    falls = P.trackAll("frames:" + folder, show = False, saveVideo = False,
                       outDir = os.path.join("/root", "bench_runs", variant))
    wall = time.perf_counter() - t0
    d = np.array(detectT) * 1000.0
    return {"variant": variant, "fixture": FIXTURE, "frames": n,
            "framesDetected": int(len(d)), "wall_s": round(wall, 1),
            "wall_ms_per_frame": round(1000.0 * wall / n, 2),
            "fps_overall": round(n / wall, 1),
            "detect_ms_median": round(float(np.median(d)), 3),
            "detect_ms_mean": round(float(d.mean()), 3),
            "detect_ms_p90": round(float(np.percentile(d, 90)), 3),
            "gateCalls": gate.calls if gate else 0,
            "npuCalls": getattr(gate, "npuCalls", None),
            "falls": len(falls), "peakMB": peakMB()}


if __name__ == "__main__":
    what = sys.argv[1]
    res = {"model": model, "accuracy": accuracy, "batched": batchedAgreement}.get(what)
    out = res() if res else pipeline(sys.argv[2])
    print(json.dumps(out), flush = True)
