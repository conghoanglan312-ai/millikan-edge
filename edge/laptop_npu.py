"""The same classifier on the laptop's NPU (Snapdragon X Elite, Hexagon), measured the
way the board's was -- one more row beside the RK3588.

Runs in a native ARM64 Python on Windows; the laptop's usual Python is x64 under
emulation and cannot load Qualcomm's libraries:

    C:\\Users\\LAN\\tools\\py312arm64\\python.exe edge/laptop_npu.py
        (Python 3.12.10 ARM64 from the `pythonarm64` NuGet package, unzipped;
         pip install onnxruntime-qnn numpy onnx -> onnxruntime 1.31.0,
         onnxruntime-qnn 2.7.0, Qualcomm's plugin execution provider)

    -> results/laptop_npu/README.md, laptop_npu.json

Like for like with results/board:
  models      results/fp32/model.onnx (batch 1) and data/models/model_b32.onnx (the
              batch baked into the export, as for the RKNN models)
  INT8        QDQ from onnxruntime's QNN recipe, 8-bit activations and weights,
              calibrated on data/calib -- the 499 crops the RKNN model used
  agreement   all 41,038 validation crops, FP32 on the CPU against INT8 on the NPU:
              decision at 0.5, pass rate by kind, AP
  timing      50 warm-up calls, then 2,000; median and p90, per call and per crop
CPU fallback is switched off for the NPU sessions, so a node the NPU cannot take
fails the run instead of quietly running on the CPU.
"""
import glob
import json
import os
import platform
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(REPO, "results", "fp32", "model.onnx")
ONNX_B32 = os.path.join(REPO, "data", "models", "model_b32.onnx")
CALIB = os.path.join(REPO, "data", "calib")
VAL = os.path.join(REPO, "data", "dataset", "val.npz")
WORK = os.path.join(REPO, "data", "laptop_npu")
OUT = os.path.join(REPO, "results", "laptop_npu")
SCALE = 32.0
WARMUP, CALLS = 50, 2000


def apOf(score, y):
    """edge/train.py's AP, without importing torch."""
    order = np.argsort(-score)
    y = y[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(1, y.sum()))


def quantize(src, dst, batch):
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import (get_qnn_qdq_config,
                                                                  qnn_preprocess_model)
    crops = np.concatenate([np.load(p) for p in sorted(glob.glob(os.path.join(CALIB, "c*.npy")))])
    batches = [crops[k:k + batch] for k in range(0, len(crops) - batch + 1, batch)]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"patch": b.astype(np.float32)} for b in batches])

        def get_next(self):
            return next(self.it, None)

    import onnx
    from onnx import version_converter
    # the export is opset 12; a per-channel DequantizeLinear (its axis) needs 13+
    op = dst.replace(".onnx", "_op17.onnx")
    onnx.save(version_converter.convert_version(onnx.load(src), 17), op)
    pre = dst.replace(".onnx", "_pre.onnx")
    model = pre if qnn_preprocess_model(op, pre) else op
    cfg = get_qnn_qdq_config(model, Reader(), activation_type = QuantType.QUInt8,
                             weight_type = QuantType.QInt8, per_channel = True)
    quantize(model, dst, cfg)
    return {"calibrationCrops": int(sum(len(b) for b in batches)),
            "activations": "uint8, asymmetric", "weights": "int8, per channel"}


def sessions():
    import onnxruntime as ort
    import onnxruntime_qnn as qnn
    ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
    npu = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"
           and d.device.type == ort.OrtHardwareDeviceType.NPU]
    assert npu, "no Qualcomm NPU device"

    def onNpu(path):
        o = ort.SessionOptions()
        o.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        o.add_provider_for_devices(npu, {"backend_path": qnn.get_qnn_htp_path(),
                                         "htp_performance_mode": "burst"})
        return ort.InferenceSession(path, sess_options = o)

    def onCpu(path, threads):
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        return ort.InferenceSession(path, o, providers = ["CPUExecutionProvider"])
    return onNpu, onCpu, ort.__version__, qnn.__version__


def timed(sess, x):
    name = sess.get_inputs()[0].name
    for _ in range(WARMUP):
        sess.run(None, {name: x})
    t = []
    for _ in range(CALLS):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        t.append(time.perf_counter() - t0)
    a = np.array(t) * 1000.0
    return {"calls": CALLS, "median_ms": round(float(np.median(a)), 4),
            "p90_ms": round(float(np.percentile(a, 90)), 4),
            "per_crop_ms": round(float(np.median(a)) / len(x), 4)}


def logits(sess, x, batch):
    name = sess.get_inputs()[0].name
    out = []
    for k in range(0, len(x), batch):
        b = x[k:k + batch]
        n = len(b)
        if n < batch:
            b = np.concatenate([b, np.zeros((batch - n,) + b.shape[1:], np.float32)])
        out.append(sess.run(None, {name: b})[0].ravel()[:n])
    return np.concatenate(out)


def main():
    assert platform.machine() == "ARM64", "run in the ARM64 Python (see the header)"
    os.makedirs(WORK, exist_ok = True)
    os.makedirs(OUT, exist_ok = True)
    q1, q32 = os.path.join(WORK, "model_qdq.onnx"), os.path.join(WORK, "model_b32_qdq.onnx")
    recipe = quantize(ONNX, q1, 1)
    quantize(ONNX_B32, q32, 32)
    onNpu, onCpu, ortVersion, qnnVersion = sessions()

    z = np.load(VAL)
    x = ((z["gray"].astype(np.float32) - z["bg"].astype(np.float32)) / SCALE)[:, 4:28, 4:28]
    x = np.ascontiguousarray(x[:, None])
    y, kind, kinds = z["y"].astype(np.float32), z["kind"], [str(k) for k in z["kinds"]]
    fp32 = logits(onCpu(ONNX, 1), x, 1)
    npu1 = onNpu(q1)
    int8 = logits(npu1, x, 1)
    npu32 = onNpu(q32)
    int8b = logits(npu32, x, 32)
    p32, p8, p8b = [1 / (1 + np.exp(-v)) for v in (fp32, int8, int8b)]
    acc = {"crops": int(len(y)),
           "agreement@0.5": float(((p32 >= .5) == (p8 >= .5)).mean()),
           "batch32 vs batch1 agreement@0.5": float(((p8b >= .5) == (p8 >= .5)).mean()),
           "mean_abs_dp": float(np.abs(p8 - p32).mean())}
    for tag, p in (("fp32_cpu", p32), ("int8_npu", p8)):
        acc[tag] = {"ap": apOf(p, y)}
        acc[tag].update({"pass rate, %s" % n: float((p[kind == k] >= .5).mean())
                         for k, n in enumerate(kinds) if (kind == k).any()})

    one = x[:1]
    b32 = x[:32]
    t = {"fp32_cpu_1thread": timed(onCpu(ONNX, 1), one),
         "fp32_cpu_4threads": timed(onCpu(ONNX, 4), one),
         "fp32_cpu_batch32_4threads": timed(onCpu(ONNX_B32, 4), b32),
         "int8_npu_1crop": timed(npu1, one),
         "int8_npu_batch32": timed(npu32, b32)}
    res = {"machine": "ASUS Vivobook S15, Snapdragon X Elite X1E78100, Windows 11 ARM64",
           "python": sys.version.split()[0], "onnxruntime": ortVersion,
           "onnxruntime-qnn": qnnVersion, "htp_performance_mode": "burst",
           "quantization": recipe, "accuracy": acc, "timing": t}
    with open(os.path.join(OUT, "laptop_npu.json"), "w", encoding = "utf-8") as f:
        json.dump(res, f, indent = 1)
    print(json.dumps(res, indent = 1), flush = True)


if __name__ == "__main__":
    main()
