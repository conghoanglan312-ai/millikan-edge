"""ONNX -> RKNN INT8 for the RK3588, and what quantization costs, on the PC simulator.

Runs in WSL (RKNN-Toolkit2 2.3.2 -- the board's runtime is 2.3.2 too):

    ~/venvs/rknn/bin/python /mnt/c/Users/LAN/millikan-edge/edge/convert.py

    -> results/int8/model_int8.rknn, QUANTIZATION.md, quantization.json

    ... edge/convert.py --onnx results/fp32_v2/model.onnx --out results/int8_v2                         --calib data/calib_v2      another model, its own folders

Calibration: CALIB crops drawn from the TRAINING set, half droplet and half noise,
the noise spread over every kind -- the activation ranges INT8 is fitted to must
cover both.  Validation and test crops never enter calibration.

Accuracy delta: every validation crop through the FP32 ONNX (onnxruntime) and the
INT8 model (RKNN simulator), compared on the same metrics train.py reports.  The
test set is still untouched; on the board, the INT8 model is scored through
edge/score.py like the FP32 one.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge import train as T                                          # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(REPO, "results", "fp32", "model.onnx")
OUT = os.path.join(REPO, "results", "int8")
CALIB = 500


def calibration(train, rng, folder):
    rows = list(rng.choice(np.where(train["y"] == 1)[0], CALIB // 2, replace = False))
    neg = np.where(train["y"] == 0)[0]
    kinds = np.unique(train["kind"][neg])
    for k in kinds:
        pool = neg[train["kind"][neg] == k]
        rows += list(rng.choice(pool, min(len(pool), CALIB // 2 // len(kinds)), replace = False))
    os.makedirs(folder, exist_ok = True)
    x = T.crop(train["x"][np.array(rows)], 4)
    paths = []
    for k, patch in enumerate(x):
        p = os.path.join(folder, "c%04d.npy" % k)
        np.save(p, patch[None, None].astype(np.float32))       # NCHW, as the ONNX input
        paths.append(p)
    lst = os.path.join(folder, "list.txt")
    with open(lst, "w") as f:
        f.write("\n".join(paths) + "\n")
    return lst, len(paths)


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def main():
    from rknn.api import RKNN
    import onnxruntime as ort
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default = ONNX)
    ap.add_argument("--out", default = OUT)
    ap.add_argument("--calib", default = os.path.join(REPO, "data", "calib"))
    a = ap.parse_args()
    ONNX_, OUT_ = os.path.abspath(a.onnx), os.path.abspath(a.out)
    rng = np.random.default_rng(20261010)
    train, val = T.load("train"), T.load("val")
    lst, n = calibration(train, rng, os.path.abspath(a.calib))
    os.makedirs(OUT_, exist_ok = True)
    rk = RKNN(verbose = False)
    rk.config(target_platform = "rk3588", quantized_dtype = "w8a8",
              quantized_algorithm = "normal", quantized_method = "channel")
    assert rk.load_onnx(model = ONNX_) == 0
    assert rk.build(do_quantization = True, dataset = lst) == 0
    out = os.path.join(OUT_, "model_int8.rknn")
    assert rk.export_rknn(out) == 0
    assert rk.init_runtime() == 0                       # the PC simulator

    x = T.crop(val["x"], 4)[:, None].astype(np.float32)
    sess = ort.InferenceSession(ONNX_, providers = ["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    fp32 = sigmoid(np.array([sess.run(None, {name: x[k:k + 1]})[0].ravel()[0]
                             for k in range(len(x))]))
    int8 = sigmoid(np.array([rk.inference(inputs = [x[k:k + 1]], data_format = "nchw")[0].ravel()[0]
                             for k in range(len(x))]))
    rk.release()

    y = val["y"]
    res = {"calibrationCrops": n, "valCrops": int(len(y)), "rknnBytes": os.path.getsize(out)}
    for tag, s in (("fp32", fp32), ("int8", int8)):
        ap, auc = T.ap_auc(s, y)
        p = s >= 0.5
        r = {"ap": ap, "auc": auc}
        for k, kname in enumerate(val["kinds"]):
            m = val["kind"] == k
            if m.any():
                r["pass rate, %s" % kname] = float(p[m].mean())
        res[tag] = r
    res["agreement@0.5"] = float(((fp32 >= 0.5) == (int8 >= 0.5)).mean())
    res["max |p_int8 - p_fp32|"] = float(np.abs(int8 - fp32).max())
    res["mean |p_int8 - p_fp32|"] = float(np.abs(int8 - fp32).mean())
    with open(os.path.join(OUT_, "quantization.json"), "w", encoding = "utf-8") as f:
        json.dump(res, f, indent = 1)
    keys = ["ap", "auc"] + ["pass rate, %s" % k for k in val["kinds"]]
    L = ["# INT8 quantization — what it costs (RKNN simulator, validation crops)", "",
         "`%s` → `model_int8.rknn` (%d bytes), RKNN-Toolkit2 2.3.2, "
         "rk3588, w8a8, per-channel, calibrated on %d training crops (half droplet). "
         "Written by `edge/convert.py`." % (os.path.relpath(ONNX_, REPO).replace(os.sep, "/"),
                                            res["rknnBytes"], n), "",
         "| metric | FP32 (onnxruntime) | INT8 (simulator) |", "|---|---|---|"]
    L += ["| %s | %.4f | %.4f |" % (k, res["fp32"][k], res["int8"][k]) for k in keys]
    L += ["", "Same decision at 0.5 on %.2f%% of %d crops; mean |Δp| %.4f, max %.4f." %
          (100 * res["agreement@0.5"], len(y), res["mean |p_int8 - p_fp32|"],
           res["max |p_int8 - p_fp32|"])]
    with open(os.path.join(OUT_, "QUANTIZATION.md"), "w", encoding = "utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L), flush = True)


if __name__ == "__main__":
    main()
