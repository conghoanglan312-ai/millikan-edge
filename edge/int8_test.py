"""The INT8 model's own score on the test set, without the board (RKNN simulator).

Runs in WSL:  ~/venvs/rknn/bin/python /mnt/c/Users/LAN/millikan-edge/edge/int8_test.py
    -> data/replays/filter_fp32.npz, data/replays/filter_int8.npz  (scored by score.py)

    ... edge/int8_test.py --onnx results/fp32_v2/model.onnx                           --calib data/calib_v2/list.txt --tag _v2     another model
    ... edge/int8_test.py --fixtures unit1-3_1015 unit3-2_1015
    -> data/replays/filter_{fp32,int8}_<fixture>.npz      the unseen units (score_unseen.py)

The classifier sits after the gates, so at a labelled frame it sees exactly the blobs
the classical detector kept there.  data/autolabel/bench_0908.npz holds those blobs
with their crops, from a replay of the current code; each model's decision filters
them, and edge/score.py scores the survivors like any other row.

"Offline filter", not a replay: in the pipeline the tracker's own state feeds back
into which faint blobs the detector keeps (keepNear / measuring), so a full replay
with the classifier can differ slightly.  Both models are filtered the same way here,
so FP32 vs INT8 is like for like; the FP32 replay row in the score shows how far the
shortcut is from the real thing.

INT8 comes from the simulator, built from the same ONNX with the same calibration
list as results/int8/model_int8.rknn.  On the 41,038 validation crops the simulator
and the real NPU both agreed with FP32 on 99.9% (results/int8, results/board).
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(REPO, "results", "fp32", "model.onnx")
CALIB = os.path.join(REPO, "data", "calib", "list.txt")
SCALE = 32.0


def blobs(fixture):
    """The blobs the classical detector kept at the labelled frames, and their crops."""
    with np.load(os.path.join(REPO, "data", "autolabel", fixture + ".npz")) as f:
        z = {k: f[k] for k in ("gray", "bg", "gate", "frame", "x", "y", "roi",
                               "confirmPoints")}
    kept = z["gate"] == 0
    diff = (z["gray"][kept].astype(np.float32) - z["bg"][kept].astype(np.float32)) / SCALE
    return z, kept, np.ascontiguousarray(diff[:, 4:28, 4:28][:, None])


def main():
    import onnxruntime as ort
    from rknn.api import RKNN
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default = ONNX)
    ap.add_argument("--calib", default = CALIB)
    ap.add_argument("--tag", default = "", help = "suffix of the output files")
    ap.add_argument("--fixtures", nargs = "+", default = ["bench_0908"],
                    help = "data/autolabel/<fixture>.npz; outputs for any fixture but "
                           "bench_0908 carry its name")
    a = ap.parse_args()
    onnx, calib = os.path.abspath(a.onnx), os.path.abspath(a.calib)
    sess = ort.InferenceSession(onnx, providers = ["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    rk = RKNN(verbose = False)
    rk.config(target_platform = "rk3588", quantized_dtype = "w8a8",
              quantized_algorithm = "normal", quantized_method = "channel")
    assert rk.load_onnx(model = onnx) == 0
    assert rk.build(do_quantization = True, dataset = calib) == 0
    assert rk.init_runtime() == 0
    out = os.path.join(REPO, "data", "replays")
    os.makedirs(out, exist_ok = True)
    for fixture in a.fixtures:
        z, kept, x = blobs(fixture)
        fp32 = np.array([sess.run(None, {name: x[k:k + 1]})[0].ravel()[0]
                         for k in range(len(x))])
        int8 = np.array([rk.inference(inputs = [x[k:k + 1]], data_format = "nchw")[0]
                         .ravel()[0] for k in range(len(x))])
        suffix = a.tag + ("" if fixture == "bench_0908" else "_" + fixture)
        for tag, logit in (("fp32", fp32), ("int8", int8)):
            keep = logit >= 0.0                              # p >= 0.5
            np.savez_compressed(
                os.path.join(out, "filter_%s%s.npz" % (tag, suffix)),
                frame = z["frame"][kept][keep], x = z["x"][kept][keep],
                y = z["y"][kept][keep], gate = np.zeros(int(keep.sum()), np.int8),
                sightings = np.zeros((0, 5), np.float32), roi = z["roi"],
                confirmPoints = z["confirmPoints"])
            print("%s %s: kept %d of %d blobs" % (fixture, tag, keep.sum(), len(keep)),
                  flush = True)
        agree = float(((fp32 >= 0) == (int8 >= 0)).mean())
        print("%s: same decision on %.2f%% of the %d blobs" % (fixture, 100 * agree, len(x)),
              flush = True)
    rk.release()


if __name__ == "__main__":
    main()
