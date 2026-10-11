"""The patch classifier as a detector gate, installed from outside cv/.

    gate = Gate("results/fp32/model.onnx")
    install(P, gate, "after")      # or "replace"

Two ways in, both fixed before the test set was scored (2026-10-10):

  after     detectDroplets runs untouched; every blob it keeps is then judged by
            the classifier.  Can only remove detections.
  replace   the area and circularity gates are switched off for the call, so
            every contour the threshold produced reaches the classifier, which
            then does their job.  The position masks (grid lines, panel margin,
            static marks, frame edge) still apply -- the classifier was never
            trained to replace them.

THRESHOLD 0.5: training drew droplet and noise equally, so 0.5 is the class
boundary the model learned.  On validation it passes 97.2% of droplets and 2.2% of
the phantom tracks; the max-F1 threshold (0.80) would cost 6 points of droplet
recall, and a droplet missed mid-fall is a broken measurement.

The crop is exactly what training saw: the 24x24 window centred on the blob's
rounded centroid, frame minus background, divided by 32.
"""
import cv2
import numpy as np

SIDE = 24
HALF = SIDE // 2
SCALE = 32.0
THRESHOLD = 0.5
REPLACE_AREA_MIN = 1.0          # "replace": any contour with an area reaches the model


class Gate(object):
    def __init__(self, onnx_path, threshold = THRESHOLD):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, opts,
                                         providers = ["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        # model_dynamic.onnx (same weights) takes every crop of a frame in one call;
        # model.onnx has the fixed batch of 1 the NPU conversion starts from.
        self.batched = not isinstance(self.sess.get_inputs()[0].shape[0], int)
        self.threshold = threshold
        self.calls = 0

    def crops(self, gray, background, centres):
        g = cv2.copyMakeBorder(gray, HALF, HALF, HALF, HALF, cv2.BORDER_CONSTANT, value = 0)
        b = cv2.copyMakeBorder(np.asarray(background, np.uint8), HALF, HALF, HALF, HALF,
                               cv2.BORDER_CONSTANT, value = 0)
        out = np.empty((len(centres), 1, SIDE, SIDE), np.float32)
        for k, c in enumerate(centres):
            ix, iy = int(round(c[0])), int(round(c[1]))
            out[k, 0] = (g[iy:iy + SIDE, ix:ix + SIDE].astype(np.float32)
                         - b[iy:iy + SIDE, ix:ix + SIDE].astype(np.float32)) / SCALE
        return out

    def probs(self, gray, background, centres):
        if not len(centres):
            return np.zeros(0, np.float32)
        return self.infer(self.crops(gray, background, centres))

    def infer(self, x):
        """Probabilities for crops already cut: x is (n, 1, 24, 24) float32."""
        if self.batched:
            logit = self.sess.run(None, {self.name: x})[0].ravel().astype(np.float32)
        else:
            logit = np.array([self.sess.run(None, {self.name: x[k:k + 1]})[0].ravel()[0]
                              for k in range(len(x))], np.float32)
        self.calls += len(x)
        return 1.0 / (1.0 + np.exp(-logit))


class NpuGate(Gate):
    """The INT8 model on the RK3588's NPU (RKNN-Toolkit-Lite2), same crops, same 0.5.

    Only exists on the board.  `core` picks one of the three NPU cores; the model is
    15k parameters, so one core is plenty and the others stay free.
    """
    def __init__(self, rknn_path, threshold = THRESHOLD, core = None):
        from rknnlite.api import RKNNLite
        self.rk = RKNNLite(verbose = False)
        if self.rk.load_rknn(rknn_path) != 0:
            raise RuntimeError("load_rknn failed: %s" % rknn_path)
        mask = RKNNLite.NPU_CORE_0 if core is None else core
        if self.rk.init_runtime(core_mask = mask) != 0:
            raise RuntimeError("init_runtime failed")
        self.threshold = threshold
        self.calls = 0

    def infer(self, x):
        logit = np.array([self.rk.inference(inputs = [x[k:k + 1]], data_format = "nchw")[0]
                          .ravel()[0] for k in range(len(x))], np.float32)
        self.calls += len(x)
        return 1.0 / (1.0 + np.exp(-logit))


class BatchedNpuGate(Gate):
    """INT8 on the NPU, a frame's crops packed into as few calls as possible.

    Measured on the board 2026-10-11 (NHWC input, NPU core 0), one call costs
    0.24 ms for 1 crop, 0.58 ms for 8, 1.05 ms for 16 and 1.76 ms for 32 -- the call
    overhead dominates a 15k-parameter model, so per crop that is 0.240 / 0.073 /
    0.066 / 0.055 ms.  An RKNN model's batch is fixed when it is built, so one model
    per size is kept and each chunk goes to the smallest that holds it, padded with
    zeros.  Input is handed over as NHWC: given NCHW the runtime converts it on
    every call, which cost a quarter of the time at batch 1.
    """
    def __init__(self, paths, threshold = THRESHOLD, core = None):
        from rknnlite.api import RKNNLite
        self.models = {}
        for b, path in sorted(paths.items()):
            rk = RKNNLite(verbose = False)
            if rk.load_rknn(path) != 0:
                raise RuntimeError("load_rknn failed: %s" % path)
            if rk.init_runtime(core_mask = RKNNLite.NPU_CORE_0 if core is None
                               else core) != 0:
                raise RuntimeError("init_runtime failed: %s" % path)
            self.models[b] = rk
        self.sizes = sorted(self.models)
        self.threshold = threshold
        self.calls = 0
        self.npuCalls = 0

    def infer(self, x):
        x = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
        out, k = [], 0
        while k < len(x):
            left = len(x) - k
            b = next((s for s in self.sizes if s >= left), self.sizes[-1])
            chunk = x[k:k + b]
            if len(chunk) < b:
                chunk = np.concatenate([chunk, np.zeros((b - len(chunk),) + chunk.shape[1:],
                                                        np.float32)])
            y = self.models[b].inference(inputs = [chunk], data_format = "nhwc")[0].ravel()
            out.append(y[:min(b, left)])
            k += b
            self.npuCalls += 1
        self.calls += len(x)
        return 1.0 / (1.0 + np.exp(-np.concatenate(out).astype(np.float32)))


def install(P, gate, mode):
    """Wrap P.detectDroplets so the classifier judges what it keeps."""
    if mode not in ("after", "replace"):
        raise ValueError(mode)
    orig = P.detectDroplets

    def detectDroplets(gray, background, s, *a, **k):
        if mode == "replace":
            saved = (P.CIRCULARITY_MIN, s["areaMin"], s["areaMax"])
            P.CIRCULARITY_MIN, s["areaMin"], s["areaMax"] = 0.0, REPLACE_AREA_MIN, 1e12
            try:
                out = orig(gray, background, s, *a, **k)
            finally:
                P.CIRCULARITY_MIN, s["areaMin"], s["areaMax"] = saved
        else:
            out = orig(gray, background, s, *a, **k)
        centres = out[0]
        p = gate.probs(gray, background, centres)
        kept = [c for c, q in zip(centres, p) if q >= gate.threshold]
        return (kept,) + tuple(out[1:])

    P.detectDroplets = detectDroplets
    return orig
