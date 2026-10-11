"""Batched INT8 models for the NPU: one ONNX export per batch size, then RKNN.

Runs in WSL.  RKNN-Toolkit2's rknn_batch_size option produced models that answer
wrongly (2026-10-11: first crop logit -2.23 instead of 3.99, any layout) -- it
replicates a batch-1 graph and the flatten->linear tail does not survive it.  So the
batch is baked into the ONNX export instead, calibration files carry B crops each,
and the simulator is checked against FP32 before the board ever sees a model.
"""
import os, sys, glob, numpy as np, torch
sys.path.insert(0, "/mnt/c/Users/LAN/millikan-edge")
from edge import train as T
from rknn.api import RKNN
base = "/mnt/c/Users/LAN/millikan-edge"
net = T.model(); net.load_state_dict(torch.load(base + "/data/models/fp32_20261009_2255/model.pt")); net.eval()
calib = np.concatenate([np.load(p) for p in sorted(glob.glob(base + "/data/calib/c*.npy"))])   # (499,1,24,24)
for B in (8, 16, 32):
    onnx = base + "/data/models/model_b%d.onnx" % B
    torch.onnx.export(net, torch.zeros(B, 1, 24, 24), onnx, input_names=["patch"], output_names=["logit"], opset_version=12)
    folder = base + "/data/calib_b%d" % B; os.makedirs(folder, exist_ok=True)
    paths = []
    for k in range(len(calib) // B):
        p = folder + "/c%03d.npy" % k; np.save(p, calib[k * B:(k + 1) * B]); paths.append(p)
    open(folder + "/list.txt", "w").write("\n".join(paths) + "\n")
    r = RKNN(verbose=False)
    r.config(target_platform="rk3588", quantized_dtype="w8a8", quantized_algorithm="normal", quantized_method="channel")
    assert r.load_onnx(model=onnx) == 0
    assert r.build(do_quantization=True, dataset=folder + "/list.txt") == 0
    out = base + "/results/int8/model_int8_b%d.rknn" % B
    assert r.export_rknn(out) == 0
    # simulator check against the batch-1 FP32 ONNX on the first 2B calibration crops
    assert r.init_runtime() == 0
    y = r.inference(inputs=[calib[:B]], data_format="nchw")[0].ravel()
    import onnxruntime as ort
    ref = ort.InferenceSession(base + "/results/fp32/model_dynamic.onnx").run(None, {"patch": calib[:B]})[0].ravel()
    print("B", B, "size", os.path.getsize(out), "sim vs fp32 max |dlogit| %.3f" % float(np.abs(y - ref).max()))
    r.release()
