"""The FP32 model with a variable batch, for the CPU gate (one call per frame).

Same weights as results/fp32/model.onnx -- checked: identical logit.
"""
import sys, torch
sys.path.insert(0, "/mnt/c/Users/LAN/millikan-edge")
from edge import train as T
net = T.model(); net.load_state_dict(torch.load("/mnt/c/Users/LAN/millikan-edge/data/models/fp32_20261009_2255/model.pt")); net.eval()
torch.onnx.export(net, torch.zeros(1, 1, 24, 24), "/mnt/c/Users/LAN/millikan-edge/results/fp32/model_dynamic.onnx",
                  input_names=["patch"], output_names=["logit"], opset_version=12,
                  dynamic_axes={"patch": {0: "n"}, "logit": {0: "n"}})
import onnxruntime as ort, numpy as np
a = ort.InferenceSession("/mnt/c/Users/LAN/millikan-edge/results/fp32/model.onnx").run(None, {"patch": np.ones((1,1,24,24), np.float32)})[0]
b = ort.InferenceSession("/mnt/c/Users/LAN/millikan-edge/results/fp32/model_dynamic.onnx").run(None, {"patch": np.ones((5,1,24,24), np.float32)})[0]
print("same weights:", float(abs(a[0,0]-b[0,0])), b.shape)
