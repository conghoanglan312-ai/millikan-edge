# INT8 quantization — what it costs (RKNN simulator, validation crops)

`results/fp32_v2/model.onnx` → `model_int8.rknn` (77317 bytes), RKNN-Toolkit2 2.3.2, rk3588, w8a8, per-channel, calibrated on 499 training crops (half droplet). Written by `edge/convert.py`.

| metric | FP32 (onnxruntime) | INT8 (simulator) |
|---|---|---|
| ap | 0.9697 | 0.9713 |
| auc | 0.9980 | 0.9981 |
| pass rate, droplet | 0.9514 | 0.9525 |
| pass rate, phantom track | 0.0218 | 0.0220 |
| pass rate, sweep blob | 0.0081 | 0.0040 |
| pass rate, gate reject | 0.0059 | 0.0060 |

Same decision at 0.5 on 99.94% of 41038 crops; mean |Δp| 0.0022, max 0.3065.
