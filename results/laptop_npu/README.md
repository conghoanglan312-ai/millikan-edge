# The same classifier on a laptop NPU — Snapdragon X Elite, 2026-10-17

ASUS Vivobook S15, Snapdragon X Elite X1E78100 (Hexagon NPU), Windows 11 ARM64;
native ARM64 Python 3.12.10, ONNX Runtime 1.31.0 with Qualcomm's plugin execution
provider onnxruntime-qnn 2.7.0, HTP performance mode `burst`. Made by
`edge/laptop_npu.py`; raw JSON in `laptop_npu.json`. Only the model runs here — the
pipeline around it is the board's job.

INT8: onnxruntime's QNN QDQ recipe, uint8 activations, int8 per-channel weights,
calibrated on the 499 crops the RKNN model used (`data/calib`). CPU fallback is off
for the NPU sessions, so every node ran on the NPU.

## Accuracy, all 41,038 validation crops

| | FP32, CPU | INT8, laptop NPU | INT8, RK3588 NPU (`results/board`) |
| :-- | --: | --: | --: |
| same decision as FP32 at 0.5 | — | **99.93%** | 99.90% |
| AP | 0.9568 | 0.9583 | — |
| droplets through | 97.2% | 97.3% | 97.3% |
| phantom tracks through | 2.15% | 2.13% | 2.18% |

Batch 32 against batch 1 on the laptop NPU: same decision on 99.96%.

## Time per call (50 warm-up calls, then the median of 2,000)

| | laptop: per call | per crop | RK3588: per call | per crop |
| :-- | --: | --: | --: | --: |
| FP32, CPU, 1 thread | 0.129 ms | 0.129 ms | 0.224 ms | 0.224 ms |
| FP32, CPU, 4 threads | 0.088 ms | 0.088 ms | 0.117 ms | 0.117 ms |
| INT8, NPU, 1 crop | 0.058 ms | 0.058 ms | 0.193 ms (NHWC) | 0.193 ms |
| INT8, NPU, 32 crops | 0.120 ms | **0.0037 ms** | 1.448 ms | 0.045 ms |

- The laptop's NPU is ~12× the RK3588's per crop when batched, and on this NPU a
  single crop already beats the CPU — the call overhead that made the RK3588's NPU
  lose to its own CPU at batch 1 is ~3× smaller here.
- Quantization costs nothing on either NPU: both agree with FP32 on 99.9% of crops.
- FP32 on the CPU, batch 32 with 4 threads: 0.036 ms a crop.
