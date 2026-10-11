"""NPU call latency by batch size, on the board (run from /root/millikan-edge).

One model per batch size (results/int8/model_int8[_bN].rknn), NHWC input, core 0,
50 warm-up calls then 1,000 timed.  -> results/board/bench_npu_batch.json
"""
import time, json, numpy as np
from rknnlite.api import RKNNLite
out = {}
rng = np.random.default_rng(0)
for B, path in ((1, "results/int8/model_int8.rknn"), (8, "results/int8/model_int8_b8.rknn"),
                (16, "results/int8/model_int8_b16.rknn"), (32, "results/int8/model_int8_b32.rknn")):
    rk = RKNNLite(verbose=False)
    assert rk.load_rknn(path) == 0
    assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0
    x = (rng.standard_normal((B, 24, 24, 1)) * 0.5).astype(np.float32)        # NHWC
    y = rk.inference(inputs=[x], data_format="nhwc")[0]
    for _ in range(50): rk.inference(inputs=[x], data_format="nhwc")
    t = []
    for _ in range(1000):
        t0 = time.perf_counter(); rk.inference(inputs=[x], data_format="nhwc"); t.append(time.perf_counter() - t0)
    a = np.array(t) * 1000
    out["batch%d" % B] = {"output_shape": list(np.shape(y)), "call_ms_median": round(float(np.median(a)), 4),
                          "per_crop_ms": round(float(np.median(a)) / B, 4)}
    rk.release()
print(json.dumps(out))
