# Power: the board against the laptop, doing the same work

Both machines ran `edge/soak.py --minutes 10`: the live session with its window, on
the busiest recording (`spray_labelled_0907`, looped). Each was measured on its own
power input, so neither a charger's nor a power brick's losses are counted.

| | idle | live view | added by the live view | updates a second |
| :-- | --: | --: | --: | --: |
| **board** (RK3588S, INT8 classifier on the NPU) | **1.1 W** | **2.9 W** | **1.7 W** | 18.3 |
| laptop (Snapdragon X Elite, FP32 on the CPU, x64 Python under emulation) | 8–9 W | 15–17 W | ~7 W | 18.9 |

The board does the work on about a fifth of the laptop's power, and the live view adds
about a quarter as much.

## Board, 2026-10-29

- **Supply:** GW Instek GPP-6030, CH1 at 12.000 V, current limit 3 A; readback
  ±(0.3% + 10 mA), about ±0.12 W here. No monitor attached.
- **Run:** 12:58:09–13:08:10, the laptop's USB-C cable unplugged once the run had
  started; clocks at their maximum throughout, 38 → 45 °C (`board_soak_20261029.json`).
- **Reading the watts:** the supply's screen was filmed for the whole run. One frame
  every 2 s, read by Windows' OCR; `board_readings_20261029.csv` holds every reading and
  which were used. Readings under 0.5 W while the board was on are misreads (the OCR
  picking up the idle channel's 000.00) and are dropped.
  - Idle before the run: 36 readings, mean 1.17 W, median 1.03 W.
  - The run, from 20 s after it started (the laptop's cable out) to its end: 202
    readings, mean **2.87 W** (sd 0.86, so ±0.06 W on the mean), median 2.56 W.
  - Idle after: 33 readings, mean 1.16 W.
  - Cross-check: where a frame also gave the current, 12 V × amps agrees with the watts
    within 0.15 W in 109 of 122 frames; those frames alone give 2.81 W for the run and
    1.10 and 1.18 W idle.
- The single readings swing between about 1 and 5 W: the board works in bursts each
  frame and idles between them. Only the average is a result.
- A first run on 2026-10-27 measured the same way (18.3 updates a second) but its
  readings stayed in the supply's memory, which froze when copied out.

## Laptop, 2026-10-20

- **Battery only:** charger out, nothing else on its ports, screen at full brightness;
  Windows' own discharge reading every 5 s (`laptop_battery_20261020.csv`).
- Idle 8.1 W by the battery's rate, 9.3 W by its falling capacity; the run 14.6 W by
  rate (after the first 90 s), 16.6 W by capacity. The two readings differ by ~2 W,
  hence the ranges. 18.9 updates a second (`laptop_soak_20261020.json`).
- An earlier run with an iPad charging from the laptop read 26 W idle and was
  discarded.
