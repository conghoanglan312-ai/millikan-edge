# Live on the board, on units held back for it (2026-10-27)

The board (RK3588S, INT8 classifier on the NPU after the gates) ran the whole session
live: the camera dongle on the board, its window on the laptop over USB-C
(`edge/live_viewer.py --record`). Units 1-1 and 3-3 are reserved: never recorded for
training, validation or any test before this (HANDOVER §4).

| unit | board session | legal falls | window, updates a second | camera delivered | pipeline per frame |
| :-- | :-- | --: | --: | --: | --: |
| 1-1 | `20261027_103523` | 5 | 21.5 | 21.5 of 25 fps | 16.8 ms |
| 3-3 | `20261027_105012` | 6 | 21.3 | 21.3 of 25 fps | 20.3 ms |

The camera, not the pipeline, set the pace: the dongle's raw mode delivers about 22
frames a second, and each frame took the board 17–20 ms.

## Fall times against an independent timing

The operator filmed the laptop's screen with a phone and, blind to the board's
numbers, timed each fall from the video's clock: the droplet crossing the top and the
bottom line. Falls are matched by the session clock.

| unit | fall | phone video (s) | board (s) | board − phone (s) |
| :-- | --: | --: | --: | --: |
| 1-1 | 1 | 19.58 | 18.93 | −0.65 |
| 1-1 | 2 | 19.58 | 19.14 | −0.44 |
| 1-1 | 3 | 19.58 | 18.71 | −0.87 |
| 1-1 | 4 | 18.77 | 18.76 | −0.01 |
| 1-1 | 5 | 19.87 | 19.17 | −0.70 |
| 3-3 | 1 | 33.56 | 33.91 | +0.35 |
| 3-3 | 2 | 34.95 | 34.51 | −0.44 |
| 3-3 | 3 | 33.57 | 33.26 | −0.31 |
| 3-3 | 4 | 34.11 | 33.89 | −0.22 |
| 3-3 | 5 | 33.85 | 33.40 | −0.45 |

- Every fall agrees within 0.87 s; on average 0.53 s (2.7%) on unit 1-1's 19 s falls
  and 0.35 s (1.0%) on unit 3-3's 34 s falls. The board's time is the shorter in 9 of
  the 10.
- Unit 3-3's sixth legal fall (38.67 s) is a second droplet, not timed by the operator.

## Charge, one droplet per unit

The parent's formula on the board's falls (`charges_*.csv`): unit 1-1, 1.486 × 10⁻¹⁹ C
from 5 falls at 182 V (spread 1.7%, −7.2% from e); unit 3-3, 1.430 × 10⁻¹⁹ C from 5
falls at 152 V (spread 2.3%, −10.7%).

## Files

| file | what |
| :-- | :-- |
| `falls_1-1.csv`, `falls_3-3.csv` | every traverse the board measured, legal or not, as written by the session |
| `charges_1-1.csv`, `charges_3-3.csv` | the charge per fall and per droplet, as written by the session |
| `demo_fall_1-1.gif` | 10 s of unit 1-1's third fall at 2× speed, from the board's screen recording; to keep the file small, pixels that barely changed between frames are held from the previous frame |
