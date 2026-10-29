# millikan-edge — handover

Everything a fresh session needs to start work here. Written 2026-09-25, the day
the parent project was submitted, by the Claude Code session that did most of
this month's CV work.

**Updated 2026-09-26:** board ordered (§8.1); the build machine is now the
Vivobook under WSL2, not the RTX 4060 laptop (§8).

**Updated 2026-09-29:** the board arrived and passed (§8.1); the "before" numbers
are frozen (§3); §4's plan for auto-labels was wrong about where the data is and is
rewritten; **§11 is where the work stands and the decisions waiting on the owner.**

Read §6 and §11 before writing any code.

---

## 1. What this repo is

A **personal fork** of a four-person competition project, seeded with its full
history. The parent is the 12th National Undergraduate Physics Experiment
Competition, Topic 6 (AI + physics experiment): a real-time assistant for the
**Millikan oil-drop experiment**, submitted 2026-09-25 and not expected to change
again.

This fork exists for one purpose: **an Edge AI portfolio piece**, taking the
droplet detector to an RK3588 board and recording what it costs.

```
CV droplet detection  →  ONNX  →  RKNN  →  INT8  →  RK3588
```

Then measure: latency, FPS, memory, accuracy delta before and after quantization.

**The deliverable is the numbers, not a demo.** The benchmark table is the
artefact; "it works" is a claim.

### Credit

The history carries two authors: 122 commits by the repo owner, 42 by a teammate
(`Sky Alva`) whose work is the app and LLM side (`app_js/`, `llm/`). The CV
module — `cv/pipeline.py`, `cv/api.py`, `cv/source.py`, the panel reader, the
live tutor's rules — is the owner's. `git log --author` settles any question.

**Public since 2026-10-28**, the owner's decision, made when supervisors for the
graduation project were to be contacted; the parent was already public. The history
was scanned for keys before the switch: none (`.env.example` holds placeholders).

---

## 2. What is NOT in this clone, and where it is

**`test/` is gitignored in the parent and therefore absent here.** That is where
almost everything this fork needs actually lives:

Parent clone on the Vivobook: `C:\Users\LAN\Documents\GitHub\Physics-Experiment-Assistant\`
(from WSL2: `/mnt/c/Users/LAN/Documents/GitHub/Physics-Experiment-Assistant/`).

| what | where (in the parent clone) | why it matters |
| :-- | :-- | :-- |
| 126 recorded sessions, ~65 GB | `test/output/live_*/` | every frame this project has ever seen |
| fixtures | `test/fixtures/` | `bench_0908`, `one_droplet_slow`, the panel sweeps |
| the acceptance gate | `test/tools/check.py` | the parent's regression test |
| the labelling harness | `test/tools/detacc_label.py`, `detacc_score.py` | see §4 |
| the live-path check | `test/tools/live_check.py` | where the baseline number comes from |

Copy what you need in deliberately, and decide per item whether it belongs in
git. The 65 GB does not; a sampled, labelled subset does. From WSL2, read the
65 GB in place through `/mnt/c/`; do not copy it into the Linux filesystem (C:
had 46 GB free on 2026-09-26).

The labelled data itself **is** tracked, under
`cv/for_report/detection_accuracy/` — that came across with the clone.

---

## 3. The problem this fork should solve

Not a contrived portfolio exercise. **The classical detector produces phantoms,
and it is documented, measured, and unsolved.**

- `test/tools/live_check.py` fails its own acceptance bar: **3.11 detections a
  frame against a 0.87 baseline** on the worst-case fixture.
  *Correction 2026-10-10:* that fixture contains real droplets (field on, falling
  in one column), and `live_check` counts every detection, so 3.11 vs 0.87 is not a
  pure phantom count. The classifier removes 484 of its 1,465 blobs (3.11 → 2.08 a
  frame), and by eye those 484 are the noise (`results/fp32/PHANTOM_RATE.md`). The
  hand-labelled score is the one that separates phantoms from droplets.
- The square lands on noise. The whole of 2026-09-21/22 went into guarding
  *around* it — a minimum track age before anything may be picked, an area
  tolerance for measuring tracks, a cap on how many re-acquisitions a fall may
  be stitched from.
- **Two hand-written detector gates were tried and both failed the acceptance
  gate outright**: one testing the blob before the `0` crossing (2 traverses
  against a baseline of 46, all five reference falls lost), one testing a second
  after it (4 of 5 reference falls lost). Both were reverted. The source note at
  the reverted site says the real fix is a shape or point-spread test, which is
  precisely a small learned classifier.

That is the strongest thing this portfolio piece has: **a baseline that a
competent hand-tuned classical gate could not fix, measured and written down
before any model existed.** Problem → what I did → measured result, with the
problem independently dated.

**Protect those numbers.** Copy `live_check`'s output and
`cv/for_report/detection_accuracy/` into the fork's own results directory before
touching the pipeline, so the "before" column cannot drift.

*Done 2026-09-29* — `results/baseline/`, indexed in `results/README.md`. `live_check`
re-run: still 3.11 against 0.87. The labels stay where they are and are pinned by
hash instead of copied; the evaluation region and predictions are frozen in
`eval_bench_0908.npz`. `edge/score.py` reproduces §8.5 of the parent's report to the
decimal, and scores the **current** code's detector at **81.0% precision, 95.8%
recall, 1.35 false detections a frame** — the row a classifier has to beat.

---

## 4. The labelled set — counted 2026-09-25

In `cv/for_report/detection_accuracy/labels.json`:

- **165 screens** — 150 distinct frames drawn at a fixed seed from four phases of
  one recording, plus 15 unannounced repeats that measure the labeller's own
  consistency
- **1,863 marks, of which 1,267 are "sure"** and 596 "unsure"
- each mark is `[x, y, sure]` in full-frame pixels

`sample.json`, `review.json` (a blind second look at the 459 spots where the
marks and the tracker disagreed) and `control_answers.json` / `control_key.json`
(40 empty spots and 40 real droplets, shown blind) are alongside it. The method
is in `cv/for_report/HANDOVER.md` §7.7 and §8.5.

**The catch: all of it is from one session (`bench_0908`).**

So: patch count is fine, session diversity is not. A model trained and tested on
one session's frames will report an accuracy that means nothing — the background,
focus and noise are shared.

### Getting the rest without going back to the lab

The lab is still **not** needed, but the plan written here on 2026-09-25 was wrong
about where the data is (checked 2026-09-29):

- **The 126 sessions cannot be used.** Each holds only `annotated_multi.mp4` — the
  ROI with the overlay burned in, so every tracked droplet carries a ring and a crop
  of it carries its own label — plus `falls.csv` (no frame spans) and `rejects.txt`
  (counts per gate, not positions). `REJECT_POINTS` lives one frame in memory and is
  never written out.
- **Raw frames exist only in `test/fixtures`**, dated from the frame files:

| fixture | captured | frames | droplets? | role |
| :-- | :-- | --: | :-- | :-- |
| `bench_0908` | 09-08 11:51 | 56,543 | yes | **test** — hand-labelled |
| `live_20260903_noisy` | 09-03 | 471 | yes | **eval** — `live_check`'s phantom rate; keep out of training |
| `one_droplet_slow` | 09-05 15:03 | 27,585 | yes | pool (saved pre-cropped; ROI from `roi.txt`, as `check.py` does) |
| `one_droplet_full` | 09-05 15:30 | 10,026 | a few | pool |
| `spray_labelled_0907` | 09-07 11:43 | 1,592 | yes | pool |
| `zoom_labelled_0907` | 09-07 12:28 | 5,368 | yes | pool |
| `voltage_sweep_0907` | 09-07 17:08 | 1,575 | **none** | pool |
| `panel_sweep_0908` | 09-08 17:33 | 15,896 | **none** | pool |
| `panel_sweep2_0908` | 09-08 21:25 | 12,771 | **none** | pool |
| `panel_sweep3_0909` | 09-09 17:04 | 37,367 | **none** | pool |
| `unit2-4_1009` | 10-09 16:51 | 24,277 | yes — sprays, field off | pool; unit 2-4. 77,568 confirmed crops, 67,819 on tracks ≥ 200 sightings (87% above the phantom level) — more than all earlier recordings together |

**Apparatus numbering (from 2026-10-09):** the lab has 16 units in 4 rows of 4. A unit
is `row-position`: row counted from the professor's seat (1–4), position counted left
to right as seen from the professor's seat (1–4). Record the unit with every new
recording; the fixtures above predate this and their units are unknown. 2026-10-09:
unit **2-4** (unit 2-3 showed no droplets when sprayed; dropped).
2026-10-15: units **2-1, 3-2, 3-4, 4-2, 4-1, 1-3** (`unit<r-p>_1015`, 12–15 min each,
sprays with the field off, 0 frames dropped). Stored on D:\millikan-fixtures (C: is
full) and linked into `test/fixtures` by directory junctions. Unit 3-2 showed no
droplets at first; the owner found the fault and fixed it. Backgrounds range from
gray 25 to 122 across units; droplets stand 76–139 above them. This completes the
training data: 7 units. **Held out as the unseen-unit test: 1-3 and 3-2**, drawn at
random from the six new units (`random.Random(20261015).sample(units, 2)`);
never trained or tuned on, hand-labelled with `edge/label.py`.
**Reserved for the live demo (owner, 2026-10-15): units 1-1, 2-2, 3-3, 4-4** —
never recorded for training, so the demo runs on units the model has never seen.

The sweeps are recordings of the voltage panel with nothing sprayed: every blob the
detector keeps in them is a phantom **with certainty**, which makes them the cleanest
negatives there are.

`edge/autolabel.py` replays a fixture through the unmodified pipeline (wrapped from
outside, as the parent's `detacc_replay.py` does) and writes every contour — kept,
or rejected and by which gate — with a 32×32 crop of frame and background, labelled
from the tracker afterwards. All ten fixtures but `live_20260903_noisy` have been
run; the output is in `data/autolabel/` (gitignored, ~230 MB), with contact sheets
in `data/autolabel/sheets/`.

**The tracker's "confirmed" is not a clean positive label**, measured two ways:

- against the hand labels on `bench_0908`: confirmed blobs are droplets **93.3%** of
  the time (85% on tracks of 6–50 sightings, 94.5% at 50+); kept-but-unconfirmed
  blobs are droplets **32%** of the time, so they are **not** safe negatives
- on `one_droplet_slow`, the noisy 09-05 session, by contrast against the sweeps'
  certain phantoms (peak over background noise in the crop): only **49%** of
  confirmed crops stand above the phantoms' 95th percentile — 19–32% on tracks
  under 200 sightings, 96% at 200+. The tracker there kept phantom tracks alive for
  up to 200 sightings, which is §3's problem again. `spray_labelled_0907`: 94%.

Use the hand-labelled `bench_0908` set as the **held-out test set** — it is
independent of the tracker, which the auto-labels are not.

---

## 5. What the detector currently does

`cv/pipeline.py`, `detectDroplets()`. There is no neural network anywhere in it.

1. **rolling background subtraction** — median over a 300 s window
   (`BG_WINDOW_S`), rebuilt when detections stay 4× above the session's own quiet
   rate
2. **adaptive threshold** — `max(Otsu, median + 6·MAD)`; the MAD floor exists
   because Otsu alone splits a histogram with no droplets in it
3. **shape gates** — area 30–900 px, circularity ≥ 0.52 (`CIRCULARITY_MIN`),
   plus masks for the grid lines, the panel margin and static marks
4. **track confirmation** — a blob must be seen 6 times before it is a track

A learned patch classifier replaces **step 3** and nothing else: given a ~24×24
crop around a candidate, is this a droplet or a speck. That is the smallest
change that addresses §3, and the easiest thing in the world to quantize.

Useful constants live near the top of `cv/pipeline.py` with the measurement that
set each one written above it. The comments in that file are the real
documentation — they record what was measured and what failed, not what the code
does.

---

## 6. THE OPEN DECISION

The original handover framed this as undecided. It is not, on the evidence in §3.

**Option A — train a small classifier.** Gives a real model to convert and
quantize, keeps the Edge AI story intact, and is motivated by a failure the
project already documented. The condition the original handover set — "only
justified if plain thresholding actually struggles" — **is met and measured.**

**Option B — benchmark the classical pipeline on the board.** Still a legitimate
edge story (ARM optimisation, throughput under constraint, CPU vs NPU), but it
skips the quantization and runtime-conversion steps that Edge AI job descriptions
name explicitly.

**Recommendation: A, with B's numbers kept as the baseline row.** The comparison
is the artefact — classical / FP32 / INT8 across latency, FPS, memory and
accuracy. Three rows beat one, and you get B for free on the way to A.

The board was ordered on 2026-09-26 without waiting on this (§8.1); it serves
either option.

---

## 7. Method cautions that will decide whether the numbers mean anything

1. **Split the test set by SESSION, not by frame.** Frames within a session share
   background, focus and noise. A random frame split will overstate accuracy
   badly, and it is the single most common way results like this become
   worthless.
2. **Keep the classical path in the repo and runnable.** It is the baseline row
   and the sanity check.
3. **Get the FP32 number and stop.** That is a publishable result even if the
   board never arrives, and it isolates "the model helps" from "the conversion
   preserved it".
4. **Each step should produce a number you could publish before starting the
   next.** The parent project's own lesson, learned the expensive way: define
   what done looks like before executing.
5. **Accuracy delta is measured against the hand-labelled set**, not against the
   tracker's own output. Otherwise you are measuring agreement with the thing you
   are replacing.

---

## 8. Environment

- **Build on the Vivobook under WSL2 Ubuntu** (ASUS Vivobook S15 S5507QA,
  Snapdragon X Elite, Windows 11 ARM64). This supersedes the 2026-09-25 plan to
  build on the RTX 4060 machine. RKNN-Toolkit2 ships Linux ARM64 packages as well
  as x86_64 (`rknn-toolkit2/packages/arm64` in airockchip/rknn-toolkit2, checked
  2026-09-26); only native Windows on ARM is unsupported, and WSL2 is Linux. The
  Vivobook runs classical CV fine (~28.7 fps measured).
  - No CUDA on the Vivobook: training runs on the CPU. Slower, and fine for a
    ~24×24 patch classifier. The RTX 4060 machine stays available for faster
    training and for x86-only tools (reflashing, §8.1).
- **WSL2 installed 2026-09-30:** Ubuntu 22.04, aarch64, Python 3.10.12 — the
  board's Python. No reboot was needed. Default user `lan` (passwordless sudo; set
  one with `passwd`). Toolchain in `~/venvs/rknn`, installed by `edge/setup_wsl.sh`
  in ~4 min: RKNN-Toolkit2 **2.3.2** (matches the board's runtime), torch 2.2.0,
  onnx 1.16.1, onnxoptimizer 0.3.8 (no ARM64 wheel — built from source, hence
  `build-essential cmake`), numpy 1.26.4. `pip freeze` in `~/rknn-pkg/freeze.txt`.
  From Git Bash, call it with `MSYS_NO_PATHCONV=1 wsl.exe ...`, or `/mnt/c/...`
  paths get rewritten into Windows ones.
- **The code already runs on Linux.** The Windows-only parts of `cv/` — the
  camera backend choice in `cv/source.py`, the DPI call in `cv/pipeline.py`, the
  `winsound` beep — sit behind `sys.platform` checks or a `try`. Two exceptions:
  `cv/UVC_check.py` hard-codes DirectShow and needs a Linux path before it is used
  on the board, and the 30 fps Media Foundation result in `cv/source.py` is a
  Windows measurement — on the board OpenCV uses V4L2, so capture fps must be
  measured again (it belongs in the benchmark anyway). `app_js/wrapper.py` is
  pywin32-only and out of scope.
- **Docker:** not needed to start. Recommended after the FP32 result: put the
  conversion toolchain in a Dockerfile so its pinned Python and numpy versions
  travel with the repo — that is most of the "reproducible" criterion in §9.
- **Expect the RKNN toolchain to be the schedule risk**, not the model. Thin
  documentation, rough edges.
- **Claude Code on a VPN machine:** a 403 on `/login` means the terminal is going
  out unproxied while the browser is on the VPN. Set `HTTP_PROXY`/`HTTPS_PROXY`
  to the proxy client's mixed port. For Clash Verge specifically, rules only
  apply in **Rule** mode — Global mode ignores them entirely — and
  `anthropic.com` / `claude.ai` need DIRECT rules or the session dies when the
  proxy comes up. A working extend script is at `test/clash_extend_script.js` in
  the parent.

### 8.1 The board — ordered 2026-09-26, arrived and passed 2026-09-29

**Dingchang (定昌) DC-AI588S V02: RK3588S, 8 GB RAM, 32 GB eMMC**, with a 12 V
adapter.

| | |
| :-- | :-- |
| NPU | 6 TOPS, INT4/INT8/INT16. Identical on every RK3588/RK3588S; the listing's "AI" tag names the board line, not a better chip |
| CPU | 4× Cortex-A76 + 4× Cortex-A55, up to 2.4 GHz |
| OS | Ubuntu 22.04.5 LTS, installed by the vendor (the spec sheet lists only Android 12) |
| camera | 1× USB 3.0 + 1× USB 2.0 host (MIPI CSI also present) |
| power | 12 V DC, 5.5×2.5 mm barrel, 3–5 A. The USB-C port is OTG (debug, flashing), not power. All USB ports share 3 A |
| storage | 32 GB eMMC; M.2 2242 PCIe SSD slot; TF card up to 128 GB |
| display, network | 1× HDMI (the product photo shows two; the spec sheet says one); gigabit Ethernet |
| spec sheet | [`docs/DC-AI588S-V02_spec.pdf`](DC-AI588S-V02_spec.pdf) — vendor 产品规格书 V1.0, 2026-01-04 |

**The listing excludes 7-day no-reason returns.** A return needs the board to
differ from the order (e.g. no Ubuntu), so run the checks below the day it
arrives. Support for images and the SDK is the vendor's after-sales engineers.

**Arrival checks** (~1 h, on the board):

| check | how | if it fails |
| :-- | :-- | :-- |
| Ubuntu, and which | `lsb_release -a` | Android → return. 20.04 and 22.04 are fine. 18.04's Python 3.6 is too old for RKNN-Toolkit-Lite2 (needs ≥ 3.7): workable, but ask for a newer image |
| NPU driver | `dmesg \| grep -i rknpu` prints a version | Nothing → ask support for an image with the driver; return if none comes within a few days. An old version is not a fault: pin the RKNN runtime, and the host toolkit, to a matching version |
| camera | plug into USB 3.0, `ls /dev/video*`, grab one frame with OpenCV | Try the USB 2.0 port; `v4l2-ctl --list-devices` |
| NPU runs | `pip install rknn-toolkit-lite2`, run its bundled ResNet example | Almost always a runtime/driver version mismatch → match the versions, or ask support |

Return only if it is not Ubuntu or the NPU cannot be used at all; everything else
is fixable. Once it passes, write the Ubuntu version, `uname -r`, the NPU driver
version and the RKNN runtime version into the repo — they are part of
"reproducible".

**Results, 2026-09-29 — passed; nothing to return:**

| | |
| :-- | :-- |
| OS | Ubuntu 22.04.5 LTS, kernel `6.1.118` aarch64, GNOME desktop, user `ztl`; Docker installed |
| NPU driver | **0.9.8** (`/sys/kernel/debug/rknpu/version`). The `can't request region` lines in `dmesg` are harmless: the NPU runs |
| RKNN runtime | `librknnrt` **2.3.2**, preinstalled — so the host RKNN-Toolkit2 must be **2.3.2** too |
| NPU runs (C) | Rockchip's `rknn_common_test`, MobileNetV1 INT8: **2.95 ms/inference (~338 FPS)**, dog image → dog class |
| NPU runs (Python) | RKNN-Toolkit-Lite2 2.3.2, bundled ResNet18: "space shuttle" at 0.9997 |
| Python | 3.10.12; installed offline: numpy 1.26.4, opencv-python-headless 4.10.0.84, pandas 2.2.3, scipy 1.15.3, rknn-toolkit-lite2 2.3.2 (+ psutil, ruamel.yaml). No pip came with the image; pip 26.2.1 was bootstrapped from its own wheel |
| RAM, disk | 7.7 GiB; 29 GB root, 21 GB free |
| CPU | 8 cores, governor `performance` |
| camera | **The lab's capture dongle works (2026-10-09):** MacroSilicon `345f:2109`, UVC 1.00 → `/dev/video0`, no driver needed. 1280×720: **MJPG 30.0 fps, YUYV 22.9 fps** (measured 10 s each, OpenCV V4L2). `autoROI` finds the reticle at x21 y36 909×645, 61.0 px/div — the same framing as `bench_0908`. Used in the **USB 2.0** port: the dongle is too thick for the USB 3.0 port next to the USB-C cable. MJPG is compressed and every recorded fixture is uncompressed, so which one the live path uses is a decision, not a default |

**Access without a network: ADB over the USB-C port.** Board USB-C → laptop USB-A.
It enumerates as `2207:0006` "rk3xxx"; Windows binds its in-box WinUSB driver by
itself, and `adb shell` is a root shell with no password. `adb push` moves files at
~33 MB/s. Google's platform-tools (x64) run fine on the Vivobook. The board has no
Wi-Fi, and no Ethernet cable was needed for any of the above: packages are
downloaded on the laptop (`pip download --platform manylinux2014_aarch64
--python-version 310`) and pushed. Staging lives in `/root/edge_setup/` and
`/root/millikan-edge/` on the board.

**No RTC battery:** the clock comes up wrong after a power loss. Set it before
anything that timestamps: `adb shell "date -s @$(date +%s)"`.

**Reset, 2026-09-29 ~23:12 — a loose adapter plug.** The board lost power twice in a
row and a file it had just written came back empty. The owner found the adapter
loose; not a board fault. Keep the habit it taught: `sync` after writing results on
the board.

**Reflashing, only if ever needed:**

1. **USB stick** — the spec sheet lists PC/U盘 upgrade; get the image and steps
   from vendor support. Needs no driver on any laptop.
2. **`rkdeveloptool`** (open source) on Linux — from WSL2 through usbipd-win,
   which has an ARM64 build.
3. **Rockchip RKDevTool on x86 Windows** — the RTX 4060 machine, not the
   Vivobook: Windows on ARM cannot load x86/x64 kernel drivers.

---

## 9. Timeline and definition of done

| date | item |
| :-- | :-- |
| 26 Sep 2026 | ✓ board ordered: DC-AI588S, RK3588S 8+32G (§8.1) |
| 29 Sep 2026 | ✓ board arrived; arrival checks passed except the camera (§8.1) |
| 29 Sep 2026 | ✓ "before" numbers frozen; auto-labels extracted from every raw fixture (§3, §4, §11) |
| 30 Sep 2026 | ✓ WSL2 and the RKNN toolchain installed on the Vivobook (§8) |
| 16 Oct 2026 | ✓ unseen-unit test: F1 77.7% → 90.3% on two units never trained on (§11) |
| 17 Oct 2026 | ✓ the board on real HDMI; live view 18.1 → 20.6 updates a second, same answers; the hunt aid's suggestions reviewed — no change (§11) |
| 27 Oct 2026 | ✓ live on reserved units 1-1 and 3-3: 11 legal falls, each within 0.9 s of a phone-video timing; raw and MJPG recordings of unit 3-2 for the MJPG test (§11) |
| 28 Oct 2026 | ✓ MJPG scored against hand labels: classical precision 84.6% → 37.5%, the classifier's F1 85.4% → 85.1%; the repo made public (§11) |
| 29 Oct 2026 | ✓ live on MJPG, unit 2-2: 27.6 updates a second; power: the board 2.9 W, the laptop 15–17 W (§11) |
| Mar 2027 | the owner's capture thread and C++ ports measured live — one lab visit |
| Oct 2026 – Jan 2027 | build window |
| **Jan 2027** | **MUST: inference running on the board, numbers recorded** — ✓ met 11 Oct 2026 |
| Nov–Dec 2026 | BEST: README with the before/after table, demo video |
| Dec 2026 | BEST: Gitee mirror |
| Mar–May 2027 | internship applications — what this exists to support |

**Done means:**

- [x] model (or classical pipeline) running inference on the RK3588 — 11 Oct 2026, live
  on reserved units 27 Oct
- [x] before/after recorded: latency, FPS, memory, accuracy delta — `results/`
- [x] reproducible — 27 Oct 2026, from a fresh clone: the FP32 replay of the test
  session came out bit-identical and `edge/score.py` gave the published table row for
  row (the INT8 rows reuse the WSL step's output); laptop packages pinned in
  `edge/requirements.txt`

Everything past that is upside.

Commit as you build. A repo that appears fully formed three months before
applications reads as job-hunt preparation, and the activity period is the first
thing an interviewer checks after opening the link.

---

## 10. Inherited conventions worth keeping

From the parent, because they earned their place:

- **Never invent a threshold.** Every constant in `cv/pipeline.py` has the
  measurement that set it written above it. Keep that habit; it is most of why
  the baseline in §3 is defensible.
- **An acceptance gate before a refactor.** `check.py fixtures` must keep five
  reference falls and stay above a traverse baseline. Build the equivalent here
  before changing the detector: a fixed labelled set and a score, run before and
  after every change.
- **The operator's observation is evidence.** When the bench and the logs
  disagree, the bench is right and the logs need explaining.

---

## 11. Where the work stands — 2026-09-29

### Built

| path | what |
| :-- | :-- |
| `edge/common.py` | paths, the fixture registry (dates, roles), the parent's matching rule |
| `edge/score.py` | **the fork's gate** (§10's "fixed labelled set and a score"): any row of predictions against the hand labels, all rows by the same code |
| `edge/autolabel.py` | replay → every contour, 32×32 crops, tracker labels. **Validated:** with its hooks in place `one_droplet_slow` gives the same 52 traverses and the same 14 kinematically legal falls as the parent's unhooked `check.py` on the same code, which passes |
| `edge/contact_sheet.py` | 96-crop sheets per class, for eyeballing labels |
| `edge/bench_detect.py` | `detectDroplets()` latency on fixed frames, any machine |
| `results/` | the frozen baseline, scores and latencies — indexed in `results/README.md` |

### Numbers so far

| | |
| :-- | :-- |
| classical detector, current code, `bench_0908` | precision 81.0%, recall 95.8%, F1 87.8%, **1.35 false detections a frame** |
| phantom rate, `live_check` | **3.11** detections a frame against the 0.87 bar |
| `detectDroplets()` latency | board CPU 4.62 ms (noisy) / 5.70 ms (typical); Vivobook 2.02 / 2.92 ms; identical detection counts on both |
| NPU sanity | MobileNetV1 INT8 2.95 ms on the board |

### Labels available (every raw fixture but the eval one)

| | count, excluding `bench_0908` |
| :-- | --: |
| confirmed-track crops, any length | 17,936 |
| …on tracks of ≥ 50 sightings | 13,044 |
| …on tracks of ≥ 200 sightings | 5,519 |
| kept blobs in the droplet-free sweeps — certain phantoms | 2,809 |
| gate-rejected crops (sampled, ≤ 40 a frame) | 226,753 |

How clean "confirmed" is depends on the session, measured as the share of crops
whose peak stands above the sweeps' phantom 95th percentile (a proxy — a defocused
real droplet fails it too): `spray_labelled_0907` 94%, `one_droplet_full` 85%,
`one_droplet_slow` 49%, `zoom_labelled_0907` 21%; at ≥ 200 sightings every session
is 85–98%. On `bench_0908`, where there is truth, confirmed blobs are 93.3% droplets.

`one_droplet_slow` also shows the phantom problem at the level of results: the
current code finds 14 kinematically legal falls where the operator timed 5 (the
five are all kept; the nine extra are unverified — other droplets or phantom
tracks).

### Decisions waiting on the owner

1. ~~**§6, option A**~~ — **decided 2026-09-30: A**, train the classifier.
2. ~~Positives, negatives, split~~ — **decided 2026-10-09**, on the owner's blind
   review of 236 tracks (`data/review/REPORT.md`; droplets per 20 shown, by track
   length 6–49 / 50–199 / 200+): spray 19 / 18 / 20, unit 2-4 11 / 11 / 17,
   one_droplet_slow 0 / 2 / 19, zoom 0 / 0 / 9 of 16.
   - **Positives:** spray, every confirmed track; unit 2-4, one_droplet_slow and
     one_droplet_full, tracks ≥ 200 sightings; zoom, none.
   - **Negatives:** the phantom tracks the review found (zoom confirmed < 200,
     one_droplet_slow confirmed < 50), kept blobs in the droplet-free sweeps, and
     gate rejects.
   - **Split by day:** train 10-09 (unit 2-4), 09-05, 09-09 — 71,659 positive /
     201,974 negative crops; validate 09-07, 09-08 evening — 2,143 / 88,588; test
     `bench_0908` (09-08 morning, hand labels) only.
   - Unit 2-4 is 95% of the positives: cap crops per track and balance recordings.

### Ready to train (2026-10-09)

- `edge/dataset.py` → `data/dataset/` (gitignored): train 78,484 crops (32,468
  droplet / 46,016 noise), validate 41,038 (1,831 / 39,207); per-recording table in
  `data/dataset/SUMMARY.md`.
- `edge/train.py`, in WSL: 15,345-parameter conv net on the 24×24 frame-minus-
  background centre; class- and recording-balanced sampling; the best epoch is
  chosen on validation AP only. Smoke run passed, ONNX exported, and RKNN-Toolkit2
  2.3.2 converted that ONNX for rk3588 (FP16; it logs `E RKNN: Unkown op target: 0`
  yet returns 0 and writes the .rknn — check on the board).
- Full run: `wsl -d Ubuntu-22.04 -- bash -c "~/venvs/rknn/bin/python
  /mnt/c/Users/LAN/millikan-edge/edge/train.py"` — estimated ~1 h on the CPU.
- **Trained 2026-10-09** (`results/fp32/`): 40 epochs, ~58 s each. Best epoch 26 on
  validation: AP 0.957, AUC 0.998; at 0.5 it passes 97.2% of droplets, 2.2% of the
  review's phantom tracks, 1.2% of sweep blobs, 0.8% of gate rejects. Crop-level and
  validation-only — next is the classifier in the pipeline, scored on the test set.

### First before/after (2026-10-10) — `results/fp32/README.md`

- Classifier in the pipeline (`edge/gate.py`, from outside `cv/`), threshold 0.5.
  **Detector stage on the test set: false detections 1.35 → 0.45 a frame,
  precision 81.0 → 92.6%, recall 95.8 → 93.3%, F1 87.8 → 93.0%.** All 19 legal
  falls kept, durations within 0.04 s. Tracker stage about unchanged. Kept variant:
  classifier *after* the gates ("replacing" them lost a legal fall).
- `live_check` fixture: 3.11 → 2.08 a frame; the removed third is the noise (§3).
- INT8 (`results/int8/`): same decision as FP32 on 99.89% of validation crops,
  AP 0.9582 vs 0.9568 — on the RKNN simulator. Next: the NPU on the board.
- Replays of `bench_0908` run on Windows: from WSL, reading the 21 GB of PNGs across
  `/mnt/c` was twelve times slower.

### On the board (2026-10-11) — `results/board/README.md`

- INT8 on the real NPU = FP32 on 99.90% of validation crops.
- One crop per NPU call loses to the CPU (0.31 vs 0.12 ms): a 15k-parameter model is
  all call overhead. Batched (one model per batch size 8/16/32, NHWC input) it is
  0.045 ms a crop. `rknn_batch_size` gave wrong answers; the batch is baked into the
  ONNX export instead (`edge/convert_batched.py`).
- Whole pipeline, dense spray (26 blobs a frame): classical 38.3 ms/frame, 175 MB;
  classifier INT8 on the NPU, batched, 41.1 ms/frame, 200 MB; detector + gate 8.9 ms
  median (p90 16.3) — inside a 30 fps budget. Same 17 falls in every variant.

### Live on the board, window on the laptop (2026-10-11)

`python edge/live_viewer.py [--record]` — the whole live session from the laptop:
the board runs the pipeline (classifier INT8 on the NPU, batched, after the gates)
with its window streamed over the USB-C cable, and clicks and `q` go back. Click a
droplet to pick it, as in the parent's window. `--record` saves what you see as MP4
in `data/live/`; the board's session folder (falls.csv, report) is copied there too.
`--source frames:/root/test/fixtures/spray_labelled_0907` rehearses on a recording.

- Board side `edge/live_board.py`: takes over the pipeline's window calls from
  outside `cv/` (imshow → stream, setMouseCallback → laptop clicks, waitKey → laptop
  keys). The parent's `--publish` stream was sent before the HUD, and its remote
  clicks needed a real window.
- Chinese HUD text (平衡/提升) is drawn with Noto Sans CJK on the board: `cv2.putText`
  draws `?` for it — in the parent's own window too.
- Rehearsed 2026-10-11 on `spray_labelled_0907`: ~11 frames a second in the viewer,
  a laptop click picked the droplet under it, clean stop on `q`.
- 2026-10-12, owner's hands-on test: clicks, timer, HUD all fine, but choppy — 7.9
  updates a second, 31% of gaps over 150 ms (measured from their screen recording).
  Cause: the pipeline shows every 2nd frame (`SHOW_EVERY`, set for the parent's
  14 ms Windows imshow). `live_board.py` sets `P.SHOW_EVERY = 1`: 21.8 updates a
  second, gap median 44 ms, p90 59 ms on the same recording. The viewer prints this
  smoothness line at the end of every session.
- At the bench: dongle → board **USB 2.0** port (the USB 3.0 port fouls the USB-C
  plug), board USB-C → laptop USB-A, 12 V seated. `adb` lives in
  `data/tools/platform-tools/` (gitignored).

### Without the board (2026-10-14)

- **INT8 on the test set** (`results/int8/QUANTIZATION.md`, `edge/int8_test.py`):
  the classical blobs at the labelled frames filtered by each model, INT8 from the
  RKNN simulator — INT8 92.1% precision / 93.7% recall / F1 92.9% / 0.48 false a
  frame, FP32 92.0 / 93.3 / 92.6 / 0.49 by the same filter; same decision on 99.49%
  of 1,568 test blobs. The filter sits within 0.04 false a frame of the full replay.
- **Labelling tool for the unseen units:** `edge/label.py sample|label|review
  <recording>` — the parent's `detacc_label.py`, same screens and keys, made
  general: 100 frames from four equal slices of the session plus 10 unannounced
  repeats; labels saved to `labels/<recording>/` (tracked). Run `sample` once per
  unseen unit after recording, then `label` (about 2 h a unit). Scoring an unseen
  unit still needs its replay and detection-area geometry — to build after the
  recordings exist.
- **README.md** written: problem, results, what the board taught, how to reproduce.

### More units, a retrained model, the unseen-unit test (2026-10-15)

- **Six units recorded** (§4): 2-1, 3-4, 4-1, 4-2 train; **1-3 and 3-2 drawn at random
  as the unseen units**; 1-1, 2-2, 3-3, 4-4 reserved for the live demo.
- **Auto-labels** for the four training units (`data/autolabel/`): 18,446–39,899
  confirmed crops each. Unit 2-1 is the noisiest (62,801 tracks against 4,000–16,000
  elsewhere; noise 5.9 grey levels): its short tracks sit at the noise level (median
  peak 14–15 over background) while its 200+ tracks peak at 70 — the 10-09 review's
  rule (positives = tracks of 200+ sightings) separates them, and holds on every unit
  (200+ tracks 70–103 against 14–55 for shorter ones). No short-track negatives from
  the new units: no review covers them.
- **Dataset v2:** 146,026 training crops (69,990 droplet), validation unchanged
  (41,038). v1 kept as `data/dataset_v1`.
- **Retrained** (`results/fp32_v2`, `results/int8_v2`): validation AP 0.9697 vs 0.9568,
  precision@0.5 85.1% vs 82.0%, droplet pass rate 95.1% vs 97.2%. INT8 agrees with FP32
  on 99.94% of validation crops. On the bench_0908 test session the two models score
  the same (offline filter: F1 92.7% vs 92.6%, 0.48 vs 0.49 false a frame) — the
  choice of v2 rests on validation and on the unseen-unit test, not on bench_0908.
- **The unseen-unit test** (`edge/score_unseen.py`): truth from `labels/<unit>/`
  (`edge/label.py`, corrected by its blind review); the detection area each unit's
  gates used, saved by `edge/autolabel.py <unit> --labelled` (both units replayed);
  rows: classical detector and tracker, and each model as an offline filter; quarters
  and units weighted equally. `--todo <unit>` writes the blind review's spots once a
  unit is fully labelled.
- `edge/label.py`'s stretch contrast now centres on each recording's own background
  (backgrounds run from grey 25 to 122 across units; the fixed table turned unit 1-3
  white). Same table as before at bench_0908's level.
- Batched v2 models for the board were not built: the unseen-unit test kept the first
  model (below).

### The unseen-unit test: the classifier holds, the first model stays (2026-10-16)

- **Labels** (`labels/unit1-3_1015/`, `labels/unit3-2_1015/`): 100 frames a unit plus
  10 repeats; 2,791 sure droplets in the detection area on 1-3 and 3,359 on 3-2 —
  three to four times bench_0908's density. Blind review: 918 spots on 1-3, 2,771 on
  3-2. `--todo` now skips spots in the blanked parts of the detection area, which are
  never scored (about 600 on 3-2).
- **Score** (`results/unseen/SCORE.md`), both units pooled, each unit and each quarter
  weighted the same:

  | row | precision | recall | F1 | false / frame |
  | :-- | --: | --: | --: | --: |
  | classical detector | 69.8% | 87.6% | 77.7% | 11.66 |
  | classical tracker | 84.0% | 83.0% | 83.5% | 4.87 |
  | first model, FP32 | 93.5% | 87.4% | 90.3% | 1.86 |
  | first model, INT8 (simulator) | 93.5% | 87.4% | 90.3% | 1.86 |
  | retrained, FP32 | 93.6% | 86.8% | 90.1% | 1.83 |

- **Per unit:** 3-2 is the noisy one — the classical detector keeps 21.65 false a frame
  at 57.3% precision; the classifier cuts that to 2.16 (93.1%) for recall 86.5 →
  86.1%. On 1-3 the classical detector is already 93.7% precise (1.68 false a frame);
  the classifier leaves it at 94.1% (1.56), recall unchanged.
- **The review by source** (withheld from the reviewer): on 3-2, of 1,917 unlabelled
  detections both models dropped, 1,909 were judged not droplets and 1 a droplet;
  645 of 659 hand labels that some row missed were confirmed. Of the unlabelled
  detections a model kept, 57 of 180 (3-2) and 71 of 151 (1-3) were droplets the
  labeller had missed.
- **INT8 = FP32 here too:** same decision on 100.00% of 2,668 blobs (1-3) and 99.90% of
  5,168 (3-2) — `edge/int8_test.py --fixtures unit1-3_1015 unit3-2_1015`.
- **Decision (owner): the first model stays.** Four more training units moved neither
  test (F1 90.1% vs 90.3% here, 92.7% vs 92.6% on bench_0908), and the first model is
  the one built and measured on the board. `results/fp32_v2`, `int8_v2` stay as the
  record that more units did not help.
- No more lab visits for data: the one left is the live demo on the reserved units.

### Beyond accuracy: what the operator sees, the board on its own, heat (2026-10-17)

- **The hunt aid's suggestions** (`edge/suggestions.py`; the measure is fixed in its
  header, written before any replay ran): the pipeline's display path replayed with
  nobody clicking, recording every track `candidates()` rings and every track the
  idle auto-pick squares — test session and both unseen units, classical vs the
  classifier after the gates (FP32). The two suggest nearly the same tracks: of 477
  suggested objects, 410 are suggested by both, 34 only without the classifier and
  33 only with it (test session 204 / 28 / 18, unit 1-3 110 / 4 / 12, unit 3-2
  96 / 2 / 3). The aid's own tests (a steady fit, travel, charge) already turn away
  most of what the classifier removes. The owner's blind review of all 477
  (2026-10-17: 401 droplet, 47 not, 29 unsure; `results/suggestions/REPORT.md`):
  **the classifier does not reduce wrong suggestions** — wrong suggestions per 10
  minutes 9.2 → 8.7 on the test session, 1.7 → 1.7 and 3.3 → 3.3 on the unseen units;
  the share that are not droplets 18% → 17%, 2% → 2%, 4% → 4%. Of the 34
  suggestions only the classical detector makes, 7 are not droplets; of the 33 only
  the classifier makes, 4; the other 36 wrong ones are made either way — the
  classifier passes them too. So its gain is where it was measured, in detections
  (the rings, the tracked count, what the tracker has to sort through), not in what
  the hunt aid points the operator at, which its own tests already kept clean. The
  clips are cut by `review_tracks.clip` with each clip's background first shifted
  to grey level 31: unit 1-3's video reads ~116 (its brightness turned up), and the
  stretch set on the others turned its 126 clips white. One rule for every clip,
  set before any answer was given.
- **The board on its own screen** (`edge/live_board.py --local`, set up once by
  `edge/board_local.sh`): the board already runs Ubuntu's desktop with automatic
  login and has HDMI; its pip OpenCV is headless, so opencv-python 4.10.0.84 (Qt)
  sits apart in `/opt/cvgui`, pushed over adb (the board has no network). Desktop
  launchers "Millikan live" (cam:0) and "Millikan rehearsal" (spray_labelled_0907)
  run as the desktop user, NPU included; closing the window ends the session as `q`
  does (tested); sessions land in `~/live`. The window costs ~2 ms a frame (imshow
  0.8 ms, its event handling 1.1 ms more than none): 16.8 updates a second without
  it, 16.4 with it, on spray_labelled_0907's first 1,500 frames.
  On real HDMI 2026-10-17 — the board's output into the HDMI capture card, shown on
  the laptop (the board's own mouse). The board picks 1920×1080. The window is about
  1280×850 with Qt's toolbar and status bar: it fits; the CJK text, click-to-pick and
  close-to-stop all work. "Millikan rehearsal": 23.0 updates a second over the
  first 955 frames (about the recording's own 23.5), 18.1 over the whole recording
  (the dense spray at the end is slower), **20.6** after the matching fix in "Speed"
  — gap p90 99 → 73 ms, gaps over 150 ms 2% → 1%, the falls file byte for byte the
  same. Two fixes to the launchers:
  `adb shell` runs with umask 0000, so `board_local.sh` had written them world-
  writable and GNOME refused them ("此桌面文件权限非法") — it now sets `umask 022`.
  A second double-click while the first session was still starting opened a second
  session, with two windows and one session folder — `run_local.sh` now holds a
  `flock`, and a second launch exits.
- **Heat** (`edge/soak.py --minutes 15`, `results/board/soak_npu_15min.json`): the
  live path with its window, INT8 on the NPU, spray_labelled_0907 looped. Hottest
  zone 49.9 → 62.8 °C, levelling off; every CPU and NPU clock at its maximum
  throughout — the board's first throttle point is 75 °C. 14.1 updates a second,
  per minute 10.8–18.2 with no downward trend. That is the full live path (hunt
  aid, rings, HUD every frame) on the busiest recording; `board_bench.py`'s 24.3 FPS
  ran the pipeline without its window.
- **Speed of the live path** (`edge/fast.py`, `results/board/README.md`): with its
  window the live path costs 62 ms a frame on the busiest recording (~10 ms of it PNG
  decoding, which a camera does not pay); `board_bench.py`'s 24.3 FPS ran without
  the window. Profiled: `Track.recentSpeed` took 21% of the time. `edge/fast.py`
  computes it and `windowFit` with the same arithmetic, cached per track: identical
  falls, rings, picks and sightings on 6,000 frames of bench_0908 and 4,000 of unit
  3-2; 62.1 → 55.8 ms a frame, 14.2 → 15.7 updates a second. On in `live_board.py`
  and `soak.py`.
  The owner then saw the view slow down in the dense spray (2026-10-17). Timed frame
  by frame, cost follows the number of live tracks — about 22 ms plus 0.42 ms a track;
  the spray's last 637 frames carry ~92 (up to 239). There `assign` (matching tracks
  to blobs) was the largest cost: its cost matrix filled one pair at a time.
  `fast.py` now builds it as one numpy expression with the same float64 operations;
  `python edge/fast.py exact <recording>` runs both on every call and compares the
  matrices bit for bit — identical on the board (all 1,592 frames of the spray) and on
  the laptop (spray, unit 1-3 and unit 3-2 whole, bench_0908's first 15,000 frames).
  Dense spray 76.4 → 61.4 ms a frame (13.1 → 16.3 updates a second), whole recording
  55.2 → 48.1 ms (18.1 → 20.8), two runs each. The occasional stall is a fall
  completing: the latched voltage is read from ~70 panel crops at once, ~230 ms a
  fall; left alone, as the clock has stopped by then. The parent's own frame-rate fix
  (Media Foundation's MJPG at 30 fps over DirectShow's raw YUY2 at 23.6) is about
  the camera, not this: the board's dongle offers the same two (MJPG 30.0, YUYV
  22.9), but the dense spray is bound by processing, and every recording the
  classifier was trained and scored on is raw. What is left is spread thin — Qt's
  painting, the detector, panel reading, the hunt aid's fits — a few ms each. The
  board's copies (`/root/millikan-edge`, `/opt/millikan-edge`) have it.
- **Power** (`edge/soak.py --minutes 10`, the same looped recording on either
  machine). The laptop on its battery, Windows' own discharge reading every 5 s
  (2026-10-20, nothing else on its ports, screen at full brightness): 8–9 W idle,
  15–17 W in the live view (18.9 updates a second); the two readings the battery gives
  — its rate and its falling capacity — differ by ~2 W, hence the ranges. A first run
  with an iPad charging from the laptop read 26 W idle and was discarded. The board
  from a lab bench supply at 12.0 V (GW Instek GPP-6030), 2026-10-29: **1.1 W idle,
  2.9 W in the live view** (18.3 updates a second) — about a fifth of the laptop.
  The supply's screen was filmed through the run and read every 2 s by Windows' OCR:
  202 readings, ±0.06 W on the mean; where a frame also gave the current, 12 V × amps
  matched the watts in 109 of 122 (`results/power/README.md`). The first board run
  (2026-10-27) used the supply's recorder, which saves to its internal memory; its
  storage menus froze whenever the recording was to be copied out, so that run's
  readings were never read — filming the main screen needs none of those menus.

- **The laptop's NPU** (`edge/laptop_npu.py`, `results/laptop_npu/`): the same model
  as INT8 QDQ (onnxruntime's QNN recipe, the RKNN model's 499 calibration crops) on
  the Vivobook's Hexagon NPU through Qualcomm's plugin provider. Needs native ARM64
  Python — the usual one is x64 under emulation: Python 3.12.10 ARM64 unzipped from
  the `pythonarm64` NuGet package into `C:\Users\LAN\tools\py312arm64`, then
  `pip install onnxruntime-qnn numpy onnx` (onnxruntime 1.31.0, onnxruntime-qnn
  2.7.0). The export is opset 12; per-channel DequantizeLinear needs 13+, so it is
  converted to 17 first. Agrees with FP32 on 99.93% of 41,038 validation crops;
  0.058 ms for one crop, 0.0037 ms a crop at batch 32 (RK3588: 0.193, 0.045).

### The demo visit: checklist

Units **1-1, 2-2, 3-3, 4-4 only** — reserved, never recorded for training (§4).

**The window is on the laptop.** The lab's monitors are too small for it: it needs
about 1280×850 and the benches have small screens, which would cut off the balance
banner at the bottom. The board does all the work; the laptop only shows the
picture and sends the clicks back over the USB-C cable.

**Bring:** the board and its 12 V adapter (push the barrel plug fully home — it has
come loose twice); the camera dongle; the laptop and the USB-C cable; a phone on a
stand for the video; a stopwatch.

**Before leaving:** the laptop is charged.

**At the bench:**
1. Board's USB-C into the laptop: `adb shell "date -s @$(date +%s)"` (no RTC: the
   clock is wrong after every power loss, and session folders are named by it).
2. Dongle into the board's **USB 2.0** port (the USB 3.0 port fouls the USB-C plug).
3. On the laptop: `python edge/live_viewer.py --record`. It starts the session on the
   board and opens its window here. Keep droplets out for the first 30 s — it learns
   the background.
4. Per unit: spray, click a droplet, balance, measure as usual; time a few of the
   same falls with the stopwatch and note them by fall number. `q` ends the session.
5. Film the screen and the apparatus together with the phone. The screen recording
   and the board's session folder (falls, report) land in `data/live/`.

**Then raw recordings, ~25 min, the dongle moved to the laptop** — so that later
versions of the pipeline (the C++ ports) can be measured on a demo unit, and the
dongle's MJPG mode scored, at home, with no further lab visit. In the parent repo:
6. `python cv/UVC_check.py` (which index is the dongle), then
   `python test/tools/capture_bench.py` (~1.5 min, the chamber may be empty). Its
   default row is DirectShow, its `msmf` rows Media Foundation: expect raw YUY2 at
   ~23.6 fps and MJPG at ~30. Every recording so far is the raw one.
7. One demo unit, raw: `$env:MILLIKAN_CAPTURE="dshow"` then
   `python test/tools/raw_capture.py unit<r-p>_1027_demo <index>` — chamber empty
   through the 60 s learn, then spray as usual, ~5 min, `q`. For replay only, never
   training.
8. Unit 1-3 or 3-2 (never trained on, not reserved), back to back with the same
   spray routine, ~5 min each: raw as in 7 (`unit<r-p>_1027_yuyv`), then
   `$env:MILLIKAN_CAPTURE="mjpg"` (Media Foundation) and `unit<r-p>_1027_mjpg`.
   ~2 GB each. Check each folder's delivered rate before moving on.

**If the laptop route fails:** the board's own screen still works — a monitor of
1920×1080 or more on HDMI and a USB mouse, then double-click "Millikan live" once
(tested on real HDMI 2026-10-17).

### Live on the reserved units (2026-10-27) — `results/live/README.md`

- **Units 1-1 and 3-3, live on the board** (2-2 and 4-4 were optional and skipped):
  5 and 6 legal falls, 21.5 and 21.3 updates a second. The camera set the pace — the
  raw mode delivered ~21.5 of 25 frames a second while the board spent 17–20 ms a
  frame. The operator timed each fall from a phone video of the screen, blind to the
  board's numbers: every fall within 0.87 s, on average 0.53 s (1-1, 19 s falls) and
  0.35 s (3-3, 34 s falls); the board's is the shorter in 9 of 10.
- **Raw recordings, all at full rate with no frame dropped**, on D: and linked into
  the parent's `test/fixtures/`: `unit3-3_1027_demo` (23.6 fps, a reserved unit: replay
  only, never training — for measuring later versions, e.g. the C++ ports);
  `unit3-2_1027_yuyv` (23.6 fps) and `unit3-2_1027_mjpg` (30.0 fps), the MJPG test.
  For the blind labelling the pair is linked a second time as `unit3-2_1027_A` and
  `_B` (common.FIXTURE_INFO role "mjpg"), which is which revealed after the review.
- **What the day taught:** the board's USB link to the laptop did not come up twice
  after it was replugged; switching the board off and on fixed it both times. Under
  Media Foundation (`MILLIKAN_CAPTURE=mjpg`) `raw_capture.py`'s device 0 opened the
  laptop's own webcam — pass device 1. A droplet drifting through the 60 s learn spoils
  it; waiting a few minutes after a spray before starting is cheaper than restarting.
- **MJPG, a first look without labels** (the pipeline with the FP32 classifier over
  both whole recordings): falls and the panel's voltage came through MJPG as through
  raw — 11 legal falls in each. But MJPG frames carry more candidate blobs (15.7 a
  frame vs 11.2) and the classifier kept 42% of them against 89%; droplet-like blobs
  stand out less (median peak 51 vs 60 grey levels over background). Whether it is
  rejecting compression artefacts or real droplets is what the hand labels decide.

### MJPG against raw, hand-labelled (2026-10-28) — `results/mjpg/SCORE.md`

- **The test:** 100 frames of each 3-2 recording, labelled blind as A and B (assigned
  at random; A = MJPG, B = raw), with the blind second look on every disagreement
  (1,561 and 670 spots); 41 and 42 minutes of labelling.
- **The classical detector breaks on MJPG:** precision 84.6% → 37.5%, false
  detections 1.86 → 11.47 a frame — compression artefacts pass its gates.
- **The classifier holds:** INT8 (simulator) F1 85.4% raw, 85.1% MJPG; precision
  93.3% → 97.9%, false 0.70 → 0.14 a frame; recall 78.7% → 75.3%, the intervals
  overlapping (74.9–82.5 vs 72.6–78.4). FP32 the same within half a point; INT8 and
  FP32 agree on 99.53% of the blobs in each. So the camera's faster mode is usable
  only with the classifier, for ~3 points of recall.
- **On the board (2026-10-28, no apparatus):** with `MILLIKAN_CAPTURE=mjpg` the
  board's dongle switches to MJPG and delivers 30.0 fps (300 frames); decoding one of
  these frames as JPEG costs the board ~5 ms, reading it as PNG 9.2 ms. The 27 Oct
  live sessions spent 17–20 ms a frame in the pipeline and ~30 ms waiting for the raw
  camera, so a 33 ms MJPG frame should leave room — a prediction for the live run
  (`python edge/live_viewer.py --capture mjpg`). A replay through the board's own
  desktop window managed 20.5 updates a second on this recording, but that path also
  draws the window on the board and reads PNGs; it is not the live viewer's.
- **Live on MJPG (2026-10-29), unit 2-2, the third reserved unit:** 27.6 updates a
  second against 21.3–21.5 raw (+29%), 6 legal falls (`results/live/README.md`). Not
  30: the loop waits for a frame (~21 ms), then processes it (~16 ms), in turn, so about
  one frame in ten is missed. Capturing in a thread of its own would overlap the two —
  the first step of the owner's C++ study below. The first attempt refused to start
  ("no camera showing the grid"): the apparatus was not cabled to the converter, whose
  solid blue no-signal screen is what the board saw.

### Next, needing only minutes of the owner

- ~~WSL2 and the toolchain~~ — done 2026-09-30 (§8).
- ~~Track review~~ — done 2026-10-09 (`data/review/REPORT.md`).
- ~~**The hunt aid's suggestions**~~ — reviewed 2026-10-17: no reduction in wrong
  suggestions (§11, "Beyond accuracy").
- ~~**The board on a monitor**~~ — done 2026-10-17, on real HDMI through the capture
  card (§11, "Beyond accuracy").
- ~~**Power**~~ — done 2026-10-29: the board 2.9 W in the live view, the laptop
  15–17 W (§11, "Power"; `results/power`).
- **Later, the owner's C++ study (winter break):** first a capture thread (frames read
  in their own thread, stamped when captured, so the 21 ms wait overlaps the 16 ms of
  processing — MJPG live toward 30 updates a second); then the per-track loop (tracker and
  the hunt aid's per-track tests) and the panel digit reader in C++, installed from
  outside `cv/` like `fast.py` and held to the same proof (`fast.py check`, `exact`).
  Expected: the dense spray from ~16 to ~20–23 updates a second, the fall-end stall
  gone; cut-off 15 Feb 2027, after which Claude ports them and the owner reads and
  defends them. Then the dongle's MJPG mode (30 fps), scored against hand labels
  on the raw/MJPG pair recorded on the demo trip before the live path uses it —
  not before, as every recording so far is raw.
- **Live, not only replayed (2026-09-30).** The fork's claim is real-time on an edge
  device, so replays alone undersell it. The plan:
  - **Recorded fixtures stay the source of every published number** — they are what
    makes the table reproducible by someone else.
  - **No webcam.** The live path accepts a camera only if its picture shows the
    reticle (`LiveSource.accept` = `autoROI`), so a webcam cannot run the pipeline
    live; it could only time capture.
  - ~~**A short lab trip after the owner's vacation, for the dongle.**~~ Done
    2026-10-11 (§11, "Live on the board"). The apparatus's BNC composite output
    (视频输出) reaches the laptop through a converter that Windows recognised with
    no driver installed — which on Windows means USB video class, which the board's
    kernel has built in (§8.1). Very likely fine, never tried; the dongle's model is
    not recorded anywhere. Take the board, its 12 V adapter, the laptop and the USB-C
    cable; dongle into the board's USB 3.0 port; over ADB: `v4l2-ctl --list-devices`,
    `--list-formats-ext`, real FPS per format (MJPG vs YUYV), `autoROI` accepts the
    frame, a short live run. ~30 min. Optional, same trip: record raw sessions with
    droplets (`test/tools/raw_capture.py`) — positives currently come from only two
    days (§11).
  - ~~**One lab session at the end**~~ — done 2026-10-27 (§11, "Live on the reserved
    units").
