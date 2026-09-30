# Retail Behavioural Analytics — Computer Vision Take-Home

An end-to-end video analytics pipeline that measures customer interest, shelf
engagement, and staff–customer interaction from two fixed CCTV clips of a mall
footwear store.

Everything runs locally. No footage is sent to any external service.

> **About this repo.** This is a portfolio copy of a computer-vision take-home
> assessment. The source footage, hand-annotated zone geometry
> (`configs/*_zones.json`), and per-run outputs (`outputs/`) are excluded
> because they're derived from an employer's confidential footage — the code
> and the engineering decisions below are my own and are shared as-is. The
> results reported in §1 are real numbers produced by this pipeline on that
> footage; reproducing them requires supplying your own equivalent video and
> re-annotating zones with `src/annotate_zones.py`.

---

## 1. Results

### Task 1 — Store Interest and Conversion (`entrance.mp4`)

| Metric | Count |
|---|---:|
| Total Interested | 17 |
| Interested Entered | 1 |
| Interested Passed By | 16 |

Interest-cue breakdown (a track may satisfy several): `slowing` 13,
`approaching` 3, `touching_display` 1, `entered` 1. `facing` appears on 15
tracks but only ever as corroboration — see §4.1.

### Task 2 — Per-Shelf Customer Interest (`interior.mp4`)

| Shelf | Interest events | Unsure |
|---|---:|---:|
| rack_A | 4 | 3 |
| rack_B | 1 | 1 |
| rack_C | 0 | 0 |
| rack_D | 3 | 1 |
| rack_E | 0 | 1 |
| rack_F | 0 | 0 |
| rack_G | 1 | 0 |
| rack_H | 1 | 0 |
| **Total** | **10** | **6** |

The same analysis was also run on `entrance.mp4` (9 sure, 4 unsure across 13
storefront racks). The brief assigns Task 2 to `interior.mp4`; the entrance
figures are supplementary.

### Task 3 — Staff–Customer Interaction (`entrance.mp4`)

| Staff instance | Interaction sessions |
|---|---:|
| Staff 1 | 16 |
| **Average sessions per staff member** | **16.0** |

Session durations: 2.5s min, 4.3s median, 86.3s max. The longest corresponds to
a continuous 86s stretch in which the staff member is recognised without a break
(two adjacent tracks spanning 256s–342s) — consistent with fitting shoes for one
customer.

---

## 2. Setup

Requires Python 3.10+, and an NVIDIA GPU for reasonable runtime (CPU works but
the tracking pass is much slower). Developed on Windows with an RTX 4060.

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1               # Windows
# source venv/bin/activate                # Linux/macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

PyTorch is installed separately because the CUDA build lives on PyTorch's own
index, not PyPI. `requirements.txt` therefore excludes torch/torchvision.

Model weights download automatically on first use, or fetch them explicitly:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m-pose.pt')"
```

Place your own videos at `data/raw_videos/entrance.mp4` and
`data/raw_videos/interior.mp4` (not included in this repo — see the note at
the top of this README).

## 3. Running

The tracking pass is expensive and everything else reads its cached output, so
build the cache once:

```bash
python src/build_cache.py                     # ~8 min for both videos on a 4060
```

Then generate results:

```bash
python src/task1_interest_conversion.py       # Task 1 CSVs
python src/task2_shelf_interest.py --video interior
python src/task2_shelf_interest.py --video entrance
python src/task3_staff_interaction.py         # Task 3 CSVs
python src/render_entrance_video.py           # entrance video (Tasks 1 + 3)
python src/render_task2_video.py              # interior video (Task 2)
```

Or reproduce everything in a container:

```bash
docker build -t hendricks-analytics .
docker run --gpus all   -v "$(pwd)/data:/app/data"   -v "$(pwd)/outputs:/app/outputs"   hendricks-analytics
```

Supporting analysis (regenerates the evidence behind the thresholds):

```bash
python src/analyse_thresholds.py              # distributions + sensitivity sweep
python src/compare_trackers.py                # tracker benchmark
python src/analyse_apron_colour.py            # apron colour verification
python src/calibrate_staff.py                 # one-class threshold sweep
python src/test_episode_detector.py           # 9 behavioural tests
```

Scene geometry is hand-annotated once per video (the camera is fixed). This
repo ships without `configs/*_zones.json` since the coordinates are only
meaningful for the original, confidential footage — regenerate them for your
own video with:

```bash
python src/annotate_zones.py data/raw_videos/entrance.mp4 --frame 100
```

---

## 4. Method

```
video
  └─> YOLOv8m-pose  ......... person boxes + 17 COCO keypoints
        └─> BoT-SORT ........ per-frame boxes linked into tracks
              └─> per-track grouping .. trajectory, speed, signal history
                    └─> EpisodeDetector ...... temporal event rules
                          └─> task logic ..... CSV + annotated video
```

Each layer answers one question and nothing more. Detection answers *where are
the people*; tracking answers *is this the same person as last frame*; the
episode layer answers *how many distinct events*; task logic answers *what does
this mean for the business metric*. Keeping them separate is what let the
tracker be swapped on evidence (§5) without touching any event logic.

### 4.1 Task 1 — interest and conversion

**Scene model.** The storefront is open-plan: several walkable openings
separated by fixtures, no door. Three `Entry` segments are annotated across
those openings. They proved near-collinear — extending the main segment
predicts the far segment's start to within 1px — so a PCA best-fit line through
all their endpoints defines the **storefront axis**, used for "which side is a
person on, and how far from the front". Crossings are tested against the drawn
**segments**, never the infinite line, so someone walking past a solid fixture
cannot register an entry.

**Interest cues.** Any one qualifies, all evaluated only while the person is
outside and within the attention band:

| Cue | Rule | Rationale |
|---|---|---|
| `slowing` | reached ≥40px/s, then dropped ≤15px/s | Deceleration in front of a shop is the classic browse signal, and the cue the brief names first. Requiring both ends of the transition avoids flagging someone who was simply stationary throughout. |
| `lingering` | ≥2.0s in the band **and** median speed ≤15px/s | Catches someone who stops rather than merely slows. Fired 0 times — nobody actually stops in this corridor. |
| `approaching` | closed ≥40px toward the front, ended within 60px | Direction-aware; separates walking up to the shop from passing at constant distance. |
| `touching_display` | ≥15 frames with a wrist inside a storefront rack, while outside | Strongest available evidence — the person handled the merchandise without stepping in. Several entrance racks face the walkway within arm's reach. |
| `facing` | ≥10 frames with a visible face | **Corroboration only, never sufficient.** See below. |

**Why `facing` cannot stand alone.** The camera looks *outward* from inside the
shop, so anyone walking down the corridor toward it presents a detectable face.
That is a consequence of camera geometry, not of attention. Used as a standalone
cue it flagged 61% of all passers-by as interested; it now only decorates a
track that already met a movement-based cue.

**Passer-by gate.** Interest cues describe someone *outside* deciding whether to
come in, so they may only be evaluated for a track that genuinely is a
passer-by. A track must spend ≥50% of its life more than 15px outside, for at
least 1.5s. Tracks that entered are exempt.

This gate exists because of a real failure found by reviewing the output video:
track 367 sat inside the store for **4.6 minutes** (median distance +180px, i.e.
deep indoors) but 1.74% of its frames dipped marginally outside through
bounding-box jitter at the threshold — enough to fire `slowing` and label a
seated customer an interested passer-by. Adding the gate dropped Total
Interested from 25 to 16.

**Entered.** Requires the track to start outside, end inside, and pass through
an annotated opening. An earlier version demanded a sign flip across the
storefront axis **and** a segment intersection on the *same frame pair*; measured
on this footage that never once co-occurred (6 tracks flipped sign, 7 crossed a
segment, zero did both), because the global best-fit axis and the individual
short openings do not align closely enough. The trajectory-level test also
absorbs box jitter at the threshold.

### 4.2 Task 2 — per-shelf interest

**Interest is evidenced by a wrist entering a shelf region**, not by body
proximity. Measured on this footage the wrist signal fires **2.7× more often**
than a foot-position test (22.0% vs 8.1% of person-observations), because
shoppers stand *beside* a rack and reach *into* it. A body-position rule misses
most genuine engagement.

This also resolves the brief's "customer between two shelves" case directly:
whichever region the wrist is observed in is the shelf being engaged with, and
where regions overlap the nearest centroid wins. The hand decides, not the body.

**Evidence hierarchy.** Every event carries its evidence type:

| Verdict | Evidence |
|---|---|
| **sure** | a wrist was confirmed inside the region for ≥1.5s |
| **unsure** | the person stood inside the region >4.0s but their wrist was never confirmed there |

Unsure events are reported in a separate column, never folded into the headline
count, so the primary metric rests only on confirmed evidence. They are not
discarded either: they surfaced 6 engagements that the wrist signal alone missed
entirely, including one at `rack_E` which otherwise showed zero.

A mid-episode keypoint dropout does **not** downgrade a sure event. A dropout is
a sensing failure, not counter-evidence; one long enough to matter would have
exceeded the gap tolerance and ended the episode anyway.

---

## 5. Key decisions, and the evidence for them

### 5.1 Pretrained models, no fine-tuning

The brief permits pretrained weights and states that perfect accuracy is not
expected; 45% of the rubric is reasoning and methodology. Person detection and
multi-object tracking are mature solved problems — the engineering value here is
in selecting and integrating them correctly and in the behavioural logic built
on top, not in retraining a detector on 15 minutes of footage.

`yolov8m-pose` serves as the single model: it returns person boxes **and** 17
keypoints in one forward pass, so the wrist signal costs no extra inference over
plain detection. Size `m` was chosen as an accuracy/runtime compromise for
offline batch processing on an 8GB GPU.

### 5.2 Tracker chosen by benchmark, not reputation

`src/compare_trackers.py` ran three configurations over both videos:

| Tracker | entrance IDs | frag. pairs | interior IDs | frag. pairs |
|---|---:|---:|---:|---:|
| ByteTrack | 513 | 174 | 338 | 121 |
| BoT-SORT, ReID off | 487 | 142 | 311 | 97 |
| BoT-SORT, ReID **on** | 634 | 249 | 380 | 134 |
| **BoT-SORT, ReID off, gmc=none** | **260** | — | **161** | — |

Two findings worth stating plainly:

- **Enabling ReID made fragmentation worse**, not better. With `model: auto`
  the appearance features are weak enough that they mis-associate
  similarly-dressed shoppers, creating more identity switches than motion
  matching alone.
- **Disabling global motion compensation halved the track count.** GMC corrects
  for *camera* movement; both clips come from a fixed mount, so estimating
  camera motion was pure overhead injecting noise into association.

Caveat: the final row also changed the detector (`yolov8m` → `yolov8m-pose`), so
the improvement cannot be attributed cleanly to `gmc_method` alone without a
controlled re-run.

### 5.3 Thresholds derived from measured distributions

No threshold in `configs/thresholds.yaml` was chosen by intuition; each carries
its evidence in a comment. Two examples:

**Shelf `min_duration_s` = 1.5s.** Raw wrist-in-rack run lengths show a clear
noise floor — 75% of runs last under 0.5s (median 0.07s, one or two frames),
which is a wrist clipping a polygon edge while walking past. Genuine engagement
sits in the tail (p90 = 1.47s, p95 = 3.72s).

**Shelf `max_gap_s` = 4.0s.** Sweeping this parameter:

| max_gap | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 7.0 | 10.0 |
|---|---|---|---|---|---|---|---|---|
| events | 12 | 12 | 12 | 11 | 10 | 10 | 10 | 10 |

The count **plateaus from 4.0s onward** — identical totals and identical
per-shelf breakdown all the way to 10s. That flatness is the justification:
every genuine return visit in this footage is separated by more than 10 seconds,
so there is an empty band between "hand withdrawn mid-browse" (≤4s) and "walked
away and came back" (>10s). Nothing ambiguous falls inside it. 4.0s sits at the
start of the plateau, so the choice tolerates being wrong by 2× without changing
any reported number. The whole metric moves by only 2 events across a 20× sweep.

### 5.4 Counting rules are unit-tested

`src/test_episode_detector.py` encodes each of the brief's counting
requirements as an executable test (9/9 passing) — e.g. "continuously
interacting … counted as one interest event" and "returning after the previous
interaction has ended … counted as a new event". The counting logic is the part
most likely to be quietly wrong, so it is pinned down rather than assumed.

---

## 6. Task 3 — staff identification and interaction

### 6.1 Identifying staff by apron colour

Staff wear a **red apron**. The colour was verified before relying on it, from
manually curated reference crops: hue is red with **0.97 agreement across 36
independent entrance tracks** and 0.95 across 14 interior tracks, and it reads
the same from front and back — which matters because staff spend much of their
interaction time kneeling with their back to the camera, exactly when Task 3
needs to recognise them.

Two mistakes had to be corrected before the signal worked, and both are
instructive:

**Mistake 1 — measuring the box instead of the garment.** The first version
sampled a horizontal band of the bounding box as "the torso". Measured result:

| Torso extraction | Verified staff | Random population |
|---|---:|---:|
| Bounding-box fraction band | 56% red | 49% red |

Useless. A person's box is mostly background, and this store's warm wood
shelving under warm lighting occupies the same hue range as the apron — so the
measurement described the scene, not the uniform. The fix is to build the torso
from **shoulder and hip keypoints**, which the pose model already provides at no
extra inference cost. That polygon contains garment and little else.

**Mistake 2 — an unjustified threshold.** The initial 0.30 was a guess, and it
labelled 105 of 201 tracks as staff — impossible when there is one staff member.

### 6.2 One-class calibration

There were no confirmed *customer* crops to calibrate against; the curated set
contains only verified staff. But a decisive test exists anyway, because the
**ground-truth count is known**: exactly one staff member appears in this
footage. Any threshold labelling far more tracks than one person can account for
is refuted without needing a single negative label.

Sweeping the threshold (`src/calibrate_staff.py`):

| threshold | verified-staff recall | tracks labelled staff |
|---|---:|---:|
| 0.30 | 97% | 105 ← the original guess, refuted |
| 0.70 | 89% | 41 |
| **0.80** | **76%** | **24** ← chosen |
| 0.85 | 65% | 14 |
| 0.95 | 46% | 1 |

0.80 retains three quarters of the verified staff crops while labelling a track
count consistent with one person under this tracker's fragmentation.

### 6.2b Aggregation, and enforcing the one-staff constraint

Two further corrections were needed, both found by watching the output video and
seeing the staff member labelled CUSTOMER.

**The median was the wrong statistic.** The apron is only clearly visible for
part of any track — the staff member turns, bends to fit a shoe, and is partly
occluded by the customer being served — so a median sits below threshold and the
track is called a customer. A second ground-truth anchor makes the cost
measurable: the dense extraction of the 36 verified staff tracks saved 4,427
frames, i.e. **≈148 seconds of staff screen time**.

| aggregation | tracks | staff seconds | overlapping pairs |
|---|---:|---:|---:|
| median ≥ 0.80 | 24 | 32 | 0 |
| p70 ≥ 0.80 | 31 | 161 | 9 |
| p75 ≥ 0.80 | 34 | 181 | 10 |
| p60 ≥ 0.70 | 49 | 433 | 45 |

**No threshold satisfies both goals.** Settings recovering realistic coverage
also produce staff-labelled tracks that overlap *in time* — and two staff cannot
be on screen at once when there is one. Overlap count is therefore a precision
metric requiring no labels at all.

So the constraint is **enforced** rather than hoped for: among staff candidates
overlapping in time, the one with the most total apron evidence
(score × observed frames) is kept and the others are demoted to customer.
Ranking by evidence rather than by score alone matters — score-only ranking let
a 0.2s track scoring 0.99 evict a 30s track scoring 0.85, costing 88s of
correctly identified staff time.

| method | staff seconds | overlaps |
|---|---:|---:|
| median ≥ 0.80 (original) | 32 | 0 |
| best clean threshold | 43 | 0 |
| p70 + score tie-break | 73 | 0 |
| **p70 + evidence-weighted tie-break** | **127** | **0** |

127s against a 148s ground truth — 86% of the staff's actual screen time, with
zero internal contradictions.

### 6.2c The single-staff constraint applies to entrance only

The constraint above is a claim about the scene, so it was tested per video
rather than assumed globally. The test: do two tracks that BOTH score ≥0.90
(near-total apron coverage) appear concurrently for more than 2 seconds? Weak
overlaps prove nothing — a marginal track may simply be a customer in warm
clothing — but two confidently-aproned people on screen at once cannot be one
person.

| video | concurrent high-confidence staff pairs | conclusion |
|---|---|---|
| `entrance.mp4` | none | one staff member; constraint applied |
| `interior.mp4` | several, 20+ seconds each at 0.98–0.99 | multiple staff; constraint NOT applied |

`interior.mp4` includes the service counter, and staff work it while another
serves customers on the floor. Applying the constraint there demoted tracks
scoring 0.99 for 23 seconds to "customer" and visibly mislabelled staff on
screen — caught by reviewing the output video, not by any metric.

Task 3 is computed on `entrance.mp4` only, where the single-staff assumption is
supported by this test rather than merely asserted. Staff labelling in the
interior video is presentational (Task 2 does not use it), so there the
threshold alone decides.

### 6.3 Merging fragments into one staff instance

The 24 staff tracks were then checked for temporal overlap, because if two
genuinely overlap they cannot be the same person and merging would be wrong.

**Zero pairs overlap by more than 0.5s**, and several are directly adjacent —
track 605 ends at 248.8s and track 612 begins at 248.8s. That is the signature
of one person repeatedly re-acquired under new IDs.

They are therefore merged into a single logical staff instance. The brief
permits either treatment ("if they leave the view and later re-enter, they may
be treated as a new instance … re-identification across separate appearances is
optional"), but reporting 24 instances would divide one person's sessions across
24 rows and yield an average describing nothing real. The overlap check is
re-run and printed every time the task executes, so the assumption stays visible
rather than buried.

### 6.4 What counts as an interaction

| Rule | Value | Rationale |
|---|---|---|
| proximity | ≤1.2 × staff bbox height | A **ratio**, not raw pixels. This camera has strong perspective — a person at the back is a third the pixel height of one at the front — so a fixed radius would mean metres deep in the scene and centimetres near the lens. Box height is a per-frame scale proxy, so the rule holds a constant *physical* distance. |
| duration | ≥2.0s | Brushing past a customer en route to a shelf is not an interaction. |
| session gap | 4.0s | Stepping away to fetch a shoe box and returning is one session; a longer gap closes it, so a later return counts separately, as the brief requires. |

The proximity ratio was measured, not guessed: staff-to-customer gaps normalised
by staff box height are p5=0.64, p10=0.80, p25=1.10, p50=2.61. The distribution
has a dense cluster below ≈1.1 (people sharing a space) and a long tail beyond
(people merely both in frame). 1.2 sits just above that cluster.

---

## 7. Assumptions, limitations, failure cases

**Entries can only be observed if the person was detected while still outside.**
The single detected entry (track 414) shows a clean walk-in — threshold to deep
inside over ~6s, then settling. But the detector does not pick people up until
they are close enough to be sizeable, so anyone first detected once already
inside cannot be counted. This likely contributes to the low conversion figure.

**The low entry count is nonetheless real, not a tracking artefact.** This was
checked directly: of 80 tracks ending outside, only **1** has a plausible
continuation starting inside (≤2s, ≤150px). If entries were being lost to
identity fragmentation at the threshold there would be dozens of such pairs.
Corridor traffic in this clip genuinely walks past.

**Track fragmentation inflates unique-person counts.** Even the best tracker
produced 260 entrance tracks for 7.5 minutes. Short-lived fragments are filtered
by a minimum track duration, but the counts should be read as "distinct observed
appearances", not a guaranteed unique-person count.

**A long occlusion spanning a genuine departure and return merges into one
event.** With a single fixed view this is unresolvable; multi-camera coverage
would resolve it properly. Documented rather than hidden.

**Zone geometry is hand-annotated and camera-specific.** Everything in
`configs/*_zones.json` is valid only for these two fixed viewpoints. A moved
camera invalidates all of it, and `gmc_method: none` would need reverting for
any PTZ or handheld source.

**Interest cue thresholds are tuned to this scene's pixel scale.** Speeds are in
px/s and distances in px, both of which depend on camera height and lens. They
would need re-deriving — not merely re-tuning — for a different installation.

---

## 8. Project structure

```
README.md
Dockerfile
requirements.txt
configs/
  thresholds.yaml              every threshold, with its evidence
  botsort_static.yaml          tracker config (benchmark-selected)
  botsort_reid_static.yaml     ReID-enabled config used only for the benchmark
  entrance_zones.json          hand-annotated entry lines + racks (not included — see top note)
  interior_zones.json          (not included — see top note)
src/
  build_cache.py               one-time tracking pass -> cached observations
  tracking.py                  YOLOv8m-pose + BoT-SORT wrapper
  zones.py                     polygons, boundary lines, geometric predicates
  episodes.py                  EpisodeDetector -- temporal event rules
  task1_interest_conversion.py Task 1 logic + CSVs
  task2_shelf_interest.py      Task 2 logic + CSVs
  task3_staff_interaction.py   Task 3 logic + CSVs
  render_entrance_video.py     entrance deliverable video
  render_task2_video.py        interior deliverable video
  annotate_zones.py            interactive zone annotation tool
  analyse_thresholds.py        distributions + sensitivity sweeps
  analyse_apron_colour.py      apron colour verification
  staff_id.py                  apron scoring + track-level staff classification
  calibrate_staff.py           one-class threshold calibration
  compare_trackers.py          tracker benchmark
  test_episode_detector.py     counting-rule tests
outputs/                        generated locally when you run the pipeline
  task1_store_interest.csv       (not included — derived from confidential footage)
  task1_interest_detail.csv
  task2_interior_shelf_interest.csv
  task2_interior_shelf_events_detail.csv
  task2_entrance_shelf_interest.csv
  task2_entrance_shelf_events_detail.csv
  task3_staff_interaction.csv
  task3_interaction_detail.csv
  entrance_annotated.mp4
  task2_interior_annotated.mp4
data/
  raw_videos/                  input videos (bring your own — not included)
  cache/                       cached tracking output (generated locally)
models/
  yolov8m-pose.pt              (downloaded automatically — see §2)
```

Every CSV has a `*_detail.csv` companion listing individual events with
timings, evidence type, and confidence, so any headline number can be traced
back to the observations that produced it.

## 9. Tools used

| Component | Choice |
|---|---|
| Detection + pose | Ultralytics YOLOv8m-pose (COCO-pretrained) |
| Tracking | BoT-SORT (Ultralytics), ReID off, GMC off |
| Video/image | OpenCV |
| Numerics | NumPy |
| Config | PyYAML |
| Runtime | PyTorch (CUDA 12.1) |
