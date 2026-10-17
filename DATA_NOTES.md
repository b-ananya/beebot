# Data notes

## Sources (both CC BY 4.0, credit on the attribution slide)

| Prefix | Dataset | What we use |
|---|---|---|
| `hb*` | Ratnayake, Dyer, Dorin (2020). *Honeybee video tracking data.* Monash University. [doi:10.26180/5f4c8d5815940](https://doi.org/10.26180/5f4c8d5815940) | `Experiment_Data.zip` → Experiment_1: 7 test videos (1920×1080, 60 fps, 27–71 s, Scaevola ground cover), HyDaT per-frame tracks, `Experiment1-Results.xlsx` (human-counted occluded/visible frames) |
| `sm*` | Ratnayake, Amarathunga, Zaman, Dyer, Dorin (2022). *Spatial Monitoring and Insect Behavioural Analysis Dataset.* Monash University. [doi:10.26180/21533760](https://doi.org/10.26180/21533760) | `Insect_Trajectory_Dataset.zip` (2,335 Polytrack tracks, flower positions, visits) + test videos T6 and T7 (overhead strawberry farm, 1920×1080, 30 fps, 10 min) |

Skipped: YOLO training sets (no fine-tuning in this track), full honeybee videos (24 GB), the other 8 farm videos.
The source videos behind most of the 2,335 farm tracks are not public ("available upon request").

## Working set (`python -I scripts/prepare_data.py`)

- 48 clips at 1280×720, 30 fps, 104 MB total.
- `hb{video}_w{window}`: 38 consecutive 10 s windows over the 7 honeybee videos. One bee per video. 2 windows have no truth points (bee absent or occluded).
- `sm{cam}_{day}_{HHMMSS}`: 10 clips cut around farm tracks that fall inside T6/T7. These have flower boxes and 14 visit events. Species from the filename digit (T6 tracks are vespids, T7 mostly honeybees, one lepidoptera).

## How truth was built

- Pixels → 0–1000: `x / width * 1000`. Honeybee time: `(nframe - 1) / 60 - window_start`.
- Farm time: `absframe` counts frames across a day's back-to-back 10-min videos; `absframe mod 18000 / 30` gives the offset into the video. This agrees with filename timestamps to within about 1.3 s; we use the `absframe` value. `manifest.json` records both as `offset_check`.
- Only frames the tracker recorded become points. Occluded frames (`method == "Non"`) stay gaps.
- Trackers give blob area, not boxes: box is a square of side `1.6 × sqrt(area)`, clamped 16–200 px.
- Verified visually on 3 clips (bee inside truth box, flower boxes on flowers): see `eval/checks/` after running the check render.

## Things to know

- Bees are about 20–30 px wide at 720p. A should test native 1080p (`raw/`) early in A2.
- Truth is a reference tracker, not hand labels. Report "agreement with HyDaT/Polytrack" for center error, and use the xlsx visible-frame counts for coverage.
- The farm offset is accurate to roughly ±1 s for later videos in a day; T6/T7 visual checks looked aligned.

## Farm truth timing fix (scripts/fix_truth_offsets.py)

The `absframe` timing above was off by a constant per source video: **-0.3 s** for cam 2 day 08 and **+1.4 s** for
cam 3 day 12 (close to the filename-based estimate in `offset_check`). Found by sweeping the shift that maximizes
agreement with the motion tracker (which never sees truth): coverage goes from ~30% to >90%, with a median error of
2-3 units. Confirmed visually: after the shift the bee sits in the truth box, before it the box is on empty leaves
(`eval/checks/offset_sm3_12_124027.jpg`). One offset per video, the median over its 5 clips, so a single clip can't
move it. `truth/sm*.json` now carry `time_offset_s`. `sm3_12_123633` still agrees poorly at any shift (worth a look).
