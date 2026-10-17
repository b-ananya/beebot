# Data contract (freezes at hour 2)

Changes are announced in team chat, never made silently. `python -I scripts/validate.py` checks it.

Conventions: time in seconds from clip start. Boxes are `[x1, y1, x2, y2]` normalized to 0–1000,
origin top-left (the space Mk1.5 returns). A gap stays a gap: never invent observations.

## clips/manifest.json (B → everyone)

A list of:

```json
{"clip_id": "hb1_w00", "path": "clips/hb1_w00.mp4", "fps": 30, "width": 1280, "height": 720,
 "duration_s": 10.0, "source": "Monash honeybee Experiment_1 bee_test_1.mp4",
 "source_video": "bee_test_1.mp4", "source_start_s": 0.0,
 "license": "CC BY 4.0, Monash University (cite dataset DOI)",
 "has_truth": true, "truth_kind": "HyDaT tracker output"}
```

`file_id` is added by A once clips are uploaded to the Perceptron Files API.

## tracks/<clip_id>.json (A → B, C, D) and truth/<clip_id>.json (B → B, D)

```json
{"clip_id": "sm3_12_123903", "model": "perceptron-mk1.5", "prompt_version": "v3",
 "tracks": [{"track_id": "honeybee_1", "species": "honeybee", "complete": true,
             "points": [{"t": 0.5, "box": [410, 520, 450, 570], "center": [430, 545],
                         "interpolated": false, "label": "foraging"}]}],
 "flowers": [{"flower_id": "flower_3", "box": [560, 300, 700, 440]}],
 "events": [{"track_id": "honeybee_1", "flower_id": "flower_3", "type": "visit",
             "t_start": 2.17, "t_end": 9.5}]}
```

| Field | Required | Notes |
|---|---|---|
| `points[].t`, `points[].box` | yes | sorted by `t` |
| `points[].center` | no | eval uses it when present, else box center |
| `points[].interpolated` | no, default false | true for client-side gap filling; eval excludes these |
| `points[].label` | no | behavior label for the overlay: `foraging`, `flying`, `hovering`, `walking` |
| `tracks[].species` | no | `honeybee`, `syrphidae`, `lepidoptera`, `vespidae`, `unknown` |
| `events[].type` | yes | one of `visit`, `landing`, `hover`, `takeoff` |

`model` is `ground_truth` in `truth/`, `mock` in `tracks/mock/`.

**Truth caveat (say this on stage):** truth comes from the dataset authors' trackers (HyDaT,
Polytrack), not hand labels, and boxes are estimated from blob area. Center error against it
is *agreement with a reference tracker*. `Experiment1-Results.xlsx` has human-counted visible
frames per honeybee video, which is real ground truth for coverage.

## rollouts/<clip_id>.json (C → D)

```json
{"clip_id": "hb1_w00", "source_track": "bee_1", "success": true, "time_to_flower_s": 4.1,
 "mean_path_error": 0.03, "time_scale": 4.0, "video": "rollouts/hb1_w00.mp4"}
```

## eval/report.json (B → D)

```json
{"clips": 46, "median_center_error": 18, "track_coverage": 0.82,
 "visit_recall": 0.75, "visit_precision": 0.80, "notes": "errors worst under occlusion"}
```
