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

## search/index.json (B → D)

A list of searchable moments, one per event or description, built once from Mk1.5 output.

```json
{"clip_id": "sm3_12_123903", "t_start": 2.17, "t_end": 9.5, "kind": "event",
 "text": "honeybee lands on flower_3 and forages", "source": "mk1.5"}
```

`kind` is `event` (from `events`) or `caption` (Mk1.5 description of a time window).
Live queries return the same shape with `"source": "mk1.5-live"`.

## field/<field_id>.json (B → C, D)

`field_id` is a `clip_id` for one camera view, or `farm_day<DD>` for the 9-station farm.

```json
{"field_id": "sm3_12_123903", "source": "mk1.5", "units": "norm1000", "target_visits": 4,
 "flowers": [{"flower_id": "flower_3", "center": [520, 310], "radius": 25, "state": "open",
              "visits": 2, "dwell_s": 7.3, "deficit": 2}]}
```

| Field | Notes |
|---|---|
| `source` | `mk1.5` or `truth`; never mix the two in one file |
| `units` | `norm1000` for a camera view; `cm` for the farm (pixels scaled by flower size, approximate) |
| `state` | `open`, `bud`, `fruit`, `unknown`; only `open` flowers get a deficit |
| `deficit` | `max(target_visits - visits, 0)`; `target_visits` is a setting, not a biological claim |
| `station` | farm only: station number 1–9 |
| `visits_scope` | what the visit counts cover, e.g. 5 clips of camera view sm3_12 (source mk1.5) or a whole station-day (truth) |
| `clips`, `flowers[].seen_in_clips` | mk1.5 only: clips accumulated into this map, and how many of them detected each flower |

## plan/<field_id>.json (C → D)

```json
{"field_id": "sm3_12_123903", "planner": "ortools", "start": [0, 1000], "budget_s": 60,
 "route": ["flower_4", "flower_1", "flower_6"], "path_length": 820, "deficit_covered": 0.75,
 "robot": {"speed": 50, "dwell_s": 5.2, "time_scale": 4.0},
 "baselines": {"nearest_all": 0.60, "random": 0.41, "follow_bee": 0.22},
 "video": "rollouts/sm3_12_123903_plan.mp4"}
```

`deficit_covered` is the share of total deficit the route clears within `budget_s`, using the
same units for all baselines. `robot.dwell_s` comes from the median bee dwell time in truth.

## eval/report.json (B → D)

```json
{"clips": 46, "median_center_error": 18, "track_coverage": 0.82,
 "visit_recall": 0.75, "visit_precision": 0.80, "notes": "errors worst under occlusion"}
```
