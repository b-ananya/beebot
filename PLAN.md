# BeeBot plan

**Pitch:** Mk1.5 turns ordinary field video into a pollination map, and BeeBot tells a robot
pollinator where to go next.

**Demo (3 min):** pick a clip → bee tracked with boxes and trail, flower boxes → analytics panel
(visits, dwell, heatmap, timeline) → ask "when does the bee land on a flower?" and the video
jumps there → zoom out to the field: pollination map with deficits → the robot's optimal route
beside baselines.

Team: 4 people, 20 hours, Perceptron **Mk1.5 only**, $50 credits each ($200 total).

## Phases and gates

| Phase | Hours | Goal | Gate |
|---|---|---|---|
| 1. Video analysis (**must-have**) | 0–12 | Tracking with boxes and trail, analytics panel, natural-language search | G1 h2 contract frozen · G2 h8 real tracks in the app · G3 h12 Phase 1 demo works end to end |
| 2. Field planner | 12–17 | Pollination map + optimal robot route at patch and field scale | G4 h17 code freeze, numbers locked |
| 3. Freeze and present | 17–20 | Offline demo, backup video, rehearse, submit | G5 h19 submission-ready |

Phase 2 is cut only if Phase 1 isn't solid at G3. Standups at hours 4, 8, 12, 16.

## Workstreams

### A: Perception (Mk1.5)
| Step | Hours | Deliverable | Done when |
|---|---|---|---|
| A1 | 0–2 | Shared Mk1.5 client: caching to `tracks/raw/`, cost logger, $40 cap per key | First call parses; real cost per call recorded in this file |
| A2 | 2–5 | Prompts: one-bee tracking, flower boxes from a reference frame; 720p vs 1080p test | Overlay looks right on 4 of 5 clips |
| A3 | 5–8 | Parser → `tracks/*.json` for 5 clips | `scripts/validate.py` passes; D loads them **(G2)** |
| A4 | 8–11 | Batch run over 48 clips | Unattended; rerun makes zero API calls |
| A5 | 11–14 | Flower state via visual Q&A on crops (open / bud / fruit) | `state` filled in `field/*.json` |
| A6 | 14–17 | v2 tracks from B's failure list | Failures fixed or documented |
| A7 | 17–20 | Freeze cache; `analyze_clip(path)` for D's upload button | Demo runs with network off |

### B: Search + eval + pollination map
| Step | Hours | Deliverable | Done when |
|---|---|---|---|
| B1 | 0–2 | Release shared; team briefed on what truth can and can't claim | Everyone has clips and passes validate |
| B2 | 2–5 | Search index v1: Mk1.5 events and descriptions per clip → `search/index.json` | Query "bee lands on flower" returns timestamps |
| B3 | 5–8 | `eval/run_eval.py`: center error, coverage, visit recall/precision | One command scores first 5 clips |
| B4 | 8–12 | Full eval + failure list to A; search covers all clips | `eval/report.json` v1 **(G3)** |
| B5 | 12–14 | `field/*.json` builder: tracks + events → visits, dwell, deficit per flower | Patch maps for all `sm*` clips, from Mk1.5 and from truth |
| B6 | 14–17 | Field-scale map: 9 stations from `flower_summary.csv` / `flower_visits.csv`, approximate layout | `field/farm_day*.json`; numbers locked **(G4)** |
| B7 | 17–20 | Fact-check every slide number, attribution slide, submission | Matches `report.json` |

### C: Planner + sim (starts on truth data, never waits for A)
| Step | Hours | Deliverable | Done when |
|---|---|---|---|
| C1 | 0–2 | 2D renderer: robot + route drawn over a real video frame → mp4 | Writes mp4 headless |
| C2 | 2–5 | Mock `field.json` from truth; planner v1: greedy + 2-opt with time budget | Route over one patch |
| C3 | 5–8 | OR-Tools orienteering solver; robot params from bee data (median dwell, speed, time scale) | Beats greedy or matches it |
| C4 | 8–12 | Baselines: visit-all nearest-neighbor, random, follow-the-bee; metric = deficit covered per meter | `plan/*.json` for patch clips |
| C5 | 12–14 | Field-scale plan over 9 stations; replay a day hour by hour | Route on `farm_day*.json` |
| C6 | 14–17 | Render polish: bee trails grey, robot route colored, deficit counter | D drops videos into the app |
| C7 | 17–20 | Assumptions paragraph (visit ≠ pollination, layout approximated, no flight physics) | Read by the team |
| C8 | stretch | MuJoCo 3D replay of the planned route (see Stretch goals 5); only if the team agrees to work past hour 17 | 10 `rollouts/<clip>_mujoco.mp4` |

### D: App + story
| Step | Hours | Deliverable | Done when |
|---|---|---|---|
| D1 | 0–2 | Streamlit skeleton on `tracks/mock/`; GitHub Projects board | All four run the app |
| D2 | 2–5 | Panel 1: video with bee box, trail, flower boxes, visit flashes | Plays smoothly on one clip |
| D3 | 5–8 | Panel 2: analytics: visits per flower, dwell, heatmap, timeline | Renders from mock |
| D4 | 8–12 | Search box (index first, live Mk1.5 as a toggle) that seeks the video; real data | **G3** demo |
| D5 | 12–14 | Field view: pollination map, robot route vs baselines; deck v1 | Team commented on deck |
| D6 | 14–17 | Offline mode, "upload your own clip", backup video | App runs with network off |
| D7 | 17–20 | Three timed rehearsals, Q&A sheet, submit | Fits in 2:30 |

## Data (already prepared, see DATA_NOTES.md)

- 48 clips with truth for 46: honeybee clips for tracking quality, farm clips with flower boxes and visits.
- Field planner: 379 flowers at 9 stations over 6 days, per-flower visits by species, 4,786 timestamped visits.
- **Gap:** station positions in the field exist only as the paper's Fig. 2b. We approximate and label it.
- Distance scale: calibrate pixels → cm from flower radius against a cited strawberry-flower size.

## Budget ($50 per account)

Measured in A1 (Sat 2:08 PM, `costs/raika.csv`): one tracking call on a 10 s 720p clip
(`hb1_w00`, prompt v1, no reasoning) used 9,133 input + 265 output tokens = **$0.0018**. The
Models API confirms input $0.15/M, output $1.50/M, context 36,864, max output 8,192. A 48-clip
batch is about $0.09; the worst case stays about $0.035 per full-context call. Reasoning
(`reasoning_effort`) adds output tokens, so it is the one setting that can move the cost.

| Account | Spent on | Estimate |
|---|---|---|
| A | Prompt tuning, batch tracking, v2 rerun | ~$15 |
| B | Search index, events, eval reruns | ~$5 |
| C | Flower boxes and flower state | ~$3 |
| D | Live search, upload demo, rehearsals | ~$5 |

Rules: every response cached; the client refuses calls past $40 spent per key; the demo plays
from cache, live calls are a bonus.

## Tech stack

Python 3.11 · `perceptron` SDK 0.4.0 (Mk1.5) · ffmpeg + OpenCV · numpy / pandas / scipy ·
OR-Tools (routing) · matplotlib + imageio (2D sim) · Streamlit + Plotly · GitHub (repo, Release, Projects)
Stretch only: MuJoCo ≥ 3.2 + MuJoCo Menagerie `bitcraze_crazyflie_2` (3D replay, `requirements-stretch.txt`)

## Stretch goals (after G3, stop at hour 17)

1. Live replanning: a bee visits a flower mid-route, so the robot reroutes.
2. Multiple robots (OR-Tools vehicle routing).
3. Synthetic fields for stress-testing the planner at scale.
4. Battery model sets the route budget.
5. **MuJoCo 3D replay (C8).** Needs the team's OK: it runs past the hour-17 stretch stop.

   | | |
   |---|---|
   | What | A Crazyflie 2 (MuJoCo Menagerie, `bitcraze_crazyflie_2`) replays the planner route in 3D over the clip's first frame; flowers are disks colored by deficit. Kinematic replay only, no flight physics, so the C7 assumption still holds. |
   | Inputs | `field/<clip>.json` and `plan/<clip>.json` (what `sim/render.py` reads) + frame 0 of `clips/<clip>.mp4`. Optional path source: waypoints in `rollouts/<clip>_rl.json` (the MuJoCo hand-off in `sim/README.md`). No new file formats; CONTRACT.md unchanged. |
   | Output | `rollouts/<clip>_mujoco.mp4`, beside `<clip>_plan.mp4` |
   | Code | `sim/mj_render.py`: `python -m sim.mj_render <clip> --menagerie ..\mujoco_menagerie`. Add `mujoco>=3.2` to `requirements-stretch.txt`. Menagerie is cloned outside the repo and never committed. |
   | Mapping | norm1000, origin top-left, y down: `x_w = (x/1000 - 0.5) * W`, `y_w = (0.5 - y/1000) * H`, `W = H * 16/9`. `H ≈ 0.42 m` from the median flower radius in `field/*.json` (29 units; half the larger box side, which here is the vertical extent, so frame-height units like the planner) and a strawberry flower ≈ 2.5 cm across (cite a source before quoting). Speed `robot.speed * H/1000` m/s, matching the leg timing in `sim/render.py`; dwell `robot.dwell_s`. |
   | Watch out | The reference plans `plan/truth/sm3_12_*` have empty routes (every flower already meets the target): skip empty routes, don't crash. The 10 `plan/*.json` routes are non-empty and every route flower exists in its field. |
   | Done when | mp4s render for the 10 `plan/*.json` clips; the robot stops on the same flowers, in the same order, as `<clip>_plan.mp4`; D decides whether the app's "Pollination & robot plan" tab shows it next to the 2D video. |

## Cut list (in order)

1. Stretch goals. 2. Flower state (A5). 3. Field-scale view (keep patch-level plan).
4. Live Mk1.5 search (keep offline index). 5. Upload-your-own-clip.

**Never cut:** tracking with boxes, analytics panel, search, one honest accuracy number, backup video.

## Say on stage

- Truth is the dataset authors' tracker output, not hand labels: "agreement with a reference tracker."
- A visit isn't pollination; target visits per flower is a setting (cite any number used).
- Field layout is approximated; the robot is a planning prototype, not flight physics.
- If the MuJoCo view is shown: "The 3D view is a visual replay of the planned route, not a flight simulation."
