# Flower boxes (A2, piece p2): work log

Owner: Falisha. Code: `perception/flowers.py`. Tests: `tests/test_flowers.py`. Prompts: `prompts/flowers/v*.txt`.

## Status (Sat Oct 17, ~4:30 PM PDT)

| Step | State |
|---|---|
| 1. `reference_frame()` middle frame -> `tracks/raw/frames/<clip>_ref.png` | done; 10 `sm*` frames extracted |
| 2. `detect_flowers()` via `PerceptionClient("p2")`, `annotation_format="box"` | done; run on all 10 `sm*` clips |
| 3. `parse_flowers()` drop invalid, dedupe IoU>0.7, sort; `assign_view_ids()` stable ids per camera view | done, tested |
| 4. `tracks/parts/<clip>.flowers.json` + `--merge` | done; merge waits for P1 track parts on `sm*` |
| 5. `--eval` (precision, recall, mean IoU vs truth) + `--overlay` + FakeClient tests | done; results below |

Next: push flower files (Falisha) -> Raika's hybrid tracking parts for `sm*` + events -> on a branch with both,
`python -m perception.merge --all --interpolate 1.0`. Then `hb*` clips (all clips go in the demo) with a Scaevola prompt.

## Team decisions (Sat ~4 PM)

- Demo uses all clips, so `hb*` clips need flower boxes too (separate prompt; Scaevola is blue/purple, no truth).
- Flower ids must match across clips of the same camera view (option A: our own ids, stable per view; truth ids
  are only mapped in eval).
- Box only open flowers, i.e. ones that can be pollinated (v1 already skips buds and fruit).
- Raika: P1 tracks need `perception.track --mode hybrid --version f3 --tiles 2 --fps 2` (on her branch, ~$0.01/clip);
  default video mode invents straight-line tracks. Farm bee coverage is thin (5/28 frames on `sm3_12_123903`). She
  offered to derive visit events from bee-in-flower-box geometry; that needs our stable ids first. Docstring fixed.

## Results, prompt v1, 720p (`python -m perception.flowers --sm --eval --overlay`)

10 calls, $0.0042 total (~1,030 input + ~240 output tokens, ~$0.0004 per clip). 9-16 flowers per clip.

| Camera | Recall (IoU>=0.3) | Precision | Mean IoU | Truth centers with a Mk1.5 center within 40 units |
|---|---|---|---|---|
| sm2 (5 clips, 6 truth each) | 0.40 | 0.26 | 0.49 | 25/30 |
| sm3 (5 clips, 7 truth each) | 0.71 | 0.39 | 0.46 | 32/35 |
| all | 0.57 | 0.33 | 0.47 | 57/65 (88%) |

Read from the overlays (`eval/checks/sm*_flowers.png`): nearly every Mk1.5 box is on a real open flower. Low
precision is mostly real flowers missing from truth; a few boxes land on yellow fruit centers. Low `sm2` recall is
truth offset: `sm2` truth boxes sit ~20 px below/right of the flowers, so IoU falls under 0.3 though the flower is
found. Center distance is the fairer number to quote. 1080p not needed for detection.

## Compatibility with P1 (Raika's tracking)

- Uses `perception.client` for every call; cache in `tracks/raw/p2/`, cost in `costs/falisha.csv`.
- Reuses `track.manifest`, `track.clip_path`, `track.PARTS_DIR`; no P1 file is changed.
- Part shape is what `merge.py` reads (`{"clip_id", "flowers": [{"flower_id", "box"}]}`) plus run metadata like the
  track part. A test merges a P1 track part + this flowers part + an events part and passes `validate.py`.
- `merge.py` needs a track part, so `sm*` clips merge only after Raika's hybrid tracking run (see above).

## Decisions

- Stable ids: clips sharing `view_id` (clip id minus the last `_part`, e.g. `sm3_12`) are one fixed camera: frames
  across clips are shifted < 0.1 px and differ by ~2/255 gray (vs 61 between cameras). Boxes are matched one-to-one
  per clip at IoU >= 0.4 (same flower across clips: IoU >= 0.7 in ~90% of cases; distinct flowers: <= 0.09), ids
  numbered by each flower's mean box. Relabeling runs on every write and is idempotent. `sm2_08`: 10 flowers
  (9 seen in all 5 clips); `sm3_12`: 17 flowers (9 in all 5).
- One middle frame per clip, not the video: flowers are static and an image call is far cheaper.
- Ref PNGs are committed (`.gitignore` now ignores `tracks/raw/frames/*` except `*_ref.png`): the cache key hashes
  the PNG bytes, so a teammate re-extracting with another OpenCV build could miss the cache and pay again.
- Bad boxes are dropped, never fixed (same rule as `track.parse_track`).
- Eval matches one-to-one by IoU >= 0.3, looser than 0.5 because truth flower boxes are approximate.
- Prompt v1 size hint: truth flower widths at 720p are 33-49 px (`sm2`) and 41-53 px (`sm3`).

## Data observations

- Truth flowers are only the 6-7 flowers per view that the dataset annotated; other open flowers in the frame have
  no truth box, and at least one truth box (`sm2_08_112658` flower_0) sits on leaves. Expect low precision against
  truth; recall is the meaningful number. Say "agreement with dataset flower annotations".
- The 5 `sm2` clips share one camera view, as do the 5 `sm3` clips: effectively 2 distinct scenes. Flowers
  do not move within or across a view's clips (checked by frame differencing).
- Many yellow-green post-bloom centers (fruit) look like flowers; v1 tells the model to skip them.

## Open issues

- 1080p test (plan risk): not done; only needed if 720p recall is poor.
