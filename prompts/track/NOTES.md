# Tracking prompt notes (A2)

Scores from `python -m perception.tune` on 5 clips against `truth/` (reference trackers, not hand labels).
`hit` = share of model points within 40 units (~50 px) of the reference centre; `cover` = share of reference
time covered by a hit within 0.3 s; `median` = centre error of matched points, 0-1000 grid. Full numbers in
`tracks/parts/tune/results.json`. Sat Oct 17, 2:30-3:45 PM, $0.16 total.

| config | how | mean hit | mean cover | per-clip hit (hb1_w02 / hb3_w04 / hb6_w02 / sm3_12_123903 / sm2_08_112733) | cost/clip |
|---|---|---|---|---|---|
| v1 | video, box every 0.5 s | 0.21 | 0.12 | 1.00 / 0 / 0 / 0 / 0.07 | $0.002 |
| v2 | video, stricter wording | 0.18 | 0.15 | 0.88 / 0 / 0 / 0 / 0 | $0.002 |
| v2-fine | video, box every 0.25 s | 0.21 | 0.18 | 1.00 / 0 / 0 / 0.07 / 0 | $0.002 |
| v2-think | video, reasoning medium | 0.20 | 0.20 | 0.26 / 0 / 0.30 / 0.37 / 0.06 | $0.003 |
| v3-point | video, points every 0.25 s | 0.30 | 0.26 | 1.00 / 0 / 0 / 0.50 / 0 | $0.002 |
| v3-point-think | video, points + reasoning | 0.09 | 0.10 | 0 / 0 / 0 / 0.47 / 0 | $0.003 |
| f1-2fps | one image call per frame at 2 fps | 0.65 | 0.14 | 1.00 / 0.25 / 1.00 / - / 1.00 (but only 0-5 points per clip) | $0.003 |
| f2-2fps-t2 | per frame, 2x2 tiles | 0.17 | 0.14 | 0.25 / 0.11 / 0.15 / 0 / 0.33 | $0.006 |
| **h1-2fps** | **tiles + motion candidates, each verified on a 192 px crop** | **0.56** | **0.36** | **1.00 / 0.15 / 0.76 / 0.80 / 0.11** | **$0.010** |

## What we learned

1. **Video input cannot see the bee.** Every video prompt (the docs' own tracking recipe) returns a confident,
   well-formed `<track>` that is invented: boxes evenly spaced along a straight line. It only works on `hb1_w02`,
   where the bee sits still near a big flower. A 10 s 720p clip costs ~9k input tokens, so the server samples and
   downscales it until a 25 px bee is a few pixels. Reasoning does not help; it adds output tokens and ghosts.
2. **Image input can see it, but only when zoomed.** On a full 1280x720 frame the model answers `none` on 75-85%
   of frames, yet when it does answer it is within ~8 units of the reference (f1). On 2x2 tiles it answers more
   often but points at dark spots in tiles that have no bee (f2); the tile that does contain the bee is usually right.
3. **Propose, then verify, is what works (h1).** Candidates = tile points + moving blobs from frame differencing
   (free; the camera is fixed). Each candidate is re-asked on a 192 px crop, where the bee is a third of the image.
   Verified points are within 5-6 units on the clean honeybee clips with zero ghosts. Still weak when the bee is
   occluded (`hb3_w04`), and on the wasp clip (`sm2_08_112733`), where the reference itself is sparse (12 of 20 frames).
4. The farm bee (`sm3_12_123903`, 19 units wide) is found precisely (median 13) but only in 5 of 28 frames; it
   mostly sits inside a flower. More candidates (3x3 tiles, 4 fps) are the next lever, at 2-3x the cost.

## Next (A3 / A6)

- Ship h1 at 2 fps for the 5 clips (G2), then the 48-clip batch (~$0.50, ~1 h with 4 workers).
- Try 3x3 tiles and 4 fps on the two weak clips; try the verify prompt with the previous position as a hint.
- Report the honest number: this is agreement with HyDaT/Polytrack, and coverage is the weak metric.
