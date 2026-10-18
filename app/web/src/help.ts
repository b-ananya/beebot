/** Plain-language explanations shown on hover across the panels. One place, so wording stays consistent and the
 *  definitions match the code (analytics.ts mirrors analytics/metrics.py; field.tsx mirrors sim/planner.py). */

export const HELP = {
  // ---------------------------------------------------------------- analytics: visits
  visits:
    'A visit = an insect stays inside one flower’s zone (its box, grown 25% on each side) for at least 0.3 s. ' +
    'If zones overlap, the insect counts toward the nearest flower only.',
  flowersVisited: 'How many different flowers got at least one visit in this clip.',
  flowersNeverVisited:
    'Flowers Mk1.5 detected in the frame that got no visit in this clip: candidates for a robot to cover. ' +
    '(Spots where an insect merely stopped are not counted.)',
  visitsPerMin: 'Visits divided by the clip length in minutes. Clips are short, so treat this as rough.',
  revisitRate: 'Share of visits that go back to a flower the same insect already visited in this clip.',

  // ---------------------------------------------------------------- analytics: robot parameters
  robotParams:
    'Numbers a robot pollinator or a simulator can copy from the insects: how long to stay on a flower, how long ' +
    'to travel between flowers, how fast to fly.',
  medianDwell: 'Typical (median) time an insect stays on a flower per visit. A robot would hold about this long.',
  medianTransit: 'Typical time from leaving one flower to landing on the next.',
  flightSpeed:
    'Typical speed while the insect is clearly flying (faster than 40 units/s), converted to cm/s with the ' +
    'approximate scale below.',
  approachTime: 'How long the insect hovers near a flower (within about 1.5 flower-widths) before landing on it.',
  nearestFlowerRate:
    'How often the next flower is simply the closest one. High = the insect forages greedily; a robot could ' +
    'use the same simple rule.',
  medianTurn:
    'Typical change of direction between 0.2 s steps of the path. Higher = a wigglier, more searching flight.',
  movingShare: 'Share of the time the insect is moving fast (flying) rather than sitting or crawling.',
  trackedShare: 'Share of the clip during which at least one insect is being tracked.',
  interpolatedShare:
    'Share of track points filled in between real detections (for short gaps). Lower = more directly observed.',
  scale:
    'How many map units make 1 cm, used for cm/s. Honeybee clips: measured from the folding ruler in the shot. ' +
    'Farm clips: assumes a strawberry flower is about 2.5 cm across. Approximate either way.',

  // ---------------------------------------------------------------- analytics: charts
  dwellChart: 'Total time insects spent on each flower in this clip (all visits added up).',
  speedChart:
    'How often the insects move at each speed. A tall bar near 0 = lots of sitting or crawling; the long tail to ' +
    'the right = fast flights between flowers. 1 unit = 1/1000 of the frame width.',
  heatmap:
    'Where the insects spent their time. The frame is split into a 40×40 grid; each cell is colored by how many ' +
    'tracked positions fell in it (all insects, directly observed points only). Light = a little time, dark blue = ' +
    'a lot. Yellow boxes are flowers. The picture is the video frame at the current time.',
  transitions:
    'Every move from one flower to the next: how long it took, how far it was (map units) and the speed.',
  visitTable: 'Every visit, in order. Click a row to jump the video to that moment.',

  // ---------------------------------------------------------------- forecast
  insect: 'Which tracked insect to forecast.',
  forecastFrom:
    'The moment to forecast from. The model only sees the path up to this time; the grey dotted line shows what ' +
    'actually happened next, so you can compare. Moving the slider also moves the video.',
  horizon:
    'How far ahead the simulated futures run (5–30 s). It changes “Leaves within”, the expected visits and the most ' +
    'likely sequence. The 2-second path forecast on the picture does not change.',
  dwellRemaining:
    'How much longer the insect is likely to stay on its current flower, given how long it has already been there.',
  anotherVisit: 'Probability that the insect visits another flower at all, rather than leaving the patch.',
  leavesWithin: 'Share of the simulated futures in which the insect leaves the patch within the horizon.',
  likelySequence: 'The single most common order of flower visits across the simulated futures.',

  // ---------------------------------------------------------------- field
  fieldTarget:
    'How many insect visits each open flower should get. A flower’s “deficit” is this target minus the visits ' +
    'counted so far. It is a setting you choose, not a biological fact.',
  fieldBudget: 'How many seconds the robot has. The route ends when the time runs out.',
  fieldSpeed:
    'Robot speed in map units per second (1000 units = the frame height, about 42 cm), so 100 ≈ 4 cm/s. Each stop ' +
    'also costs the typical insect dwell time.',
  flowersMapped:
    'Every flower this camera has seen across its clips so far (the same flower keeps the same id in every clip).',
  neverVisitedField: 'Mapped flowers with no visit counted yet.',
  stillNeeded: 'Visits still needed to bring every open flower up to the target (sum of all deficits).',
  shareCleared: 'Share of the needed visits that the planned route delivers within the budget.',
  visitsScope:
    'Which footage the visit counts come from. Each new clip of the same camera adds its visits, so the map fills ' +
    'in over the morning.',
  ourRoute:
    'Our planner: picks flowers to clear the most deficit within the budget, then shortens the path (greedy + 2-opt).',
  nearest: 'Baseline: always fly to the closest flower that still needs visits.',
  random: 'Baseline: visit flowers in a random order (average of 20 tries).',
  followBee: 'Baseline: copy the order in which the insect visited flowers in this clip.',

  // ---------------------------------------------------------------- search
  liveSearch:
    'Off (default): instant, free search over moments already found in the tracks (visits, landings, take-offs, ' +
    'fast flights, insects arriving or leaving). On: press Enter to send your question and this clip to Mk1.5, which ' +
    'looks for the moment itself. Live costs an API call (cached afterwards), takes a few seconds, and is ' +
    'experimental: Mk1.5 is unreliable on whole-video questions at this resolution.',

  // ---------------------------------------------------------------- dataset
  coverage: 'Share of the reference tracker’s points that we matched within 50 units at the same time.',
  medianError: 'Typical distance between our point and the reference point, in map units (0–1000 across the frame on each axis).',
  falseAlarms:
    'Our points with no reference point within 50 units. Overstated: the reference follows one bee per video, and ' +
    'a visual check found most of these are real bees (see the image below).',
  visitRecall: 'Share of the reference visits we also found (farm clips only).',
  visitPrecision: 'Share of our visits that match a reference visit (farm clips only).',
} as const

export const IMAGE_NOTES: Record<string, { title: string; body: string }> = {
  'disputed_points_v0.7.jpg': {
    title: 'Are our “false alarms” really wrong?',
    body:
      '30 random moments where we tracked an insect but the reference tracker had nothing within 50 units. The red ' +
      'circle is our point. In 20 the circle is clearly on a bee, in 5 probably, and 5 are not insects (ruler marks, ' +
      'leaves). The reference follows only one bee per video, so most of these are other real bees.',
  },
  'flowers_sm2_08_boxes.jpg': {
    title: 'Camera 2: every flower box Mk1.5 drew',
    body:
      'One crop per distinct flower box (magenta) across this camera’s 5 clips; “(5/5)” = found in 5 of 5 clips. ' +
      'All 9 are open strawberry flowers, including small and partly hidden ones.',
  },
  'flowers_sm3_12_boxes.jpg': {
    title: 'Camera 3: every flower box Mk1.5 drew',
    body:
      '13 distinct boxes; 11 are open flowers. Two are mistakes, each made in only 1 of 5 clips: #10 is a green fruit ' +
      'and #13 is a pale leaf at the frame edge.',
  },
  'flowers_unboxed_candidates.jpg': {
    title: 'White things Mk1.5 did NOT box',
    body:
      'Everything white we found that has no box, zoomed in. A and E are real small open flowers (genuine misses). ' +
      'B and D are wilted flowers, C is a closed bud, and F and G are fallen petals, so leaving those unboxed is right.',
  },
}
