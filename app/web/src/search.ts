/** Offline match for search/index.json. Same tokens, synonyms, and IDF as search/search.py query(). */

export type Moment = {
  clip_id: string
  t_start: number
  t_end: number
  text: string
  kind?: string
  source?: string
}

const SYNONYMS: Record<string, string[]> = {
  land: ['visit', 'arrive', 'touch', 'alight', 'settle'],
  visit: ['land', 'forage', 'feed'],
  forage: ['visit', 'feed', 'pollinate'],
  feed: ['forage', 'visit'],
  pollinate: ['visit', 'forage'],
  leave: ['takeoff', 'depart', 'exit'],
  takeoff: ['leave', 'depart'],
  depart: ['leave', 'takeoff'],
  fly: ['flight', 'dart'],
  flight: ['fly'],
  fast: ['dart', 'quick'],
  quick: ['fast'],
  enter: ['arrive', 'appear'],
  appear: ['enter', 'arrive'],
  arrive: ['enter', 'land'],
  exit: ['leave'],
  disappear: ['leave', 'exit'],
  bee: ['honeybee', 'insect'],
  honeybee: ['bee'],
  insect: ['bee', 'wasp', 'fly', 'honeybee'],
  pollinator: ['insect', 'bee'],
  wasp: ['vespid'],
  flower: ['bloom', 'blossom'],
  bloom: ['flower'],
  long: ['linger', 'stay'],
  linger: ['long', 'stay'],
  stay: ['linger', 'long'],
  return: ['revisit'],
  revisit: ['return'],
  walk: ['crawl'],
  crawl: ['walk'],
  two: ['multiple', 'several'],
  multiple: ['two', 'several'],
  together: ['multiple', 'same'],
}

const STOP = new Set([
  'the', 'a', 'an', 'on', 'of', 'to', 'in', 'at', 'is', 'does', 'do', 'when', 'where', 'what', 'show', 'me',
  'find', 'moment', 'moments', 'time', 'it', 'its', 'and', 'or', 'with', 'from', 'for', 'any', 'this', 'that',
])

/** Naming one of these answers with no hits. The index never describes them, so a shared word like "appear" must not guess. */
const UNKNOWN_SUBJECTS = new Set([
  'bird', 'spider', 'ant', 'beetle', 'ladybug', 'ladybird', 'dragonfly', 'mosquito',
  'person', 'human', 'hand', 'dog', 'cat', 'frog', 'lizard', 'snail', 'slug', 'mouse', 'robot',
])

const STEM_FIX: Record<string, string> = {
  landing: 'land',
  lands: 'land',
  flie: 'fly',
  flying: 'fly',
  takeoff: 'takeoff',
  taking: 'take',
  visitor: 'visit',
  foraging: 'forage',
  forag: 'forage',
  leav: 'leave',
  arriv: 'arrive',
  lingr: 'linger',
  linger: 'linger',
  pollinat: 'pollinate',
}

const SUFFIXES: Array<[string, string]> = [
  ['ies', 'y'],
  ['ing', ''],
  ['ed', ''],
  ['es', ''],
  ['s', ''],
]

export function stem(word: string): string {
  let w = word.toLowerCase()
  for (const [suf, rep] of SUFFIXES) {
    if (w.endsWith(suf) && w.length - suf.length >= 3) {
      w = w.slice(0, -suf.length) + rep
      break
    }
  }
  return STEM_FIX[w] ?? w
}

export function tokens(text: string): string[] {
  const normalized = text
    .toLowerCase()
    .replaceAll('take off', 'takeoff')
    .replaceAll('takes off', 'takeoff')
    .replaceAll('took off', 'takeoff')
  const words = normalized.match(/[a-z]+/g) ?? []
  return words.filter((word) => !STOP.has(word)).map(stem)
}

/** Python 3 round-half-to-even, used for the duplicate key. */
function roundHalfEven(n: number): number {
  const floor = Math.floor(n)
  const frac = n - floor
  if (frac > 0.5) return floor + 1
  if (frac < 0.5) return floor
  return floor % 2 === 0 ? floor : floor + 1
}

function roundScore(n: number): number {
  return roundHalfEven(n * 1000) / 1000
}

export function query(q: string, index: Moment[], clipId?: string | null, k = 8): Array<Moment & { score: number }> {
  const docs = index.filter((entry) => clipId == null || entry.clip_id === clipId)
  if (docs.length === 0) return []
  const dtoks = docs.map((entry) => new Set(tokens(entry.text)))
  const df = new Map<string, number>()
  for (const ts of dtoks) {
    for (const token of ts) df.set(token, (df.get(token) ?? 0) + 1)
  }
  const n = docs.length
  const idf = (token: string) => Math.log(1 + n / (1 + (df.get(token) ?? 0)))
  const qt = tokens(q)
  if (qt.length === 0 || qt.some((token) => UNKNOWN_SUBJECTS.has(token))) return []

  const scored: Array<{ score: number; dur: number; index: number; event: Moment }> = []
  docs.forEach((event, index) => {
    const ts = dtoks[index]
    let s = 0
    for (const token of qt) {
      if (ts.has(token)) {
        s += idf(token)
        continue
      }
      const matched = (SYNONYMS[token] ?? []).filter((synonym) => ts.has(synonym))
      if (matched.length > 0) s += 0.6 * Math.max(...matched.map(idf))
    }
    if (s > 0) scored.push({ score: s / qt.length, dur: -(event.t_end - event.t_start), index, event })
  })
  scored.sort((a, b) => b.score - a.score || b.dur - a.dur || a.index - b.index)

  const out: Array<Moment & { score: number }> = []
  const seen = new Set<string>()
  for (const row of scored) {
    const key = `${row.event.clip_id}\0${roundHalfEven(row.event.t_start)}\0${row.event.text}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push({ ...row.event, score: roundScore(row.score) })
    if (out.length >= k) break
  }
  return out
}
