// pumpfun-top5-forecast-01 forecaster — trajectory Monte-Carlo (V2).
// Reads /input/task.json + history.ndjson, writes /output/forecast.json.
// Node 22, stdlib only. Fully deterministic: fixed-seed PRNG, no wall clock,
// no network, fixed iteration order. Env overrides (MAKEFUN_INPUT_DIR /
// MAKEFUN_OUTPUT_DIR) per the kit; FF_VARIANT selects the parameter set
// (default "v2"). The shipped default is the chosen variant; see LANE-B report.
//
// Model (per coin i, log10 market cap in SOL x):
//   x_i(h) = x_i0 + drift_i * h + N(0, sd_i(h)) [+ graduation jump if curve]
//   drift_i = shrunk OLS slope of log-cap vs time (per hour), capped, venue field
//             drift as prior; sd_i(h) = min(vol_venue * sqrt(h), SD_CAP).
//   Curve coins graduate w.p. 1-exp(-GRAD_RATE*h) (cap 0.6); on graduation they
//   take a log-cap jump ~ N(JUMP_MEAN, JUMP_SD) (observed curve->pool repricing)
//   and pool volatility thereafter. Non-graduated curve coins are capped at
//   log10(CURVE_CAP_MAX_SOL) — a curve cannot outgrow its bonding curve much.
// Place probabilities and size distributions come from N_SIMS deterministic
// simulations per answer read, averaged over the reads in task.json. A share
// u(H) of each place is left unnamed for newcomers (coins born after the cut,
// unnameable by construction); u grows with mean horizon H. Size distributions
// are the MC size counts blended with a broad lognormal centred on the current
// cap at that place (newcomer sizes are unknowable from the MC universe).
// All named probabilities are clamped to >= P_MIN and per-place sums are kept
// <= 1 - u(H) by construction.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// ---- parameter sets (variants) ----
type Params = {
  volPoolHr: number; volCurveHr: number; sdCap: number;
  driftCapLo: number; driftCapHi: number; slopeShrinkK: number;
  gradRateHr: number; gradCap: number; jumpMean: number; jumpSd: number;
  curveCapMaxSol: number; simsPerRead: number; seed: number;
  unBase: number; unAmp: number; unTauHr: number;
  sizeBlend: number; sizePriorSd0: number; sizePriorSdPerHr: number; sizePriorSdCap: number;
  sizeGrowthPrior: number; sizeGrowthCap: number; sizeGrowthSatHr: number;
  pMin: number; pMax: number; smoothAlpha: number; useSlope: boolean;
};
const VARIANTS: Record<string, Params> = {
  // V2: full trajectory MC (original).
  v2: { volPoolHr: 0.83, volCurveHr: 0.23, sdCap: 1.0, driftCapLo: -0.6, driftCapHi: 1.2,
    slopeShrinkK: 3, gradRateHr: 0.06, gradCap: 0.6, jumpMean: -0.31, jumpSd: 0.55,
    curveCapMaxSol: 500, simsPerRead: 5000, seed: 20261112,
    unBase: 0.30, unAmp: 0.45, unTauHr: 6, sizeBlend: 0.30,
    sizePriorSd0: 0.35, sizePriorSdPerHr: 0.05, sizePriorSdCap: 0.9,
    sizeGrowthPrior: 0.2, sizeGrowthCap: 0.9, sizeGrowthSatHr: 3,
    pMin: 0.002, pMax: 1.0, smoothAlpha: 0.3, useSlope: true },
  // V3: ablation — pure diffusion, no slope.
  v3: { volPoolHr: 0.83, volCurveHr: 0.23, sdCap: 1.0, driftCapLo: 0, driftCapHi: 0,
    slopeShrinkK: 3, gradRateHr: 0.06, gradCap: 0.6, jumpMean: -0.31, jumpSd: 0.55,
    curveCapMaxSol: 500, simsPerRead: 5000, seed: 20261112,
    unBase: 0.30, unAmp: 0.45, unTauHr: 6, sizeBlend: 0.30,
    sizePriorSd0: 0.35, sizePriorSdPerHr: 0.05, sizePriorSdCap: 0.9,
    sizeGrowthPrior: 0.2, sizeGrowthCap: 0.9, sizeGrowthSatHr: 3,
    pMin: 0.002, pMax: 1.0, smoothAlpha: 0.3, useSlope: false },
  // V4: V2 with a larger newcomer reserve and lower vol (more diffuse, more unnamed).
  v4: { volPoolHr: 0.60, volCurveHr: 0.20, sdCap: 0.85, driftCapLo: -0.6, driftCapHi: 1.2,
    slopeShrinkK: 3, gradRateHr: 0.06, gradCap: 0.6, jumpMean: -0.31, jumpSd: 0.55,
    curveCapMaxSol: 500, simsPerRead: 5000, seed: 20261112,
    unBase: 0.42, unAmp: 0.40, unTauHr: 6, sizeBlend: 0.40,
    sizePriorSd0: 0.35, sizePriorSdPerHr: 0.05, sizePriorSdCap: 0.9,
    sizeGrowthPrior: 0.2, sizeGrowthCap: 0.9, sizeGrowthSatHr: 3,
    pMin: 0.002, pMax: 1.0, smoothAlpha: 0.3, useSlope: true },
  // V5: V2 + growth-aware wide size prior, stronger slope shrinkage, per-coin
  // place-prob cap, larger newcomer reserve (3/5 of the 22:00 practice top5
  // were post-cut launches — newcomers dominate at long horizons).
  v5: { volPoolHr: 0.83, volCurveHr: 0.23, sdCap: 1.0, driftCapLo: -0.4, driftCapHi: 0.8,
    slopeShrinkK: 8, gradRateHr: 0.06, gradCap: 0.6, jumpMean: -0.31, jumpSd: 0.55,
    curveCapMaxSol: 500, simsPerRead: 5000, seed: 20261112,
    unBase: 0.45, unAmp: 0.35, unTauHr: 4, sizeBlend: 0.55,
    sizePriorSd0: 0.5, sizePriorSdPerHr: 0.05, sizePriorSdCap: 1.0,
    sizeGrowthPrior: 0.2, sizeGrowthCap: 0.9, sizeGrowthSatHr: 3,
    pMin: 0.002, pMax: 0.22, smoothAlpha: 0.3, useSlope: true },
  // V6: V5 with no slope (diffusion + graduation only).
  v6: { volPoolHr: 0.83, volCurveHr: 0.23, sdCap: 1.0, driftCapLo: 0, driftCapHi: 0,
    slopeShrinkK: 8, gradRateHr: 0.06, gradCap: 0.6, jumpMean: -0.31, jumpSd: 0.55,
    curveCapMaxSol: 500, simsPerRead: 5000, seed: 20261112,
    unBase: 0.45, unAmp: 0.35, unTauHr: 4, sizeBlend: 0.55,
    sizePriorSd0: 0.5, sizePriorSdPerHr: 0.05, sizePriorSdCap: 1.0,
    sizeGrowthPrior: 0.2, sizeGrowthCap: 0.9, sizeGrowthSatHr: 3,
    pMin: 0.002, pMax: 0.22, smoothAlpha: 0.3, useSlope: false },
  // V8: V6 + even larger newcomer reserve.
  v8: { volPoolHr: 0.83, volCurveHr: 0.23, sdCap: 1.0, driftCapLo: 0, driftCapHi: 0,
    slopeShrinkK: 8, gradRateHr: 0.06, gradCap: 0.6, jumpMean: -0.31, jumpSd: 0.55,
    curveCapMaxSol: 500, simsPerRead: 5000, seed: 20261112,
    unBase: 0.55, unAmp: 0.30, unTauHr: 4, sizeBlend: 0.55,
    sizePriorSd0: 0.5, sizePriorSdPerHr: 0.05, sizePriorSdCap: 1.0,
    sizeGrowthPrior: 0.2, sizeGrowthCap: 0.9, sizeGrowthSatHr: 3,
    pMin: 0.002, pMax: 0.22, smoothAlpha: 0.3, useSlope: false },
  // V9: V6 + heavier/wider size prior.
  v9: { volPoolHr: 0.83, volCurveHr: 0.23, sdCap: 1.0, driftCapLo: 0, driftCapHi: 0,
    slopeShrinkK: 8, gradRateHr: 0.06, gradCap: 0.6, jumpMean: -0.31, jumpSd: 0.55,
    curveCapMaxSol: 500, simsPerRead: 5000, seed: 20261112,
    unBase: 0.45, unAmp: 0.35, unTauHr: 4, sizeBlend: 0.70,
    sizePriorSd0: 0.6, sizePriorSdPerHr: 0.05, sizePriorSdCap: 1.0,
    sizeGrowthPrior: 0.2, sizeGrowthCap: 0.9, sizeGrowthSatHr: 3,
    pMin: 0.002, pMax: 0.22, smoothAlpha: 0.3, useSlope: false },
  // V10: V6 + tighter per-coin place cap.
  v10: { volPoolHr: 0.83, volCurveHr: 0.23, sdCap: 1.0, driftCapLo: 0, driftCapHi: 0,
    slopeShrinkK: 8, gradRateHr: 0.06, gradCap: 0.6, jumpMean: -0.31, jumpSd: 0.55,
    curveCapMaxSol: 500, simsPerRead: 5000, seed: 20261112,
    unBase: 0.45, unAmp: 0.35, unTauHr: 4, sizeBlend: 0.55,
    sizePriorSd0: 0.5, sizePriorSdPerHr: 0.05, sizePriorSdCap: 1.0,
    sizeGrowthPrior: 0.2, sizeGrowthCap: 0.9, sizeGrowthSatHr: 3,
    pMin: 0.002, pMax: 0.15, smoothAlpha: 0.3, useSlope: false },
  // V7: V5 with the smaller newcomer reserve (sensitivity check).
  v7: { volPoolHr: 0.83, volCurveHr: 0.23, sdCap: 1.0, driftCapLo: -0.4, driftCapHi: 0.8,
    slopeShrinkK: 8, gradRateHr: 0.06, gradCap: 0.6, jumpMean: -0.31, jumpSd: 0.55,
    curveCapMaxSol: 500, simsPerRead: 5000, seed: 20261112,
    unBase: 0.30, unAmp: 0.45, unTauHr: 6, sizeBlend: 0.55,
    sizePriorSd0: 0.5, sizePriorSdPerHr: 0.05, sizePriorSdCap: 1.0,
    sizeGrowthPrior: 0.2, sizeGrowthCap: 0.9, sizeGrowthSatHr: 3,
    pMin: 0.002, pMax: 0.22, smoothAlpha: 0.3, useSlope: true },
};
const P = VARIANTS[process.env.FF_VARIANT ?? 'v6'] ?? VARIANTS.v6!;

// ---- deterministic PRNG (mulberry32) + Box-Muller ----
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
class Gauss {
  private spare: number | null = null;
  private rnd: () => number;
  constructor(rnd: () => number) { this.rnd = rnd; }
  next(): number {
    if (this.spare !== null) { const v = this.spare; this.spare = null; return v; }
    let u = 0, v = 0, s = 0;
    do { u = this.rnd() * 2 - 1; v = this.rnd() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const m = Math.sqrt(-2 * Math.log(s) / s);
    this.spare = v * m;
    return u * m;
  }
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

// erfc (Numerical Recipes), for Normal CDF bins.
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const normCdfAbove = (x: number, mean: number, sd: number): number =>
  x === -Infinity ? 1 : x === Infinity ? 0 : 0.5 * erfc((x - mean) / (sd * Math.SQRT2));

// ---- input ----
interface Task { places: number; output: string; size: { edges: number[] }; reads: string[]; history_until: string; t0: string; }
interface Row { rank: number; mint: string; market_cap_lamports: string | number; venue: string; created_at: string; complete: boolean; mayhem: boolean; }
interface Mark { at: string; ok: boolean; rows: Row[] | null; }

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8').split('\n').filter((l) => l !== '');
const marks = lines.map((l) => JSON.parse(l) as Mark).filter((m) => m.ok === true && Array.isArray(m.rows));
if (marks.length === 0) throw new Error('the history holds no ok mark');
const lastMark = marks[marks.length - 1]!;
const tLast = Date.parse(lastMark.at);
const t0 = Date.parse(task.t0);
const edges = task.size.edges;
const PLACES = task.places;
const NBINS = edges.length + 1; // 82
const sizeOf = (lamports: string | number): number => Math.log10(Number(BigInt(lamports))) - 9;
const binOf = (x: number): number => {
  if (x < edges[0]!) return 0;
  if (x >= edges[edges.length - 1]!) return NBINS - 1;
  for (let i = 0; i < edges.length - 1; i++) if (x < edges[i + 1]!) return i + 1;
  return NBINS - 1;
};

// ---- per-coin series ----
interface Coin {
  mint: string; x0: number; venue: string; drift: number; volHr: number;
  lastRank: number; nObs: number; isCurve: boolean;
}
const series = new Map<string, Array<{ tHr: number; x: number; venue: string }>>();
const createdAt = new Map<string, number>();
for (const m of marks) {
  const tHr = (Date.parse(m.at) - t0) / 3600000;
  for (const r of m.rows!) {
    const arr = series.get(r.mint) ?? [];
    arr.push({ tHr, x: sizeOf(r.market_cap_lamports), venue: r.venue });
    series.set(r.mint, arr);
    if (!createdAt.has(r.mint)) createdAt.set(r.mint, Date.parse(r.created_at));
  }
}
// Field drift per venue (log10/hr), from dev 10-min mean deltas x6: pool -0.07, curve -0.08.
const FIELD_DRIFT: Record<string, number> = { pool: -0.07, curve: -0.08, migrating: -0.08 };
const lastChart = [...lastMark.rows!].sort((a, b) => a.rank - b.rank);
const lastRankOf = new Map<string, number>(lastChart.map((r) => [r.mint, r.rank]));

const coins: Coin[] = [];
for (const [mint, obs] of series) {
  const last = obs[obs.length - 1]!;
  const venue = last.venue;
  const isCurve = venue === 'curve';
  // OLS slope of x vs tHr (per hour), shrunk toward the venue field drift.
  let slope = FIELD_DRIFT[venue] ?? -0.08;
  if (P.useSlope && obs.length >= 2) {
    const n = obs.length;
    const mt = obs.reduce((s, o) => s + o.tHr, 0) / n;
    const mx = obs.reduce((s, o) => s + o.x, 0) / n;
    let num = 0, den = 0;
    for (const o of obs) { num += (o.tHr - mt) * (o.x - mx); den += (o.tHr - mt) * (o.tHr - mt); }
    if (den > 0) {
      const raw = num / den;
      const w = n / (n + P.slopeShrinkK);
      slope = w * raw + (1 - w) * (FIELD_DRIFT[venue] ?? -0.08);
    }
  }
  slope = clamp(slope, P.driftCapLo, P.driftCapHi);
  coins.push({
    mint, x0: last.x, venue, drift: slope,
    volHr: isCurve ? P.volCurveHr : P.volPoolHr,
    lastRank: lastRankOf.get(mint) ?? 9999, nObs: obs.length, isCurve,
  });
}

// ---- horizons ----
const reads = task.reads.map((r) => Date.parse(r)).filter((t) => Number.isFinite(t));
const horizons = reads.length > 0 ? reads.map((t) => Math.max(0, (t - tLast) / 3600000)) : [1];
const H = horizons.reduce((s, h) => s + h, 0) / horizons.length;
// Newcomer (unnamed) share per place, growing with mean horizon.
const uShare = clamp(P.unBase + P.unAmp * (1 - Math.exp(-H / P.unTauHr)), 0, 0.9);
const namedShare = 1 - uShare;

// ---- candidate set: top by expected log-cap at mean horizon ----
// mu_i(H) = x0 + drift*H (+ graduation expectation for curve coins).
const curveCapLog = Math.log10(P.curveCapMaxSol);
function muAt(c: Coin, h: number): number {
  let mu = c.x0 + c.drift * h;
  if (c.isCurve) {
    const pg = Math.min(P.gradCap, 1 - Math.exp(-P.gradRateHr * h));
    // Expected value mixing graduated (jump) vs not (capped at curve max).
    const gradMu = c.x0 + P.jumpMean + c.drift * h;
    const stayMu = Math.min(mu, curveCapLog);
    mu = pg * gradMu + (1 - pg) * stayMu;
  }
  return mu;
}
const ranked = [...coins].sort((a, b) => muAt(b, H) - muAt(a, H) || a.mint.localeCompare(b.mint));
const CAND = Math.min(120, ranked.length);
const cand = ranked.slice(0, CAND);
// Current place caps (for size prior + top5 fallback).
const curPlaceCap: number[] = [];
for (let k = 0; k < PLACES; k++) curPlaceCap.push(lastChart[k] ? sizeOf(lastChart[k]!.market_cap_lamports) : 3.0);

// ---- Monte Carlo over reads ----
const rnd = mulberry32(P.seed);
const g = new Gauss(rnd);
const placeCounts: number[][] = Array.from({ length: PLACES }, () => new Array<number>(CAND).fill(0));
const sizeCounts: number[][] = Array.from({ length: PLACES }, () => new Array<number>(NBINS).fill(0));
const sims = P.simsPerRead * horizons.length;
const xs = new Array<number>(CAND);
const idx = Array.from({ length: CAND }, (_, i) => i);
for (const h of horizons) {
  const sdScale = Math.sqrt(Math.max(h, 0));
  for (let s = 0; s < P.simsPerRead; s++) {
    for (let i = 0; i < CAND; i++) {
      const c = cand[i]!;
      const sd = Math.min(c.volHr * sdScale, P.sdCap);
      let x = c.x0 + c.drift * h + g.next() * sd;
      if (c.isCurve) {
        const pg = Math.min(P.gradCap, 1 - Math.exp(-P.gradRateHr * h));
        if (rnd() < pg) {
          x = c.x0 + P.jumpMean + c.drift * h + g.next() * Math.hypot(sd, P.jumpSd);
        } else {
          x = Math.min(x, curveCapLog + 0.05 * g.next());
        }
      }
      xs[i] = clamp(x, 1.0, 5.2);
    }
    // Top PLACES by simulated cap (ties -> lower candidate index, deterministic).
    idx.sort((a, b) => xs[b]! - xs[a]! || a - b);
    for (let k = 0; k < PLACES; k++) {
      const ci = idx[k]!;
      placeCounts[k]![ci]! += 1;
      sizeCounts[k]![binOf(xs[ci]!)]! += 1;
    }
  }
}

// ---- probabilities ----
interface ProbRow { mint: string; place: number; p: number; }
const probabilities: ProbRow[] = [];
const placeProb: number[][] = [];
for (let k = 0; k < PLACES; k++) {
  const raw: number[] = [];
  let tot = 0;
  for (let i = 0; i < CAND; i++) {
    const p = (placeCounts[k]![i]! + P.smoothAlpha) / (sims + P.smoothAlpha * CAND);
    raw.push(p); tot += p;
  }
  // Keep the strongest named coins (cap 100 rows/place), renormalise to namedShare.
  const order = Array.from({ length: CAND }, (_, i) => i).sort((a, b) => raw[b]! - raw[a]! || a - b);
  const keep = order.slice(0, 100);
  let keepTot = 0;
  for (const i of keep) keepTot += raw[i]!;
  const row: number[] = new Array<number>(CAND).fill(0);
  for (const i of keep) row[i] = (raw[i]! / keepTot) * namedShare;
  // Clamp to >= P_MIN and <= P_MAX, then rescale to namedShare.
  let scaled = row.map((p) => (p > 0 ? clamp(Math.max(p, P.pMin), 0, P.pMax) : 0));
  let ssum = scaled.reduce((s, p) => s + p, 0);
  if (ssum > namedShare) {
    const f = namedShare / ssum;
    scaled = scaled.map((p) => p * f);
  }
  // Round to 6 dp (floor, like the starter, to never exceed 1 in sum).
  const rounded = scaled.map((p) => (p > 0 ? Math.floor(p * 1e6) / 1e6 : 0));
  const rsum = rounded.reduce((s, p) => s + p, 0);
  placeProb.push(rounded);
  for (let i = 0; i < CAND; i++) {
    if (rounded[i]! > 0) probabilities.push({ mint: cand[i]!.mint, place: k + 1, p: rounded[i]! });
  }
  void rsum;
}

// ---- top5: by expected place score ----
const scoreOf = cand.map((_, i) => {
  let s = 0;
  for (let k = 0; k < PLACES; k++) s += placeProb[k]![i]! * (PLACES + 1 - k);
  return s;
});
const topOrder = Array.from({ length: CAND }, (_, i) => i)
  .sort((a, b) => scoreOf[b]! - scoreOf[a]! || muAt(cand[b]!, H) - muAt(cand[a]!, H) || a - b);
let top5 = topOrder.slice(0, PLACES).map((i) => cand[i]!.mint);
// Fallback: if MC produced nothing sensible, use the last chart's top 5.
if (top5.length < PLACES) top5 = lastChart.slice(0, PLACES).map((r) => r.mint);

// ---- sizes: MC counts blended with a broad, growth-aware lognormal prior ----
// Per-place rank-cap growth: OLS slope of the log-cap AT RANK k across marks
// (the cap profile itself inflates as bigger graduates arrive), shrunk toward
// a +0.2 log10/hr prior, capped; growth saturates after sizeGrowthSatHr.
function rankCapSlope(k: number): number {
  const pts: Array<{ t: number; x: number }> = [];
  for (const m of marks) {
    const rows = [...(m.rows ?? [])].sort((a, b) => a.rank - b.rank);
    if (rows[k]) pts.push({ t: (Date.parse(m.at) - t0) / 3600000, x: sizeOf(rows[k]!.market_cap_lamports) });
  }
  if (pts.length < 2) return P.sizeGrowthPrior;
  const n = pts.length;
  const mt = pts.reduce((s, p) => s + p.t, 0) / n;
  const mx = pts.reduce((s, p) => s + p.x, 0) / n;
  let num = 0, den = 0;
  for (const p of pts) { num += (p.t - mt) * (p.x - mx); den += (p.t - mt) * (p.t - mt); }
  if (den <= 0) return P.sizeGrowthPrior;
  const w = n / (n + 4);
  return clamp(w * (num / den) + (1 - w) * P.sizeGrowthPrior, -0.2, P.sizeGrowthCap);
}
interface SizeRow { place: number; bin: number; p: number; }
const sizes: SizeRow[] = [];
for (let k = 0; k < PLACES; k++) {
  const sdPrior = Math.min(P.sizePriorSd0 + P.sizePriorSdPerHr * H, P.sizePriorSdCap);
  const meanPrior = curPlaceCap[k]! + rankCapSlope(k) * Math.min(H, P.sizeGrowthSatHr);
  const dist = new Array<number>(NBINS).fill(0);
  for (let b = 0; b < NBINS; b++) {
    const lo = b === 0 ? -Infinity : edges[b - 1]!;
    const hi = b === NBINS - 1 ? Infinity : edges[b]!;
    const mc = (sizeCounts[k]![b]! + 0.5) / (sims + 0.5 * NBINS);
    const pr = normCdfAbove(lo, meanPrior, sdPrior) - normCdfAbove(hi, meanPrior, sdPrior);
    dist[b] = (1 - P.sizeBlend) * mc + P.sizeBlend * pr;
  }
  const tot = dist.reduce((s, p) => s + p, 0);
  for (let b = 0; b < NBINS; b++) {
    const p = Math.floor((dist[b]! / tot) * 1e6) / 1e6;
    if (p > 0) sizes.push({ place: k + 1, bin: b, p });
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);
