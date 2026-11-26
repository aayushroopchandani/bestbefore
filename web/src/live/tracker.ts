/**
 * The tracker (docs/m2c-live-boxes.md, "The frame loop"): keeps the boxes
 * shown from flickering between detections, and hides nothing.
 *
 * Each new box is matched to a shown box of its class by IoU (greedy,
 * highest first, at least 0.3). A matched box moves 60% of the way to the
 * new one, which smooths jitter; an unmatched new box appears at once; a
 * shown box missed once stays, missed twice goes. Boxes are in frame pixels.
 *
 * No React and no browser.
 */

export const MATCH_IOU = 0.3;
export const SMOOTH = 0.6; // how far a matched box moves toward the new one
export const MAX_MISSES = 1; // a box missed more times than this in a row goes

export interface Box {
  cls: number;
  conf: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Track extends Box {
  id: number;
  misses: number; // detections in a row that didn't match it
}

/** Intersection over union of two boxes (0 for boxes with no area). */
export function iou(a: Box, b: Box): number {
  const w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Pairs shown boxes with new ones of the same class: every pair at
 * `minIou` or more, highest IoU first (ties by the shown box's order, then
 * the new box's), each box used once. Returns [shown index, new index]
 * pairs in the shown boxes' order.
 */
export function matchBoxes(shown: readonly Box[], fresh: readonly Box[], minIou: number): [number, number][] {
  const pairs: { i: number; j: number; iou: number }[] = [];
  shown.forEach((s, i) =>
    fresh.forEach((f, j) => {
      if (s.cls !== f.cls) return;
      const v = iou(s, f);
      if (v >= minIou) pairs.push({ i, j, iou: v });
    }),
  );
  pairs.sort((p, q) => q.iou - p.iou || p.i - q.i || p.j - q.j);
  const usedShown = new Set<number>();
  const usedFresh = new Set<number>();
  const out: [number, number][] = [];
  for (const { i, j } of pairs) {
    if (usedShown.has(i) || usedFresh.has(j)) continue;
    usedShown.add(i);
    usedFresh.add(j);
    out.push([i, j]);
  }
  return out.sort((p, q) => p[0] - q[0]);
}

export class Tracker {
  private shown: Track[] = [];
  private nextId = 1;

  get boxes(): readonly Track[] {
    return this.shown;
  }

  /** A detection's boxes: matched ones smoothed, new ones added, missed ones kept once. */
  update(fresh: readonly Box[]): readonly Track[] {
    const matchOf = new Map(matchBoxes(this.shown, fresh, MATCH_IOU));
    const matched = new Set(matchOf.values());
    const kept: Track[] = [];
    this.shown.forEach((s, i) => {
      const j = matchOf.get(i);
      if (j !== undefined) {
        const f = fresh[j];
        const toward = (a: number, b: number) => a + SMOOTH * (b - a);
        kept.push({ id: s.id, cls: s.cls, conf: f.conf, misses: 0,
          x1: toward(s.x1, f.x1), y1: toward(s.y1, f.y1), x2: toward(s.x2, f.x2), y2: toward(s.y2, f.y2) });
      } else if (s.misses < MAX_MISSES) {
        kept.push({ ...s, misses: s.misses + 1 });
      }
    });
    fresh.forEach((f, j) => {
      if (!matched.has(j)) kept.push({ id: this.nextId++, cls: f.cls, conf: f.conf, x1: f.x1, y1: f.y1, x2: f.x2, y2: f.y2, misses: 0 });
    });
    this.shown = kept;
    return kept;
  }

  /** Empties it; says whether anything was shown. */
  clear(): boolean {
    const had = this.shown.length > 0;
    this.shown = [];
    return had;
  }
}
