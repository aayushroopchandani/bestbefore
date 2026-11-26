// The tracker (docs/m2c-live-boxes.md, "The frame loop"): each new box
// matched to a shown box of its class by IoU (greedy, highest first, at
// least 0.3); a matched box moves 60% of the way; an unmatched new box
// appears at once; a shown box missed once stays, missed twice goes.
// Written before tracker.ts (Decision 14).
import { describe, expect, it } from "vitest";
import type { Box } from "./tracker";
import { MATCH_IOU, MAX_MISSES, SMOOTH, Tracker, iou, matchBoxes } from "./tracker";

const box = (cls: number, x1: number, y1: number, x2: number, y2: number, conf = 0.5): Box => ({ cls, conf, x1, y1, x2, y2 });
const corners = (b: Box) => [b.x1, b.y1, b.x2, b.y2];

describe("iou", () => {
  it("is 1 for the same box and 0 for boxes apart or only touching", () => {
    expect(iou(box(0, 0, 0, 10, 10), box(0, 0, 0, 10, 10))).toBe(1);
    expect(iou(box(0, 0, 0, 10, 10), box(0, 20, 0, 30, 10))).toBe(0);
    expect(iou(box(0, 0, 0, 10, 10), box(0, 10, 0, 20, 10))).toBe(0);
  });

  it("is the shared area over the area of either", () => {
    expect(iou(box(0, 0, 0, 10, 10), box(0, 5, 0, 15, 10))).toBeCloseTo(50 / 150, 12); // half across
    expect(iou(box(0, 0, 0, 10, 10), box(0, 0, 0, 5, 10))).toBe(0.5); // one inside the other
    expect(iou(box(0, 0, 0, 10, 10), box(0, 5, 5, 15, 15))).toBeCloseTo(25 / 175, 12); // corner to corner
  });

  it("is 0, not NaN, for boxes with no area", () => {
    expect(iou(box(0, 5, 5, 5, 5), box(0, 5, 5, 5, 5))).toBe(0);
  });

  it("doesn't look at the class (matchBoxes does)", () => {
    expect(iou(box(0, 0, 0, 10, 10), box(3, 0, 0, 10, 10))).toBe(1);
  });
});

describe("matchBoxes", () => {
  it("pairs boxes of one class only", () => {
    expect(matchBoxes([box(0, 0, 0, 10, 10)], [box(1, 0, 0, 10, 10)], 0.3)).toEqual([]);
    expect(matchBoxes([box(1, 0, 0, 10, 10)], [box(1, 0, 0, 10, 10)], 0.3)).toEqual([[0, 0]]);
  });

  it("takes the pair of highest IoU first, then the next among boxes still free (greedy, not the best total)", () => {
    const shown = [box(0, 0, 0, 10, 10), box(0, 4, 0, 14, 10)];   // A, B
    const fresh = [box(0, 1, 0, 11, 10), box(0, -2, 0, 8, 10)];   // X: A 0.82, B 0.54; Y: A 0.67
    // Greedy keeps A-X (0.82); Y's only partner A is taken, and B's only partner X too.
    // (A-Y and B-X would add up to more, 0.67 + 0.54: greedy doesn't look for that.)
    expect(matchBoxes(shown, fresh, 0.3)).toEqual([[0, 0]]);
  });

  it("pairs every box when each has its own partner, whatever the order", () => {
    const shown = [box(0, 0, 0, 10, 10), box(0, 100, 0, 110, 10), box(2, 50, 50, 60, 60)];
    const fresh = [box(2, 51, 50, 61, 60), box(0, 101, 0, 111, 10), box(0, 1, 0, 11, 10)];
    expect(matchBoxes(shown, fresh, 0.3)).toEqual([[0, 2], [1, 1], [2, 0]]);
  });

  it("leaves pairs under the minimum IoU apart, and keeps one exactly at it", () => {
    // [0,0,10,10] and [5,0,15,10]: IoU 1/3
    expect(matchBoxes([box(0, 0, 0, 10, 10)], [box(0, 5, 0, 15, 10)], 0.34)).toEqual([]);
    expect(matchBoxes([box(0, 0, 0, 10, 10)], [box(0, 5, 0, 15, 10)], 1 / 3)).toEqual([[0, 0]]);
  });

  it("breaks a tie of IoU by the shown box's order, then the new box's", () => {
    const shown = [box(0, 0, 0, 10, 10), box(0, 0, 0, 10, 10)];
    const fresh = [box(0, 0, 0, 10, 10), box(0, 0, 0, 10, 10)];
    expect(matchBoxes(shown, fresh, 0.3)).toEqual([[0, 0], [1, 1]]);
  });

  it("returns the pairs in the shown boxes' order", () => {
    const shown = [box(0, 0, 0, 10, 10), box(0, 100, 0, 110, 10)];
    const fresh = [box(0, 100, 0, 110, 10), box(0, 2, 0, 12, 10)]; // the second pair has the higher IoU
    expect(matchBoxes(shown, fresh, 0.3)).toEqual([[0, 1], [1, 0]]);
  });
});

describe("Tracker", () => {
  it("uses the plan's numbers: IoU 0.3 to match, 60% of the way, gone at the second miss", () => {
    expect([MATCH_IOU, SMOOTH, MAX_MISSES]).toEqual([0.3, 0.6, 1]);
  });

  it("shows new boxes at once, each with its own id", () => {
    const t = new Tracker();
    const shown = t.update([box(0, 0, 0, 10, 10, 0.9), box(1, 50, 50, 60, 60, 0.4)]);
    expect(shown.map((b) => [b.id, b.cls, b.conf, ...corners(b), b.misses])).toEqual([
      [1, 0, 0.9, 0, 0, 10, 10, 0], [2, 1, 0.4, 50, 50, 60, 60, 0]]);
  });

  it("moves a matched box 60% of the way to the new one, keeping its id, with the new confidence", () => {
    const t = new Tracker();
    t.update([box(0, 0, 0, 10, 10, 0.9)]);
    const [b] = t.update([box(0, 2, 2, 12, 12, 0.7)]); // IoU 64/136, matched
    expect(b.id).toBe(1);
    expect(b.conf).toBe(0.7);
    corners(b).forEach((v, i) => expect(v).toBeCloseTo([1.2, 1.2, 11.2, 11.2][i], 12));
  });

  it("closes in on a box that stays put: what is left of the gap shrinks to 40% each detection", () => {
    const t = new Tracker();
    t.update([box(0, 0, 0, 10, 10)]);
    for (let k = 1; k <= 3; k++) {
      const [b] = t.update([box(0, 3, 0, 13, 10)]);
      expect(b.x1).toBeCloseTo(3 * (1 - 0.4 ** k), 12);
    }
  });

  it("keeps a box missed once where it was, and drops it at the second miss in a row", () => {
    const t = new Tracker();
    t.update([box(0, 0, 0, 10, 10)]);
    const once = t.update([]);
    expect(once.map((b) => [b.id, b.misses, ...corners(b)])).toEqual([[1, 1, 0, 0, 10, 10]]);
    expect(t.update([])).toEqual([]);
  });

  it("forgets a miss once the box is matched again", () => {
    const t = new Tracker();
    t.update([box(0, 0, 0, 10, 10)]);
    t.update([]);
    expect(t.update([box(0, 0, 0, 10, 10)])[0].misses).toBe(0);
    expect(t.update([])).toHaveLength(1); // one miss again: still shown
  });

  it("treats a box of another class in the same place as new, and the old one as missed", () => {
    const t = new Tracker();
    t.update([box(0, 0, 0, 10, 10)]);
    const shown = t.update([box(2, 0, 0, 10, 10)]);
    expect(shown.map((b) => [b.id, b.cls, b.misses])).toEqual([[1, 0, 1], [2, 2, 0]]);
  });

  it("adds a box that moved too far (IoU under 0.3) as new, and misses the old one", () => {
    const t = new Tracker();
    t.update([box(0, 0, 0, 10, 10)]);
    const shown = t.update([box(0, 8, 0, 18, 10)]); // IoU 20/180
    expect(shown.map((b) => [b.id, b.misses, b.x1])).toEqual([[1, 1, 0], [2, 0, 8]]);
  });

  it("is emptied by clear(), which says whether anything was shown", () => {
    const t = new Tracker();
    expect(t.clear()).toBe(false);
    t.update([box(0, 0, 0, 10, 10)]);
    expect(t.clear()).toBe(true);
    expect(t.boxes).toEqual([]);
    expect(t.update([box(0, 0, 0, 10, 10)])[0].id).toBe(2); // ids are never reused
  });
});
