import { describe, expect, it } from "vitest";
import type { BenchRow, CopySummary } from "./gateBench";
import { COPIES, benchPlan, slotAt, summariseBench } from "./gateBench";

describe("benchPlan", () => {
  it("runs every copy for its seconds, round after round, back to back", () => {
    const plan = benchPlan(["cpu-medium", "gpu-medium"], 2, 3);
    expect(plan.map((s) => [s.copy, s.round, s.from, s.to])).toEqual([
      ["cpu-medium", 1, 0, 3000], ["gpu-medium", 1, 3000, 6000],
      ["cpu-medium", 2, 6000, 9000], ["gpu-medium", 2, 9000, 12000],
    ]);
  });

  it("defaults to the four copies, two rounds of 3 s: 24 s", () => {
    const plan = benchPlan();
    expect(COPIES).toEqual(["cpu-medium", "cpu-low", "gpu-medium", "bitmap-medium"]);
    expect(plan).toHaveLength(8);
    expect(plan.at(-1)!.to).toBe(24000);
  });
});

describe("slotAt", () => {
  const plan = benchPlan(["cpu-medium", "gpu-medium"], 1, 3);
  it("finds the slot a moment falls in: from inclusive, to exclusive", () => {
    expect(slotAt(plan, 0)!.copy).toBe("cpu-medium");
    expect(slotAt(plan, 2999.9)!.copy).toBe("cpu-medium");
    expect(slotAt(plan, 3000)!.copy).toBe("gpu-medium");
  });
  it("is null before the start and from the end on", () => {
    expect(slotAt(plan, -1)).toBeNull();
    expect(slotAt(plan, 6000)).toBeNull();
  });
});

describe("summariseBench", () => {
  const plan = benchPlan(["cpu-medium", "gpu-medium"], 1, 2); // 2 s each
  const row = (copy: BenchRow["copy"], block: number, wall: number, value: number | null): BenchRow =>
    ({ t: 0, copy, round: 1, block, wall, value });
  const rows = [
    row("cpu-medium", 60, 60, null), row("cpu-medium", 64, 64, 2), row("cpu-medium", 62, 62, 4),
    row("gpu-medium", 1, 9, null), row("gpu-medium", 3, 11, 1), row("gpu-medium", 2, 10, 3),
    row("gpu-medium", 4, 12, 5),
  ];
  const frames = { "cpu-medium": { callbacks: 30, presented: 60 }, "gpu-medium": { callbacks: 60, presented: 60 } };
  const s = summariseBench(rows, plan, frames) as Record<"cpu-medium" | "gpu-medium", CopySummary>;

  it("times each copy: what it held the main thread for, and until its copy was ready", () => {
    expect(s["cpu-medium"].block.median).toBe(62);
    expect(s["gpu-medium"].block.median).toBe(2.5);
    expect(s["gpu-medium"].block.max).toBe(4);
    expect(s["gpu-medium"].wall.median).toBe(10.5);
  });

  it("counts reads a second over the copy's own seconds", () => {
    expect(s["cpu-medium"].reads).toBe(3);
    expect(s["cpu-medium"].per_second).toBe(1.5);
    expect(s["gpu-medium"].per_second).toBe(2);
  });

  it("keeps motion values without the first read of a slot (it has nothing to compare with)", () => {
    expect(s["cpu-medium"].value.n).toBe(2);
    expect(s["cpu-medium"].value.median).toBe(3);
    expect(s["gpu-medium"].value.n).toBe(3);
    expect(s["gpu-medium"].value.median).toBe(3);
  });

  it("counts the frames the page never saw: presented by the video, no callback", () => {
    expect(s["cpu-medium"].frames).toBe(60);
    expect(s["cpu-medium"].missed).toBe(30);
    expect(s["gpu-medium"].missed).toBe(0);
  });

  it("leaves out a copy the plan doesn't run", () => {
    expect(Object.keys(s)).toEqual(["cpu-medium", "gpu-medium"]);
  });
});
