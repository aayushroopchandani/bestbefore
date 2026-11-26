/**
 * The boxes over the video: a canvas the size of the video's element, the
 * boxes mapped from the frame by the same scale and offset as the video
 * (object-fit: contain), so each box sits on what the model saw. Dates
 * solid, the keyword marks (due, prod, code) thin and labelled; all of them
 * dim while the picture isn't still (the gate). Drawn on the next animation
 * frame after a change, in step with the screen.
 */
import { useEffect, useRef } from "react";
import type { FrameBox } from "../detector/types";
import { contain, toView } from "./fit";

const COLOURS = ["#22d3ee", "#f472b6", "#a3e635", "#fbbf24"]; // date, due, prod, code
const DIM = 0.35;

interface Props {
  boxes: readonly FrameBox[];
  frameWidth: number;
  frameHeight: number;
  classes: string[];
  dim?: boolean;
}

export function LiveBoxes({ boxes, frameWidth, frameHeight, classes, dim = false }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    let pending = 0;
    const draw = () => {
      pending = 0;
      const dpr = devicePixelRatio || 1;
      const { width, height } = canvas.getBoundingClientRect();
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      if (!frameWidth || !frameHeight) return;
      const fit = contain(frameWidth, frameHeight, width, height);
      ctx.font = "12px system-ui, sans-serif";
      ctx.globalAlpha = dim ? DIM : 1;
      for (const box of [...boxes].sort((a, b) => b.cls - a.cls)) {
        const [x1, y1, x2, y2] = toView(box, fit);
        const colour = COLOURS[box.cls % COLOURS.length];
        const isDate = box.cls === 0;
        ctx.strokeStyle = colour;
        ctx.lineWidth = isDate ? 3 : 1.5;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        if (!isDate) {
          const label = `${classes[box.cls] ?? box.cls} ${box.conf.toFixed(2)}`;
          ctx.fillStyle = colour;
          ctx.fillText(label, x1, Math.max(12, y1 - 3));
        }
      }
    };
    const schedule = () => {
      if (!pending) pending = requestAnimationFrame(draw);
    };
    schedule();
    const observer = new ResizeObserver(schedule);
    observer.observe(canvas);
    return () => {
      observer.disconnect();
      cancelAnimationFrame(pending);
    };
  }, [boxes, frameWidth, frameHeight, classes, dim]);

  return <canvas ref={ref} className="pointer-events-none absolute inset-0 h-full w-full" />;
}
