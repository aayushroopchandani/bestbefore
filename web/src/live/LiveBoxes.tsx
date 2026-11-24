/**
 * The boxes over the video: a canvas the size of the video's element, the
 * boxes mapped from the frame by the same scale and offset as the video
 * (object-fit: contain), so each box sits on what the model saw. Dates
 * solid, the keyword marks (due, prod, code) thin and labelled.
 */
import { useEffect, useRef } from "react";
import type { FrameBox } from "../detector/types";
import { contain, toView } from "./fit";

const COLOURS = ["#22d3ee", "#f472b6", "#a3e635", "#fbbf24"]; // date, due, prod, code

interface Props {
  boxes: FrameBox[];
  frameWidth: number;
  frameHeight: number;
  classes: string[];
}

export function LiveBoxes({ boxes, frameWidth, frameHeight, classes }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const draw = () => {
      const dpr = devicePixelRatio || 1;
      const { width, height } = canvas.getBoundingClientRect();
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const ctx = canvas.getContext("2d");
      if (!ctx || !frameWidth || !frameHeight) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const fit = contain(frameWidth, frameHeight, width, height);
      ctx.font = "12px system-ui, sans-serif";
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
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [boxes, frameWidth, frameHeight, classes]);

  return <canvas ref={ref} className="pointer-events-none absolute inset-0 h-full w-full" />;
}
