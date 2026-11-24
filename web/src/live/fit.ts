/** Where the video sits in its element when shown whole (object-fit:
 *  contain): one scale for both axes, and the offset that centres it. */
export interface Fit {
  scale: number;
  x: number;
  y: number;
}

export function contain(frameWidth: number, frameHeight: number, viewWidth: number, viewHeight: number): Fit {
  const scale = Math.min(viewWidth / frameWidth, viewHeight / frameHeight);
  return { scale, x: (viewWidth - frameWidth * scale) / 2, y: (viewHeight - frameHeight * scale) / 2 };
}

/** A box in frame pixels to the view's pixels. */
export function toView(box: { x1: number; y1: number; x2: number; y2: number }, fit: Fit): [number, number, number, number] {
  return [fit.x + box.x1 * fit.scale, fit.y + box.y1 * fit.scale, fit.x + box.x2 * fit.scale, fit.y + box.y2 * fit.scale];
}
