/**
 * A frame to the model's input, and boxes back to the frame: Ultralytics
 * 8.4.177's LetterBox and scale_boxes, step for step, so the app's boxes are
 * the ones M2b scored (docs/m2c-live-boxes.md, "One detection"; tested
 * against fixtures those functions made, letterbox.test.ts).
 *
 * No React and no app state: this folder is the part Scanify can take.
 */

export const IMGSZ = 1280; // the model's size: the long side always 1,280 px
export const STRIDE = 32; // the model's largest stride: tight inputs are multiples of it
export const PAD_VALUE = 114; // Ultralytics' grey around the photo

/** Python's round(): to the nearest integer, halves to the even one (2.5 to 2).
 *  Math.round takes halves up (2.5 to 3), which would move a pad by a pixel.
 *  `+ 0` makes JavaScript's -0 (Math.round(-0.1)) the integer 0 Python gives. */
export function pyRound(x: number): number {
  return (Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x)) + 0;
}

/** Where a frame goes in the model's input. */
export interface Letterbox {
  frameWidth: number;
  frameHeight: number;
  newWidth: number; // the frame scaled so its long side is IMGSZ
  newHeight: number;
  left: number; // grey around it
  top: number;
  right: number;
  bottom: number;
  inputWidth: number; // the model's input: newWidth + left + right
  inputHeight: number;
}

/** LetterBox.get_params: square pads to IMGSZ x IMGSZ, tight only to the
 *  next multiple of STRIDE. The frame is scaled up or down to IMGSZ on its
 *  long side (scaleup), and centred, Ultralytics' -0.1 / +0.1 putting an
 *  odd pixel of padding on the bottom and right. */
export function letterboxParams(frameWidth: number, frameHeight: number, tight = false,
  imgsz = IMGSZ, stride = STRIDE): Letterbox {
  const r = Math.min(imgsz / frameHeight, imgsz / frameWidth);
  const newWidth = pyRound(frameWidth * r);
  const newHeight = pyRound(frameHeight * r);
  let dw = imgsz - newWidth;
  let dh = imgsz - newHeight;
  if (tight) {
    dw %= stride;
    dh %= stride;
  }
  dw /= 2;
  dh /= 2;
  const top = pyRound(dh - 0.1);
  const bottom = pyRound(dh + 0.1);
  const left = pyRound(dw - 0.1);
  const right = pyRound(dw + 0.1);
  return {
    frameWidth, frameHeight, newWidth, newHeight, left, top, right, bottom,
    inputWidth: newWidth + left + right, inputHeight: newHeight + top + bottom,
  };
}

/** Draws `source` letterboxed into `ctx`, whose canvas is the input's size:
 *  grey 114 everywhere, then the frame scaled into place. The browser's own
 *  resize, not OpenCV's INTER_LINEAR: the parity benches measure whether
 *  that moves a box. */
export function drawLetterboxed(ctx: OffscreenCanvasRenderingContext2D, source: CanvasImageSource, lb: Letterbox): void {
  ctx.fillStyle = `rgb(${PAD_VALUE}, ${PAD_VALUE}, ${PAD_VALUE})`;
  ctx.fillRect(0, 0, lb.inputWidth, lb.inputHeight);
  ctx.drawImage(source, lb.left, lb.top, lb.newWidth, lb.newHeight);
}

/** A canvas's RGBA bytes as the model's input: RGB, channels first, 0 to 1
 *  in float32 (Ultralytics divides the bytes by 255 in float32; a float64
 *  quotient rounded to float32 is the same number). Written into `out` if
 *  given, so a loop needn't allocate one a frame. */
export function toTensor<T extends Float32Array>(rgba: Uint8ClampedArray, width: number, height: number,
  out: T = new Float32Array(3 * width * height) as T): T {
  const plane = width * height;
  for (let p = 0, i = 0; p < plane; p++, i += 4) {
    out[p] = rgba[i] / 255;
    out[plane + p] = rgba[i + 1] / 255;
    out[2 * plane + p] = rgba[i + 2] / 255;
  }
  return out;
}

/** scale_boxes, as Ultralytics' predictor calls it: xyxy boxes in the
 *  model's input (inputWidth x inputHeight) back to the frame, clipped to
 *  it. Like Ultralytics, it works the gains and pads out from the two
 *  shapes alone, each side's gain from its own rounded size, and it runs
 *  in float32 (torch's), hence Math.fround. */
export function scaleBoxes(boxes: readonly (readonly number[])[], inputWidth: number, inputHeight: number,
  frameWidth: number, frameHeight: number): number[][] {
  const f = Math.fround;
  const gain = Math.min(inputHeight / frameHeight, inputWidth / frameWidth);
  const newHeight = pyRound(frameHeight * gain);
  const newWidth = pyRound(frameWidth * gain);
  const gainY = f(newHeight / frameHeight);
  const gainX = f(newWidth / frameWidth);
  const padX = pyRound((inputWidth - newWidth) / 2 - 0.1);
  const padY = pyRound((inputHeight - newHeight) / 2 - 0.1);
  const clip = (v: number, hi: number) => Math.min(Math.max(v, 0), hi);
  return boxes.map(([x1, y1, x2, y2]) => [
    clip(f(f(x1 - padX) / gainX), frameWidth),
    clip(f(f(y1 - padY) / gainY), frameHeight),
    clip(f(f(x2 - padX) / gainX), frameWidth),
    clip(f(f(y2 - padY) / gainY), frameHeight),
  ]);
}
