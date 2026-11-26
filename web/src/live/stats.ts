/** Percentiles as numpy.percentile takes them by default (linear between
 *  the two closest ranks), so the app's summaries match Python's. */
export function percentile(xs: readonly number[], q: number): number {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const pos = (s.length - 1) * (q / 100);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

export function median(xs: readonly number[]): number {
  return percentile(xs, 50);
}
