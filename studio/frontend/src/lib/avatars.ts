// Deterministic avatar palette + helpers — one colour per agent/pipeline name, stable across renders.
export const AVATAR_BG = ["#dbeafe", "#dcfce7", "#fef3c7", "#fce7f3", "#e0e7ff", "#ffedd5", "#ccfbf1"];
export const AVATAR_FG = ["#2563eb", "#16a34a", "#d97706", "#db2777", "#4f46e5", "#ea580c", "#0d9488"];

export function hashIdx(s: string, n: number): number {
  let h = 0;
  for (const c of s || "") h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return h % n;
}

export function initials(s: string): string {
  const p = (s || "?").replace(/[_-]/g, " ").trim().split(/\s+/);
  return ((p[0]?.[0] || "") + (p[1]?.[0] || p[0]?.[1] || "")).toUpperCase();
}
