// Minimal className joiner — same role as `clsx`, no dependency.
export function cn(...parts) {
  return parts
    .flat(Infinity)
    .filter((p) => typeof p === "string" && p.length > 0)
    .join(" ");
}
