import { cn } from "../lib/cn.js";
import { GRADE_META } from "../lib/grades.js";

export default function GradeBadge({ grade, className }) {
  const meta = GRADE_META[grade] || GRADE_META["No DR"];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset",
        meta.color,
        className,
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", meta.dot)} />
      {meta.label}
    </span>
  );
}
