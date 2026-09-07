import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  LayoutDashboard,
  Layers,
  CalendarDays,
  TriangleAlert,
  ScanEye,
  Loader2,
  Activity,
  ChevronRight,
} from "lucide-react";
import GradeBadge from "../components/GradeBadge.jsx";
import { Screening as Store } from "../lib/store.js";
import { GRADE_ORDER, GRADE_HEX, URGENCY_HEX, URGENCY_SHORT } from "../lib/grades.js";

const monthKey = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
const monthLabel = (d) =>
  d.toLocaleDateString(undefined, { month: "short", year: "2-digit" });

function Stat({ icon: Icon, label, value, sub }) {
  return (
    <div className="rounded-2xl bg-card ring-1 ring-border p-5">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">{label}</span>
        {Icon && (
          <div className="h-9 w-9 rounded-xl bg-primary/10 flex items-center justify-center ring-1 ring-primary/15">
            <Icon className="h-4 w-4 text-primary" strokeWidth={1.7} />
          </div>
        )}
      </div>
      <div className="mt-3 font-display text-3xl font-semibold text-foreground">
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}

function GradeOverTime({ records }) {
  const data = useMemo(() => {
    const now = new Date();
    const months = [];
    for (let i = 11; i >= 0; i--)
      months.push(new Date(now.getFullYear(), now.getMonth() - i, 1));
    return months.map((m) => {
      const key = monthKey(m);
      const row = { label: monthLabel(m) };
      for (const g of GRADE_ORDER) row[g] = 0;
      for (const r of records) {
        if (monthKey(new Date(r.created_date)) === key && row[r.dr_grade] !== undefined)
          row[r.dr_grade] += 1;
      }
      return row;
    });
  }, [records]);

  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis allowDecimals={false} tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
          <Tooltip
            contentStyle={{
              borderRadius: 12,
              border: "1px solid hsl(var(--border))",
              background: "hsl(var(--card))",
              fontSize: 12,
            }}
            cursor={{ fill: "hsl(var(--muted))", opacity: 0.5 }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {GRADE_ORDER.map((g) => (
            <Bar
              key={g}
              dataKey={g}
              stackId="a"
              fill={GRADE_HEX[g]}
              radius={g === "No DR" ? [4, 4, 0, 0] : [0, 0, 0, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function UrgencyDonut({ records }) {
  const counts = { routine: 0, "follow-up": 0, urgent: 0, immediate: 0 };
  for (const r of records) {
    const k = r.urgency && counts[r.urgency] !== undefined ? r.urgency : "routine";
    counts[k] += 1;
  }
  const data = Object.keys(counts)
    .map((k) => ({ name: URGENCY_SHORT[k], value: counts[k], color: URGENCY_HEX[k] }))
    .filter((d) => d.value > 0);
  const total = data.reduce((a, d) => a + d.value, 0);

  if (total === 0)
    return (
      <p className="h-72 flex items-center justify-center text-sm text-muted-foreground">
        No urgency data yet.
      </p>
    );

  return (
    <div className="relative h-72">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            innerRadius={62}
            outerRadius={92}
            paddingAngle={3}
            strokeWidth={0}
          >
            {data.map((d) => (
              <Cell key={d.name} fill={d.color} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              borderRadius: 12,
              border: "1px solid hsl(var(--border))",
              background: "hsl(var(--card))",
              fontSize: 12,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <span className="font-display text-2xl font-semibold text-foreground">
          {total}
        </span>
        <span className="text-xs text-muted-foreground">cases</span>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [records, setRecords] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        setRecords(await Store.list());
      } catch {
        setRecords([]);
      }
    })();
  }, []);

  if (records === null)
    return (
      <div className="py-32 flex flex-col items-center gap-3 text-muted-foreground">
        <Loader2 className="h-7 w-7 text-primary animate-spin" />
        <p className="text-sm">Loading statistics…</p>
      </div>
    );

  const total = records.length;
  const thisMonth = records.filter(
    (r) => monthKey(new Date(r.created_date)) === monthKey(new Date()),
  ).length;
  const urgent = records.filter(
    (r) => r.urgency === "urgent" || r.urgency === "immediate",
  ).length;
  const confs = records.map((r) => r.confidence).filter((c) => typeof c === "number");
  const avgConf = confs.length
    ? Math.round((confs.reduce((a, c) => a + c, 0) / confs.length) * 100)
    : 0;

  return (
    <div className="py-10">
      <div className="flex items-center gap-2.5">
        <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center ring-1 ring-primary/15">
          <LayoutDashboard className="h-5 w-5 text-primary" strokeWidth={1.6} />
        </div>
        <div>
          <h1 className="font-display text-2xl sm:text-3xl font-semibold tracking-tight">
            Clinician dashboard
          </h1>
          <p className="text-sm text-muted-foreground">
            Aggregate screening statistics across all cases.
          </p>
        </div>
      </div>

      {total === 0 ? (
        <div className="mt-12 rounded-2xl bg-card ring-1 ring-border p-12 text-center">
          <div className="mx-auto h-12 w-12 rounded-2xl bg-muted flex items-center justify-center">
            <Activity className="h-6 w-6 text-muted-foreground" strokeWidth={1.5} />
          </div>
          <h2 className="mt-5 font-display text-lg font-semibold">
            No data to visualize yet
          </h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Run a few screenings and the dashboard will populate automatically.
          </p>
          <Link
            to="/screening"
            className="mt-6 inline-flex items-center gap-2 rounded-full bg-primary text-primary-foreground px-5 py-2.5 text-sm font-medium hover:shadow-md transition"
          >
            <ScanEye className="h-4 w-4" strokeWidth={1.8} /> Start a screening
          </Link>
        </div>
      ) : (
        <>
          <div className="mt-8 grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Stat icon={Layers} label="Total screenings" value={total} sub="All completed analyses" />
            <Stat
              icon={CalendarDays}
              label="This month"
              value={thisMonth}
              sub="Screenings in the current month"
            />
            <Stat
              icon={TriangleAlert}
              label="Urgent referrals"
              value={urgent}
              sub="Urgent + immediate cases"
            />
            <Stat icon={ScanEye} label="Avg. confidence" value={`${avgConf}%`} sub="Across all screenings" />
          </div>

          <div className="mt-6 grid lg:grid-cols-5 gap-5">
            <div className="lg:col-span-3 rounded-2xl bg-card ring-1 ring-border p-6">
              <h2 className="font-display text-base font-semibold">
                Grade distribution over time
              </h2>
              <p className="mt-1 text-xs text-muted-foreground">
                Screenings per month, stacked by DR severity
              </p>
              <div className="mt-4">
                <GradeOverTime records={records} />
              </div>
            </div>
            <div className="lg:col-span-2 rounded-2xl bg-card ring-1 ring-border p-6">
              <h2 className="font-display text-base font-semibold">Urgency levels</h2>
              <p className="mt-1 text-xs text-muted-foreground">
                Referral urgency across all cases
              </p>
              <div className="mt-4">
                <UrgencyDonut records={records} />
              </div>
            </div>
          </div>

          <div className="mt-6 rounded-2xl bg-card ring-1 ring-border p-6">
            <div className="flex items-center justify-between">
              <h2 className="font-display text-base font-semibold">Recent screenings</h2>
              <Link
                to="/history"
                className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
              >
                View all <ChevronRight className="h-3.5 w-3.5" />
              </Link>
            </div>
            <div className="mt-4 divide-y divide-border/60">
              {records.slice(0, 5).map((r, i) => (
                <motion.div
                  key={r.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                >
                  <Link
                    to={`/results/${r.id}`}
                    className="flex items-center gap-4 py-3 group"
                  >
                    <div className="h-11 w-11 rounded-lg overflow-hidden bg-muted shrink-0">
                      <img src={r.image_url} alt="Scan" className="h-full w-full object-cover" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <GradeBadge grade={r.dr_grade} />
                        {r.patient_id && (
                          <span className="text-xs text-muted-foreground">
                            {r.patient_id}
                          </span>
                        )}
                      </div>
                      <p className="mt-1 text-sm text-muted-foreground truncate max-w-md">
                        {r.summary}
                      </p>
                    </div>
                    <div className="text-right shrink-0">
                      <p className="text-xs text-muted-foreground">
                        {new Date(r.created_date).toLocaleDateString(undefined, {
                          day: "numeric",
                          month: "short",
                          year: "numeric",
                        })}
                      </p>
                      <ChevronRight className="ml-auto mt-1 h-4 w-4 text-muted-foreground group-hover:text-primary group-hover:translate-x-0.5 transition" />
                    </div>
                  </Link>
                </motion.div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
