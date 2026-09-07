import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { History as HistoryIcon, ScanEye, Loader2, ChevronRight, Download } from "lucide-react";
import GradeBadge from "../components/GradeBadge.jsx";
import { Screening as Store } from "../lib/store.js";

export default function HistoryPage() {
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
        <p className="text-sm">Loading screenings…</p>
      </div>
    );

  return (
    <div className="py-10 max-w-5xl mx-auto">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center ring-1 ring-primary/15">
            <HistoryIcon className="h-5 w-5 text-primary" strokeWidth={1.6} />
          </div>
          <div>
            <h1 className="font-display text-2xl sm:text-3xl font-semibold tracking-tight">
              Screening history
            </h1>
            <p className="text-sm text-muted-foreground">
              All past analyses with their explanations.
            </p>
          </div>
        </div>
        {records.length > 0 && (
          <button
            onClick={() => window.print()}
            className="inline-flex items-center gap-2 rounded-full bg-primary text-primary-foreground px-4 py-2 text-sm font-medium shadow-sm hover:shadow-md transition shrink-0"
          >
            <Download className="h-4 w-4" strokeWidth={1.8} />
            Export all
          </button>
        )}
      </div>

      {records.length === 0 ? (
        <div className="mt-12 rounded-2xl bg-card ring-1 ring-border p-12 text-center">
          <div className="mx-auto h-12 w-12 rounded-2xl bg-muted flex items-center justify-center">
            <ScanEye className="h-6 w-6 text-muted-foreground" strokeWidth={1.5} />
          </div>
          <h2 className="mt-5 font-display text-lg font-semibold">
            No screenings yet
          </h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Run your first explainable retinopathy screening.
          </p>
          <Link
            to="/screening"
            className="mt-6 inline-flex items-center gap-2 rounded-full bg-primary text-primary-foreground px-5 py-2.5 text-sm font-medium hover:shadow-md transition"
          >
            <ScanEye className="h-4 w-4" strokeWidth={1.8} /> Start a screening
          </Link>
        </div>
      ) : (
        <div className="mt-8 grid sm:grid-cols-2 gap-4">
          {records.map((r, i) => (
            <motion.div
              key={r.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: i * 0.04 }}
            >
              <Link
                to={`/results/${r.id}`}
                className="group block rounded-2xl bg-card ring-1 ring-border hover:ring-primary/30 hover:shadow-md transition-all overflow-hidden"
              >
                <div className="flex gap-4 p-4">
                  <div className="h-20 w-20 rounded-xl overflow-hidden bg-muted shrink-0">
                    <img
                      src={r.image_url}
                      alt="Scan"
                      className="h-full w-full object-cover"
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <GradeBadge grade={r.dr_grade} />
                      <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-primary group-hover:translate-x-0.5 transition" />
                    </div>
                    <p className="mt-2 text-sm text-muted-foreground line-clamp-2">
                      {r.summary}
                    </p>
                    <p className="mt-2 text-xs text-muted-foreground">
                      {new Date(r.created_date).toLocaleDateString(undefined, {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                      })}
                      {r.patient_id && ` · ${r.patient_id}`}
                    </p>
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
