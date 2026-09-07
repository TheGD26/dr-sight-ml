import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { BookOpen, Eye, Target, Brain, AlertTriangle, ScanEye, ArrowRight } from "lucide-react";
import GradeBadge from "../components/GradeBadge.jsx";
import { GRADE_META } from "../lib/grades.js";

const rise = {
  hidden: { opacity: 0, y: 14 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.5, delay: i * 0.06 } }),
};

const LESIONS = [
  ["Microaneurysms", "tiny red dots, the earliest sign of DR."],
  ["Hemorrhages", "flame-shaped or blot bleeds in the retina."],
  ["Hard exudates", "yellow-white lipid deposits, often near the macula."],
  ["Cotton-wool spots", "fluffy white patches of nerve-fiber-layer ischemia."],
  ["Venous beading & IRMA", "signs of worsening ischemia."],
  ["Neovascularization", "new, fragile vessels — the hallmark of proliferative DR."],
];

const LIMITS = [
  "This tool provides decision support, not a diagnosis.",
  "Image quality affects accuracy — use a clear, macula-centered fundus photo.",
  "Explanations are model-generated and should be verified by a clinician.",
  "It does not replace a comprehensive dilated eye examination.",
];

export default function Learn() {
  return (
    <div className="py-10 max-w-3xl mx-auto">
      <div className="text-center">
        <span className="inline-flex items-center gap-2 rounded-full bg-primary/[0.08] px-3 py-1 text-xs font-medium text-primary ring-1 ring-primary/15">
          <BookOpen className="h-3.5 w-3.5" /> Guide
        </span>
        <h1 className="mt-5 font-display text-3xl sm:text-4xl font-semibold tracking-tight">
          Understanding diabetic retinopathy &amp; the AI
        </h1>
        <p className="mt-3 text-muted-foreground">
          What the model looks for, how it explains itself, and how to read a
          result.
        </p>
      </div>

      <div className="mt-12 space-y-10">
        <motion.section
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          variants={rise}
          className="rounded-2xl bg-card ring-1 ring-border p-7"
        >
          <div className="flex items-center gap-2.5 mb-4">
            <Eye className="h-5 w-5 text-primary" strokeWidth={1.6} />
            <h2 className="font-display text-xl font-semibold">
              What is diabetic retinopathy?
            </h2>
          </div>
          <p className="text-muted-foreground leading-relaxed">
            Diabetic retinopathy is a complication of diabetes caused by damage
            to the blood vessels of the light-sensitive tissue at the back of the
            eye (the retina). It often develops silently and, left untreated, is
            a leading cause of vision loss. Early detection through regular
            retinal screening is the single most effective way to prevent
            sight-threatening disease.
          </p>
        </motion.section>

        <motion.section
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          custom={1}
          variants={rise}
          className="rounded-2xl bg-card ring-1 ring-border p-7"
        >
          <div className="flex items-center gap-2.5 mb-4">
            <Target className="h-5 w-5 text-primary" strokeWidth={1.6} />
            <h2 className="font-display text-xl font-semibold">The severity scale</h2>
          </div>
          <p className="text-muted-foreground mb-5">
            The model grades each scan on the International Clinical Diabetic
            Retinopathy scale:
          </p>
          <div className="space-y-3">
            {Object.entries(GRADE_META).map(([name, meta]) => (
              <div key={name} className="flex items-start gap-3">
                <GradeBadge grade={name} />
                <p className="text-sm text-muted-foreground flex-1">{meta.learn}</p>
              </div>
            ))}
          </div>
        </motion.section>

        <motion.section
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          custom={2}
          variants={rise}
          className="rounded-2xl bg-card ring-1 ring-border p-7"
        >
          <div className="flex items-center gap-2.5 mb-4">
            <Brain className="h-5 w-5 text-primary" strokeWidth={1.6} />
            <h2 className="font-display text-xl font-semibold">
              How the AI explains itself
            </h2>
          </div>
          <p className="text-muted-foreground leading-relaxed mb-4">
            Every prediction is paired with an{" "}
            <span className="text-foreground font-medium">
              explainability overlay
            </span>{" "}
            — a heatmap of the retinal regions that most influenced the grade.
            Each highlighted region maps to a clinical finding:
          </p>
          <ul className="space-y-2.5 text-sm text-muted-foreground">
            {LESIONS.map(([name, desc]) => (
              <li key={name} className="flex gap-2.5">
                <span className="text-primary">·</span>
                <span>
                  <span className="text-foreground font-medium">{name}</span> —{" "}
                  {desc}
                </span>
              </li>
            ))}
          </ul>
        </motion.section>

        <motion.section
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          custom={3}
          variants={rise}
          className="rounded-2xl bg-destructive/5 ring-1 ring-destructive/20 p-7"
        >
          <div className="flex items-center gap-2.5 mb-4">
            <AlertTriangle className="h-5 w-5 text-destructive" strokeWidth={1.6} />
            <h2 className="font-display text-xl font-semibold">
              Important limitations
            </h2>
          </div>
          <ul className="space-y-2.5 text-sm text-muted-foreground">
            {LIMITS.map((t) => (
              <li key={t} className="flex gap-2.5">
                <span className="text-destructive">·</span>
                <span>{t}</span>
              </li>
            ))}
          </ul>
        </motion.section>
      </div>

      <div className="mt-12 text-center">
        <Link
          to="/screening"
          className="inline-flex items-center gap-2 rounded-full bg-primary text-primary-foreground px-6 py-3 text-sm font-medium shadow-sm hover:shadow-md hover:gap-3 transition-all"
        >
          <ScanEye className="h-4 w-4" strokeWidth={1.8} /> Run a screening
          <ArrowRight className="h-4 w-4" strokeWidth={1.8} />
        </Link>
      </div>
    </div>
  );
}
