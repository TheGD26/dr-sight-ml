import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ScanEye,
  ArrowRight,
  Eye,
  Brain,
  Target,
  Activity,
  Stethoscope,
  ClipboardCheck,
  Zap,
} from "lucide-react";
import GradeBadge from "../components/GradeBadge.jsx";
import { GRADE_META } from "../lib/grades.js";
import { SAMPLE_IMAGE_URL } from "../lib/config.js";

const rise = {
  hidden: { opacity: 0, y: 16 },
  show: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] },
  }),
};

function Stat({ value, label }) {
  return (
    <div>
      <div className="font-display text-3xl font-semibold text-foreground">
        {value}
      </div>
      <div className="text-sm text-muted-foreground mt-1">{label}</div>
    </div>
  );
}

const FEATURES = [
  {
    icon: Eye,
    title: "Region-level heatmaps",
    body: "Lesion-level attention overlays show exactly where microaneurysms, hemorrhages, and exudates were detected.",
  },
  {
    icon: Brain,
    title: "Findings breakdown",
    body: "A structured list of clinical signs with per-finding severity and a description of what is visible and where.",
  },
  {
    icon: ClipboardCheck,
    title: "Actionable recommendation",
    body: "A graded follow-up suggestion — routine rescreen, refer, or urgent referral — based on severity.",
  },
  {
    icon: Zap,
    title: "Fast turnaround",
    body: "From upload to a fully explained result in roughly thirty seconds, ready for a clinician to review.",
  },
  {
    icon: Stethoscope,
    title: "Clinician-in-the-loop",
    body: "Designed as decision support: the AI explains, the clinician decides. Notes can be appended to any case.",
  },
  {
    icon: Activity,
    title: "History & tracking",
    body: "Every screening is stored with its image, grade, and explanation for longitudinal review.",
  },
];

const STEPS = [
  { n: "01", title: "Upload", body: "A macula-centered fundus photograph." },
  { n: "02", title: "Analyze", body: "Vision AI assesses lesions and grades severity." },
  { n: "03", title: "Explain", body: "Heatmap + findings show what drove the grade." },
  { n: "04", title: "Act", body: "A recommendation and clinician notes close the loop." },
];

export default function Home() {
  return (
    <div className="pb-8">
      {/* hero */}
      <section className="relative pt-16 pb-20 sm:pt-24">
        <div className="absolute inset-0 -z-10 grain opacity-60" />
        <div className="absolute top-0 right-0 -z-10 h-[480px] w-[480px] rounded-full bg-primary/[0.08] blur-3xl" />
        <div className="grid lg:grid-cols-12 gap-10 items-center">
          <motion.div
            className="lg:col-span-6"
            initial="hidden"
            animate="show"
            variants={rise}
          >
            <span className="inline-flex items-center gap-2 rounded-full bg-primary/[0.08] px-3 py-1 text-xs font-medium text-primary ring-1 ring-primary/15">
              <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
              Explainable AI · Clinical Decision Support
            </span>
            <h1 className="mt-6 font-display text-4xl sm:text-5xl lg:text-6xl font-semibold leading-[1.05] tracking-tight text-balance text-foreground">
              See what the AI sees.
              <span className="block text-primary">
                Screen diabetic retinopathy with clarity.
              </span>
            </h1>
            <p className="mt-6 text-lg text-muted-foreground max-w-xl leading-relaxed">
              Upload a retinal fundus photograph and receive a severity grade, a
              plain-language summary, and a transparent, region-level explanation
              of exactly which lesions drove the decision.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                to="/screening"
                className="group inline-flex items-center gap-2 rounded-full bg-primary text-primary-foreground px-6 py-3 text-sm font-medium shadow-sm hover:shadow-lg transition-all hover:gap-3"
              >
                <ScanEye className="h-4 w-4" strokeWidth={1.8} />
                Start a screening
                <ArrowRight
                  className="h-4 w-4 transition-transform group-hover:translate-x-0.5"
                  strokeWidth={1.8}
                />
              </Link>
              <Link
                to="/learn"
                className="inline-flex items-center gap-2 rounded-full px-5 py-3 text-sm font-medium text-foreground ring-1 ring-border hover:bg-muted transition-colors"
              >
                How it works
              </Link>
            </div>
            <div className="mt-12 grid grid-cols-3 gap-6 max-w-md">
              <Stat value="5" label="Severity grades" />
              <Stat value="~30s" label="To result" />
              <Stat value="100%" label="Explanation coverage" />
            </div>
          </motion.div>

          <motion.div
            className="lg:col-span-6"
            initial="hidden"
            animate="show"
            custom={1}
            variants={rise}
          >
            <div className="relative">
              <div className="absolute -inset-4 -z-10 rounded-[2rem] bg-gradient-to-br from-primary/10 via-accent/40 to-transparent blur-2xl" />
              <div className="relative rounded-[1.75rem] bg-card ring-1 ring-border shadow-xl overflow-hidden">
                <div className="flex items-center justify-between px-5 py-3 border-b border-border/60">
                  <div className="flex items-center gap-2">
                    <Eye className="h-4 w-4 text-primary" strokeWidth={1.7} />
                    <span className="text-sm font-medium">Live analysis</span>
                  </div>
                  <GradeBadge grade="Moderate NPDR" />
                </div>
                <div className="relative aspect-[4/3] bg-muted">
                  <img
                    src={SAMPLE_IMAGE_URL}
                    alt="Retinal fundus"
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                  <svg
                    viewBox="0 0 100 100"
                    preserveAspectRatio="none"
                    className="absolute inset-0 w-full h-full"
                  >
                    <defs>
                      <radialGradient id="hero-heat" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="rgba(255,235,59,0.8)" />
                        <stop offset="55%" stopColor="rgba(255,87,34,0.45)" />
                        <stop offset="100%" stopColor="rgba(244,67,54,0)" />
                      </radialGradient>
                    </defs>
                    <circle cx="38" cy="44" r="9" fill="url(#hero-heat)" opacity="0.8" />
                    <circle cx="62" cy="58" r="6" fill="url(#hero-heat)" opacity="0.7" />
                    <circle cx="48" cy="68" r="4" fill="url(#hero-heat)" opacity="0.6" />
                  </svg>
                  <div className="absolute bottom-3 left-3 right-3 rounded-lg bg-foreground/85 backdrop-blur px-3 py-2 text-xs text-background flex items-center justify-between">
                    <span className="font-medium">Microaneurysms · Hemorrhages</span>
                    <span className="opacity-70">Confidence 0.87</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* trust bar */}
      <section className="py-8 border-y border-border/60">
        <div className="flex flex-wrap items-center justify-center gap-x-10 gap-y-4 text-sm text-muted-foreground">
          <span className="flex items-center gap-2">
            <Stethoscope className="h-4 w-4 text-primary" strokeWidth={1.7} />
            Clinician-in-the-loop
          </span>
          <span className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" strokeWidth={1.7} />
            Vision-language reasoning
          </span>
          <span className="flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" strokeWidth={1.7} />
            Region-level explanations
          </span>
          <span className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-primary" strokeWidth={1.7} />
            International DR scale
          </span>
        </div>
      </section>

      {/* why */}
      <section className="py-20">
        <div className="max-w-2xl">
          <p className="text-sm font-medium uppercase tracking-[0.18em] text-primary">
            Why Retina
          </p>
          <h2 className="mt-3 font-display text-3xl sm:text-4xl font-semibold tracking-tight text-balance">
            Explainability isn't an add-on. It's the product.
          </h2>
          <p className="mt-4 text-muted-foreground text-lg">
            Every prediction comes with a heatmap of the regions that mattered, a
            breakdown of the clinical findings, and a recommendation you can act
            on.
          </p>
        </div>
        <div className="mt-12 grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map((f, i) => {
            const Icon = f.icon;
            return (
              <motion.div
                key={f.title}
                initial="hidden"
                whileInView="show"
                viewport={{ once: true, margin: "-80px" }}
                custom={i}
                variants={rise}
                className="group rounded-2xl bg-card ring-1 ring-border p-6 hover:ring-primary/30 hover:shadow-md transition-all"
              >
                <div className="h-11 w-11 rounded-xl bg-primary/10 flex items-center justify-center ring-1 ring-primary/15 group-hover:bg-primary/15 transition-colors">
                  <Icon className="h-5 w-5 text-primary" strokeWidth={1.6} />
                </div>
                <h3 className="mt-5 font-display text-lg font-semibold">
                  {f.title}
                </h3>
                <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
                  {f.body}
                </p>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* workflow */}
      <section className="py-16 rounded-3xl bg-card ring-1 ring-border overflow-hidden">
        <div className="px-6 sm:px-12 py-12">
          <div className="max-w-2xl">
            <p className="text-sm font-medium uppercase tracking-[0.18em] text-primary">
              Workflow
            </p>
            <h2 className="mt-3 font-display text-3xl sm:text-4xl font-semibold tracking-tight">
              From scan to explained result
            </h2>
          </div>
          <div className="mt-12 grid md:grid-cols-4 gap-6">
            {STEPS.map((s, i) => (
              <motion.div
                key={s.n}
                initial="hidden"
                whileInView="show"
                viewport={{ once: true }}
                custom={i}
                variants={rise}
                className="relative"
              >
                <div className="font-display text-4xl font-semibold text-primary/30">
                  {s.n}
                </div>
                <h3 className="mt-3 font-medium text-foreground">{s.title}</h3>
                <p className="mt-1.5 text-sm text-muted-foreground">{s.body}</p>
                {i < 3 && (
                  <div className="hidden md:block absolute top-5 -right-3 h-px w-6 bg-border" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* scale */}
      <section className="py-20">
        <div className="max-w-2xl">
          <p className="text-sm font-medium uppercase tracking-[0.18em] text-primary">
            The scale
          </p>
          <h2 className="mt-3 font-display text-3xl sm:text-4xl font-semibold tracking-tight">
            International Clinical DR grading
          </h2>
        </div>
        <div className="mt-10 grid sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {Object.entries(GRADE_META).map(([name, meta], i) => (
            <motion.div
              key={name}
              initial="hidden"
              whileInView="show"
              viewport={{ once: true }}
              custom={i}
              variants={rise}
              className="rounded-2xl bg-card ring-1 ring-border p-5"
            >
              <div className={`h-1.5 w-10 rounded-full ${meta.dot}`} />
              <h3 className="mt-4 font-display text-base font-semibold">{name}</h3>
              <p className="mt-1.5 text-xs text-muted-foreground leading-relaxed">
                {meta.blurb}
              </p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* cta */}
      <section className="py-12">
        <div className="relative overflow-hidden rounded-3xl bg-primary text-primary-foreground px-8 sm:px-14 py-16">
          <div className="absolute -top-16 -right-10 h-72 w-72 rounded-full bg-white/10 blur-3xl" />
          <div className="relative max-w-2xl">
            <h2 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight text-balance">
              Bring transparency to every screening.
            </h2>
            <p className="mt-4 text-primary-foreground/80 text-lg">
              Run your first explainable retinopathy screening in under a minute.
            </p>
            <Link
              to="/screening"
              className="mt-8 inline-flex items-center gap-2 rounded-full bg-primary-foreground text-primary px-6 py-3 text-sm font-medium hover:gap-3 transition-all"
            >
              <ScanEye className="h-4 w-4" strokeWidth={1.8} />
              Start a screening
              <ArrowRight className="h-4 w-4" strokeWidth={1.8} />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
