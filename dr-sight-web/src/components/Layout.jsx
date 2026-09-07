import { useState } from "react";
import { NavLink, Link, Outlet, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  ScanEye,
  Home,
  LayoutDashboard,
  History,
  BookOpen,
  Sparkles,
  Menu,
  X,
} from "lucide-react";
import { cn } from "../lib/cn.js";

const NAV = [
  { label: "Home", to: "/", icon: Home },
  { label: "Dashboard", to: "/dashboard", icon: LayoutDashboard },
  { label: "New Screening", to: "/screening", icon: ScanEye },
  { label: "History", to: "/history", icon: History },
  { label: "Learn", to: "/learn", icon: BookOpen },
];

export default function Layout() {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-background/80 border-b border-border/60">
        <div className="max-w-7xl mx-auto px-5 sm:px-8">
          <div className="flex h-16 items-center justify-between">
            <Link to="/" className="flex items-center gap-2.5 group">
              <div className="relative h-9 w-9 rounded-xl bg-primary/10 flex items-center justify-center ring-1 ring-primary/20">
                <ScanEye className="h-5 w-5 text-primary" strokeWidth={1.6} />
              </div>
              <div className="leading-none">
                <span className="font-display text-lg font-semibold tracking-tight text-foreground">
                  Retina
                </span>
                <span className="block text-[10px] uppercase tracking-[0.18em] text-muted-foreground mt-0.5">
                  Explainable DR Screening
                </span>
              </div>
            </Link>

            <nav className="hidden md:flex items-center gap-1">
              {NAV.map((item) => {
                const active = pathname === item.to;
                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    className={cn(
                      "relative px-4 py-2 text-sm font-medium rounded-lg transition-colors",
                      active
                        ? "text-primary"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {item.label}
                    {active && (
                      <motion.span
                        layoutId="nav-active"
                        className="absolute inset-0 -z-10 rounded-lg bg-primary/[0.08]"
                        transition={{ type: "spring", stiffness: 380, damping: 30 }}
                      />
                    )}
                  </NavLink>
                );
              })}
            </nav>

            <div className="hidden md:block">
              <Link
                to="/screening"
                className="inline-flex items-center gap-2 rounded-full bg-primary text-primary-foreground px-4 py-2 text-sm font-medium shadow-sm hover:shadow-md transition-shadow"
              >
                <Sparkles className="h-3.5 w-3.5" strokeWidth={2} />
                Start Screening
              </Link>
            </div>

            <button
              className="md:hidden p-2 rounded-lg hover:bg-muted text-foreground"
              onClick={() => setOpen((v) => !v)}
              aria-label="Menu"
            >
              {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>

        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="md:hidden overflow-hidden border-t border-border/60 bg-background"
            >
              <div className="px-5 py-4 space-y-1">
                {NAV.map((item) => {
                  const active = pathname === item.to;
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.to}
                      to={item.to}
                      onClick={() => setOpen(false)}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium",
                        active
                          ? "bg-primary/10 text-primary"
                          : "text-muted-foreground hover:bg-muted",
                      )}
                    >
                      <Icon className="h-4 w-4" strokeWidth={1.7} />
                      {item.label}
                    </Link>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      <main className="max-w-7xl mx-auto px-5 sm:px-8">
        <Outlet />
      </main>

      <footer className="mt-24 border-t border-border/60">
        <div className="max-w-7xl mx-auto px-5 sm:px-8 py-10 flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-2.5">
            <ScanEye className="h-4 w-4 text-primary" strokeWidth={1.6} />
            <span className="font-display font-medium text-foreground">Retina</span>
            <span className="text-muted-foreground/60">·</span>
            <span>Explainable AI for DR Screening</span>
          </div>
          <p className="text-xs max-w-md text-center sm:text-right">
            Decision support, not a diagnosis. Always confirm findings with a
            qualified clinician.
          </p>
        </div>
      </footer>
    </div>
  );
}
