"""Metrics for an ordinal 5-class screening problem.

Headline numbers, in order of importance for a screening tool:
  1. Sensitivity (recall) for referable DR  (grade >= 2)
  2. Quadratic weighted kappa (ordinal agreement)
  3. Per-grade sensitivity / specificity / precision / F1
  4. One-vs-rest AUROC per grade
Plain accuracy is reported but explicitly de-emphasised - grade 0 dominates.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)

from src.config import NUM_CLASSES, REFERABLE_DR_MIN_GRADE


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    return float(
        cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(NUM_CLASSES)))
    )


def _binary_rates(cm: np.ndarray, positive_classes: set[int]) -> dict[str, float]:
    idx = list(range(cm.shape[0]))
    pos = [i for i in idx if i in positive_classes]
    neg = [i for i in idx if i not in positive_classes]
    tp = cm[np.ix_(pos, pos)].sum()
    fn = cm[np.ix_(pos, neg)].sum()
    fp = cm[np.ix_(neg, pos)].sum()
    tn = cm[np.ix_(neg, neg)].sum()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return {
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "support": int(tp + fn),
    }


def compute_all(y_true, y_pred, y_prob=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_grade = {}
    for g in labels:
        per_grade[g] = _binary_rates(cm, {g})

    referable = set(range(REFERABLE_DR_MIN_GRADE, NUM_CLASSES))
    referable_rates = _binary_rates(cm, referable)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(y_true, y_pred),
        "referable_dr": {
            "min_grade": REFERABLE_DR_MIN_GRADE,
            **referable_rates,
            "headline_sensitivity": referable_rates["sensitivity"],
        },
        "per_grade": per_grade,
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        aurocs = {}
        for g in labels:
            yt = (y_true == g).astype(int)
            if yt.min() == yt.max():  # only one class present
                aurocs[g] = None
            else:
                aurocs[g] = float(roc_auc_score(yt, y_prob[:, g]))
        out["auroc_ovr"] = aurocs
    return out


def referable_threshold_sweep(
    y_true,
    y_prob,
    thresholds=(0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
) -> list[dict]:
    """Sensitivity / specificity for the binary 'referable DR' decision
    (P(grade >= REFERABLE_DR_MIN_GRADE) >= threshold) at several thresholds.

    Use this to pick DR_REFERABLE_THRESHOLD: a screening tool wants high
    sensitivity, accepting lower specificity.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob)
    p_ref = y_prob[:, REFERABLE_DR_MIN_GRADE:].sum(axis=1)
    is_ref = y_true >= REFERABLE_DR_MIN_GRADE
    rows = []
    for t in thresholds:
        pred = p_ref >= t
        tp = int((pred & is_ref).sum())
        fn = int((~pred & is_ref).sum())
        fp = int((pred & ~is_ref).sum())
        tn = int((~pred & ~is_ref).sum())
        rows.append(
            {
                "threshold": float(t),
                "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            }
        )
    return rows


def format_threshold_sweep(rows: list[dict]) -> str:
    out = ["Referable-DR decision vs P(grade>=2) threshold:",
           f"{'thr':>6} {'sens':>8} {'spec':>8} {'missed':>8} {'false_ref':>10}"]
    for r in rows:
        out.append(
            f"{r['threshold']:>6.2f} {r['sensitivity']*100:>7.1f}% "
            f"{r['specificity']*100:>7.1f}% {r['fn']:>8d} {r['fp']:>10d}"
        )
    out.append("(set DR_REFERABLE_THRESHOLD to the lowest thr whose specificity is still acceptable)")
    return "\n".join(out)


def format_report(m: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("DR-Sight evaluation report")
    lines.append("=" * 60)
    ref = m["referable_dr"]
    lines.append(
        f"HEADLINE  Referable DR (grade >= {ref['min_grade']}) sensitivity: "
        f"{ref['sensitivity']*100:.1f}%   specificity: {ref['specificity']*100:.1f}%"
    )
    lines.append(f"          Quadratic weighted kappa: {m['quadratic_weighted_kappa']:.4f}")
    lines.append(f"          Overall accuracy (de-emphasised): {m['accuracy']*100:.1f}%")
    lines.append("-" * 60)
    lines.append(f"{'grade':>6} {'sens':>7} {'spec':>7} {'prec':>7} {'f1':>7} {'auroc':>7} {'n':>6}")
    for g, r in m["per_grade"].items():
        au = m.get("auroc_ovr", {}).get(g)
        au_s = f"{au:.3f}" if au is not None else "  n/a"
        lines.append(
            f"{g:>6} {r['sensitivity']:>7.3f} {r['specificity']:>7.3f} "
            f"{r['precision']:>7.3f} {r['f1']:>7.3f} {au_s:>7} {r['support']:>6}"
        )
    lines.append("-" * 60)
    lines.append("Confusion matrix (rows=true, cols=pred):")
    for i, row in enumerate(m["confusion_matrix"]):
        lines.append(f"  {i}: " + " ".join(f"{v:5d}" for v in row))
    lines.append("=" * 60)
    return "\n".join(lines)
