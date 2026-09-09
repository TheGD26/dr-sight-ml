"""Compare the MATLAB screening backend against the reference PyTorch pipeline.

This is the artefact to show judges that the MATLAB path (matlab/screen_image.m
run through the MATLAB Engine API for Python) *reproduces* the validated PyTorch
numbers - not merely that it runs. It screens a handful of sample fundus images
through BOTH backends directly (no HTTP) and prints a side-by-side table of
grade / confidence / P(referable) plus the maximum absolute per-class softmax
difference.

    python scripts/compare_backends.py --gradeable-only --n 10
    python scripts/compare_backends.py --data-dir data/aptos/cache --n 8
    python scripts/compare_backends.py --weights models/best.pt

Many APTOS cache thumbnails fail the quality gate, so pass --gradeable-only to
pre-screen with the PyTorch gate and compare the first --n images that actually
reach the classifier (otherwise you only prove the two quality gates agree).

Requires a local MATLAB with the Deep Learning Toolbox + "Deep Learning Toolbox
Converter for ONNX Model Format" support package, and the MATLAB Engine API for
Python (see requirements.txt). models/dr_sight.onnx must already exist
(python scripts/export_onnx.py).

PS SIH26038, Team Diasight - prototype, not a certified medical device.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _fmt(v, spec="8.4f"):
    return "   n/a  " if v is None else format(v, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/aptos/cache", help="folder of sample images")
    ap.add_argument("--n", type=int, default=6, help="how many images to compare")
    ap.add_argument("--weights", default=None, help="checkpoint for the PyTorch pipeline (default: PipelineConfig)")
    ap.add_argument("--allow-untrained", action="store_true",
                    help="let the PyTorch pipeline run with random weights if no checkpoint")
    ap.add_argument("--prob-tol", type=float, default=0.02,
                    help="max |Δ softmax| before a row is flagged")
    ap.add_argument("--gradeable-only", action="store_true",
                    help="pre-screen images with the PyTorch quality gate and compare "
                         "only the first --n that are gradeable (so the run actually "
                         "exercises grading, not just quality-gate agreement)")
    args = ap.parse_args()

    data_dir = (_REPO_ROOT / args.data_dir) if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    all_images = sorted(data_dir.glob("*.png"))
    if not all_images:
        print(f"No .png images under {data_dir}", file=sys.stderr)
        return 2

    # --- PyTorch reference pipeline ---------------------------------------- #
    from src.config import PipelineConfig
    from src.inference.pipeline import DRPipeline

    cfg = PipelineConfig()
    if args.weights:
        cfg.weights_path = args.weights
    py = DRPipeline(cfg=cfg, allow_untrained=args.allow_untrained)
    print(f"PyTorch backend  : backbone={py.backbone}  trained={py.trained}  "
          f"synthetic={getattr(py, 'synthetic_weights', False)}")

    if args.gradeable_only:
        images: list[Path] = []
        for img in all_images:
            if len(images) >= args.n:
                break
            if py.run(img.read_bytes(), with_heatmap=False).to_dict()["usable_image"]:
                images.append(img)
        if not images:
            print(f"No gradeable images under {data_dir} (all fail the quality gate).",
                  file=sys.stderr)
            return 2
        print(f"Comparing {len(images)} gradeable image(s) "
              f"(scanned {all_images.index(images[-1]) + 1} of {len(all_images)})\n")
    else:
        images = all_images[: args.n]
        print(f"Comparing {len(images)} image(s) from {data_dir}\n")

    # --- MATLAB backend --------------------------------------------------- #
    from src.inference.matlab_pipeline import MatlabDRPipeline

    ml = MatlabDRPipeline()
    h = ml.health()
    print(f"MATLAB backend   : engine_start={h['engine_start_seconds']}s  "
          f"onnx_import={h['net_import_seconds']}s  net_ready={h['network_ready']}\n")

    # --- run both ------------------------------------------------------- #
    header = (
        f"{'image':<20}  {'py_g':>4} {'ml_g':>4}  {'py_conf':>8} {'ml_conf':>8}  "
        f"{'py_pref':>8} {'ml_pref':>8}  {'maxΔprob':>9}  flags"
    )
    print(header)
    print("-" * len(header))

    worst = 0.0
    grade_mismatches = 0
    graded_both = 0
    for img in images:
        raw = img.read_bytes()
        rp = py.run(raw, with_heatmap=False).to_dict()
        rm = ml.run(raw, with_heatmap=False)

        flags = []
        max_dp = None
        if rp["probabilities"] and rm["probabilities"]:
            graded_both += 1
            max_dp = max(abs(a - b) for a, b in zip(rp["probabilities"], rm["probabilities"]))
            worst = max(worst, max_dp)
            if max_dp > args.prob_tol:
                flags.append("PROB")
        if rp["usable_image"] != rm["usable_image"]:
            flags.append("USABLE")
        if rp["grade"] != rm["grade"]:
            flags.append("GRADE")
            grade_mismatches += 1

        print(
            f"{img.name:<20}  {str(rp['grade']):>4} {str(rm['grade']):>4}  "
            f"{_fmt(rp['confidence'])} {_fmt(rm['confidence'])}  "
            f"{_fmt(rp['p_referable'])} {_fmt(rm['p_referable'])}  "
            f"{_fmt(max_dp, '9.5f')}  {' '.join(flags)}"
        )

    print("-" * len(header))
    print(f"\ngraded on both backends: {graded_both}/{len(images)}  "
          f"({len(images) - graded_both} rejected by a quality gate)")
    print(f"max |Δ softmax probability| across graded images: {worst:.6f}  (tol {args.prob_tol})")
    print(f"grade mismatches: {grade_mismatches}/{graded_both}")

    if graded_both == 0:
        print("\nRESULT: INCONCLUSIVE - every sample image was rejected by a quality "
              "gate, so no grading was compared. Point --data-dir at gradeable "
              "fundus images (or raise --n).")
        ml.close()
        return 2

    ok = worst <= args.prob_tol and grade_mismatches == 0
    print("\nRESULT:", "PASS - MATLAB reproduces the PyTorch pipeline" if ok
          else "REVIEW - differences exceed tolerance (see flagged rows)")
    ml.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
