"""Measure the steady-state per-image latency of the MATLAB screening call.

SIH PS26038, Team Diasight.

``matlab/simulink/run_workflow_sim.m`` models the district screening pipeline as
a fluid queue and needs a real number for ``ai_time_s`` - the "quality gate +
AI inference per image" service time. Until now that was a 2.0 s placeholder.

This script measures it directly: it builds the real
``src.inference.matlab_pipeline.MatlabDRPipeline`` (one long-lived
``matlab.engine`` session with the ONNX network imported once), throws away the
one-time engine-startup / network-import cost **and** a couple of warm-up calls,
then times ``pipeline.run(image_bytes)`` - i.e. one ``screen_image.m`` call
*through the engine* - once per sample image, looping over the images until it
has at least ``--runs`` timed calls.

It reports the mean and the min/max range of the steady-state per-image call.
``--no-heatmap`` measures the faster Grad-CAM-free path for comparison; the
default (heatmap on) is what the live FastAPI request path
(``src/api/main.py::_screen``, ``with_heatmap=True``) actually pays per image.

Usage (from the repo root, with the MATLAB Engine API for Python installed and a
licensed local MATLAB on PATH)::

    python scripts/measure_matlab_latency.py
    python scripts/measure_matlab_latency.py --runs 10 --no-heatmap
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Sample fundus images to rotate through. Real retina images that PASS the
# quality gate, so each timed call exercises the full quality-gate + grade +
# Grad-CAM path. A gate-rejected image returns in ~50 ms (no model forward) and
# would deflate the measured per-image number, so those are filtered out below -
# ``ai_time_s`` in the Simulink model is the cost of actually screening an image.
_SAMPLE_DIRS = (
    _REPO_ROOT / "tests" / "gradcam_samples",
    _REPO_ROOT / "data" / "aptos" / "cache",
)


def _candidate_paths(limit: int) -> list[Path]:
    out: list[Path] = []
    for d in _SAMPLE_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.png")):
            if p.stat().st_size > 0:
                out.append(p)
            if len(out) >= limit:
                return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=8,
                    help="number of timed steady-state calls (default 8, min 5 enforced)")
    ap.add_argument("--warmup", type=int, default=2,
                    help="untimed warm-up calls after engine start (default 2)")
    ap.add_argument("--no-heatmap", action="store_true",
                    help="measure the Grad-CAM-free path instead of the default "
                         "with_heatmap=True request path")
    args = ap.parse_args()

    runs = max(5, args.runs)
    with_heatmap = not args.no_heatmap

    from src.inference.matlab_pipeline import (
        MatlabDRPipeline,
        MatlabEngineUnavailable,
    )

    candidates = _candidate_paths(limit=2000)
    if len(candidates) < 2:
        print("Need at least 2 candidate PNGs under tests/gradcam_samples or "
              f"data/aptos/cache; found {len(candidates)}.", file=sys.stderr)
        return 2

    print(f"Building MatlabDRPipeline (cold start - NOT timed) ...", flush=True)
    t0 = time.perf_counter()
    try:
        pipe = MatlabDRPipeline()
    except MatlabEngineUnavailable as e:
        print(f"MATLAB backend unavailable:\n{e}", file=sys.stderr)
        return 2
    cold = time.perf_counter() - t0
    h = pipe.health()
    print(f"  cold start total      : {cold:6.2f} s")
    print(f"    engine start        : {h['engine_start_seconds']} s")
    print(f"    ONNX network import  : {h['net_import_seconds']} s")
    print(f"  (both excluded from the per-image number below)\n")

    times: list[float] = []
    used: list[str] = []
    try:
        # Find quality-gate-passing images and time them. The first `warmup`
        # usable calls are discarded (first call through the engine pays JIT /
        # lazy-init costs that are not representative of steady state).
        need = runs + args.warmup
        warmed = 0
        for path in candidates:
            if len(times) >= runs:
                break
            img = path.read_bytes()
            t = time.perf_counter()
            res = pipe.run(img, with_heatmap=with_heatmap)
            dt = time.perf_counter() - t
            if not res.get("usable_image"):
                continue  # gate rejection - not a representative screening call
            if warmed < args.warmup:
                warmed += 1
                print(f"  warm-up {warmed}/{args.warmup} (discarded): "
                      f"{dt:.3f} s   ({path.name})", flush=True)
                continue
            times.append(dt)
            used.append(path.name)
            print(f"  call {len(times):2d}/{runs}  {dt:6.3f} s   ({path.name})",
                  flush=True)
        if len(times) < runs:
            print(f"\nOnly {len(times)} usable images found (needed {runs}). "
                  "Add more passing fundus samples and re-run.", file=sys.stderr)
            if len(times) < 5:
                return 3
    finally:
        pipe.close()

    mean = statistics.mean(times)
    lo, hi = min(times), max(times)
    stdev = statistics.pstdev(times)
    print("\n" + "=" * 60)
    print(f"steady-state per-image screen_image() latency through the engine")
    print(f"  with_heatmap : {with_heatmap}")
    print(f"  samples      : {len(set(used))} distinct usable images, "
          f"{len(times)} timed calls")
    print(f"  mean         : {mean:.3f} s")
    print(f"  range        : {lo:.3f} s  -  {hi:.3f} s")
    print(f"  std dev      : {stdev:.3f} s")
    print("=" * 60)
    print(f"\n-> set  ai_time_s = {mean:.2f};  in matlab/simulink/run_workflow_sim.m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
