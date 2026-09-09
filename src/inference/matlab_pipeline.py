"""MATLAB screening backend for DR-Sight.

PS SIH26038 requires the screening computation (image-quality assessment, DR
grading, explainability) to run in **MATLAB**. This module is the Python <->
MATLAB bridge: it owns ONE long-lived ``matlab.engine`` session, imports the
ONNX network ONCE, and exposes a ``run(image_bytes)`` method whose return dict
is a drop-in replacement for ``src.inference.pipeline.PredictionResult.to_dict``.

FastAPI (``src/api/main.py``) becomes a thin HTTP/auth/CORS layer that calls
``get_matlab_pipeline().run(...)`` - no PyTorch in the live request path.

Why ``eng.eval`` instead of ``eng.screen_image(path, net, opts, nargout=1)``:
``matlab.engine`` cannot marshal a MATLAB struct **array** back to Python
("only a scalar struct can be returned from MATLAB"), and ``screen_image``'s
``affected_regions`` field is exactly that. It also cannot round-trip a
``dlnetwork`` handle. So the network stays parked in the MATLAB workspace and
``screen_image`` is invoked MATLAB-side with its result ``jsonencode``-d to a
char array - the only value that actually crosses the boundary. ``_matlab_to_py``
/ ``_normalize_result`` below then coerce that into the exact PredictionResult
shape (MATLAB ``[]`` / ``""`` -> Python ``None`` for the nullable fields).

This is a Smart India Hackathon 2026 prototype, NOT a certified medical device.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("dr_sight.matlab")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MATLAB_DIR = _REPO_ROOT / "matlab"
_DEFAULT_ONNX = _REPO_ROOT / "models" / "dr_sight.onnx"

# Canonical PredictionResult key order (src/inference/pipeline.py).
_RESULT_KEYS = (
    "usable_image",
    "rejection_reason",
    "grade",
    "label",
    "confidence",
    "uncertain",
    "referable",
    "p_referable",
    "referral_escalated",
    "referral_action",
    "heatmap_base64",
    "affected_regions",
    "probabilities",
    "quality_scores",
    "disclaimer",
)
# Fields that are ``X | None`` in the dataclass - MATLAB empties map to None.
_NULLABLE_KEYS = (
    "rejection_reason",
    "grade",
    "label",
    "confidence",
    "referable",
    "p_referable",
    "referral_action",
    "heatmap_base64",
    "affected_regions",
    "probabilities",
    "quality_scores",
)
_BOOL_KEYS = ("usable_image", "uncertain", "referable", "referral_escalated")


class MatlabEngineUnavailable(RuntimeError):
    """Raised when the MATLAB Engine API is missing, unlicensed, or misconfigured.

    The message always tells the operator what to check - we never want the API
    to hang or fail with an opaque stack trace when MATLAB simply is not there.
    """


# --------------------------------------------------------------------------- #
# MATLAB <-> Python value conversion
# --------------------------------------------------------------------------- #
def _matlab_to_py(obj: Any) -> Any:
    """Recursively coerce anything that came back from MATLAB into plain Python.

    Handles the JSON-decoded structure (dict / list / scalars) and, defensively,
    raw ``matlab.*`` array types in case a caller switches to direct marshalling:
      * MATLAB struct        -> dict
      * MATLAB struct array   -> list of dicts
      * MATLAB double/logical -> float / bool (or nested lists for arrays)
      * MATLAB char           -> str
    """
    # Raw matlab.engine array types (matlab.double, matlab.logical, ...).
    try:  # pragma: no cover - only hit when direct marshalling is used
        import matlab  # type: ignore

        if isinstance(obj, (matlab.double, matlab.single, matlab.int8, matlab.int16,
                            matlab.int32, matlab.int64, matlab.uint8, matlab.uint16,
                            matlab.uint32, matlab.uint64)):
            flat = _flatten_matlab(obj)
            return [float(v) for v in flat]
        if isinstance(obj, matlab.logical):
            flat = _flatten_matlab(obj)
            return [bool(v) for v in flat] if len(flat) != 1 else bool(flat[0])
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: _matlab_to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_matlab_to_py(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", "replace")
    return obj


def _flatten_matlab(arr: Any) -> list:
    """Flatten a (possibly nested) matlab array to a Python list."""
    out: list = []
    try:
        for item in arr:
            if hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):
                out.extend(_flatten_matlab(item))
            else:
                out.append(item)
    except TypeError:
        out.append(arr)
    return out


def _is_empty(v: Any) -> bool:
    return v is None or v == [] or v == "" or v == {} or v == [[]]


def _normalize_result(d: dict, *, with_heatmap: bool) -> dict:
    """Coerce a jsonencode(screen_image(...)) dict into the exact PredictionResult
    shape: MATLAB empties -> None, numbers/bools to native types, canonical keys."""
    d = _matlab_to_py(d)

    for k in _NULLABLE_KEYS:
        if k in d and _is_empty(d[k]):
            d[k] = None

    if d.get("grade") is not None:
        d["grade"] = int(round(float(d["grade"])))
    for k in _BOOL_KEYS:
        if d.get(k) is not None:
            d[k] = bool(d[k])
    for k in ("confidence", "p_referable"):
        if d.get(k) is not None:
            d[k] = float(d[k])
    if d.get("probabilities") is not None:
        probs = d["probabilities"]
        if not isinstance(probs, (list, tuple)):
            probs = [probs]
        d["probabilities"] = [float(x) for x in probs]
    if d.get("affected_regions") is not None:
        regions = d["affected_regions"]
        if isinstance(regions, dict):  # single region -> wrap
            regions = [regions]
        d["affected_regions"] = [
            {
                "label": str(r.get("label", "")),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "radius": float(r["radius"]),
                "intensity": float(r["intensity"]),
            }
            for r in regions
        ]
    if d.get("quality_scores") is not None and isinstance(d["quality_scores"], dict):
        d["quality_scores"] = {k: float(v) for k, v in d["quality_scores"].items()}

    # with_heatmap=False must look exactly like DRPipeline.run(with_heatmap=False)
    if not with_heatmap:
        d["heatmap_base64"] = None
        d["affected_regions"] = None

    return {k: d.get(k) for k in _RESULT_KEYS}


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #
class MatlabDRPipeline:
    """Owns a single MATLAB engine session + the imported ONNX network.

    Construction is expensive (engine start ~5-15 s, ONNX import ~2-5 s) so this
    is built once per process via :func:`get_matlab_pipeline`.
    """

    def __init__(self, onnx_path: str | os.PathLike | None = None):
        try:
            import matlab.engine  # noqa: F401
        except Exception as e:  # ImportError, or a broken install
            raise MatlabEngineUnavailable(
                "Cannot import 'matlab.engine'. Install the MATLAB Engine API for "
                "Python for your local MATLAB:\n"
                "    cd <matlabroot>/extern/engines/python\n"
                "    python -m pip install .\n"
                "(see the comment block in requirements.txt). Original error: "
                f"{e!r}"
            ) from e

        self._matlab_engine = matlab.engine
        self.onnx_path = str(Path(onnx_path) if onnx_path else _DEFAULT_ONNX)

        # thresholds from the single source of truth (no torch import here)
        from src.config import DISCLAIMER, PipelineConfig

        _cfg = PipelineConfig()
        self.referable_threshold = float(_cfg.referable_threshold)
        self.uncertainty_threshold = float(_cfg.uncertainty_threshold)
        self.disclaimer = DISCLAIMER

        self.engine_start_seconds: float | None = None
        self.net_import_seconds: float | None = None
        self.network_ready = False
        # MATLAB engine is a single-threaded session; serialise access to it.
        self._call_lock = threading.Lock()

        try:
            self._start_engine()
            self._import_network()
        except Exception:
            # Don't leak a running MATLAB session if ONNX import fails.
            self.close()
            raise

    # ------------------------------------------------------------------ #
    def _start_engine(self) -> None:
        t0 = time.perf_counter()
        try:
            self._eng = self._matlab_engine.start_matlab()
        except Exception as e:
            raise MatlabEngineUnavailable(
                "matlab.engine.start_matlab() failed. Check that a local MATLAB "
                "(R2022b or newer) is installed and licensed, and that running\n"
                "    matlab -batch \"disp(matlabroot)\"\n"
                f"works from a shell. Original error: {e!r}"
            ) from e
        self.engine_start_seconds = time.perf_counter() - t0
        self._eng.addpath(str(_MATLAB_DIR), nargout=0)
        _LOG.info(
            "MATLAB engine started in %.1fs (matlab dir on path: %s)",
            self.engine_start_seconds,
            _MATLAB_DIR,
        )

    # ------------------------------------------------------------------ #
    def _import_network(self) -> None:
        """Import models/dr_sight.onnx ONCE and park the handle in the MATLAB
        workspace as ``dr_sight_net`` (a dlnetwork can't be marshalled to Python,
        so it never leaves MATLAB)."""
        if not Path(self.onnx_path).is_file():
            raise MatlabEngineUnavailable(
                f"ONNX model not found at {self.onnx_path}. Export it once from "
                "the repo root:\n    python scripts/export_onnx.py"
            )
        t0 = time.perf_counter()
        try:
            self._eng.workspace["dr_sight_onnx_path"] = self.onnx_path
            self._eng.eval(
                "dr_sight_net = load_screening_net(dr_sight_onnx_path);", nargout=0
            )
        except Exception as e:
            raise MatlabEngineUnavailable(
                "Importing the ONNX network in MATLAB failed. Ensure the 'Deep "
                "Learning Toolbox Converter for ONNX Model Format' support package "
                "is installed (Add-On Explorer) and that models/dr_sight.onnx is "
                f"valid. Original error: {e!r}"
            ) from e
        self.net_import_seconds = time.perf_counter() - t0
        self.network_ready = True
        _LOG.info(
            "Imported %s into MATLAB in %.1fs",
            Path(self.onnx_path).name,
            self.net_import_seconds,
        )

    # ------------------------------------------------------------------ #
    def run(self, image_bytes: bytes, with_heatmap: bool = True) -> dict:
        """Screen one image end-to-end in MATLAB. Returns a PredictionResult-shaped
        dict (same keys as ``PredictionResult.to_dict()``)."""
        if not self.network_ready:
            raise RuntimeError("MATLAB screening network is not loaded.")

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            tmp.write(image_bytes)
            tmp.flush()
            tmp.close()

            with self._call_lock:
                self._eng.workspace["dr_img_path"] = tmp.name
                self._eng.workspace["dr_with_heatmap"] = bool(with_heatmap)
                self._eng.workspace["dr_ref_thr"] = float(self.referable_threshold)
                self._eng.workspace["dr_unc_thr"] = float(self.uncertainty_threshold)
                try:
                    self._eng.eval(
                        "dr_result_json = jsonencode(screen_image("
                        "dr_img_path, dr_sight_net, "
                        "'WithHeatmap', logical(dr_with_heatmap), "
                        "'ReferableThreshold', dr_ref_thr, "
                        "'UncertaintyThreshold', dr_unc_thr));",
                        nargout=0,
                    )
                    raw_json = self._eng.workspace["dr_result_json"]
                except Exception as e:  # MatlabExecutionError etc.
                    raise RuntimeError(
                        f"MATLAB screen_image() failed: {e}"
                    ) from e
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        try:
            parsed = json.loads(raw_json)
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                f"Could not parse screen_image() JSON from MATLAB: {e}"
            ) from e

        return _normalize_result(parsed, with_heatmap=with_heatmap)

    # ------------------------------------------------------------------ #
    def health(self) -> dict:
        return {
            "engine_started": hasattr(self, "_eng"),
            "engine_start_seconds": (
                round(self.engine_start_seconds, 2)
                if self.engine_start_seconds is not None
                else None
            ),
            "network_ready": self.network_ready,
            "net_import_seconds": (
                round(self.net_import_seconds, 2)
                if self.net_import_seconds is not None
                else None
            ),
            "onnx_path": self.onnx_path,
        }

    def close(self) -> None:
        eng = getattr(self, "_eng", None)
        if eng is not None:
            try:
                eng.quit()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# module-level lazy singleton (mirrors src.inference.pipeline.get_pipeline)
# --------------------------------------------------------------------------- #
_PIPELINE: MatlabDRPipeline | None = None
_PIPELINE_LOCK = threading.Lock()


def get_matlab_pipeline(**kw) -> MatlabDRPipeline:
    """Return the process-wide MATLAB pipeline, constructing it on first use.

    Signature mirrors ``src.inference.pipeline.get_pipeline`` so ``src/api/main.py``
    barely changes. Accepts ``onnx_path=...``; other kwargs are ignored for
    parity with ``get_pipeline`` (which takes ``allow_untrained``).
    """
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = MatlabDRPipeline(onnx_path=kw.get("onnx_path"))
    return _PIPELINE


if __name__ == "__main__":  # tiny CLI smoke test
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Screen one image via the MATLAB backend.")
    ap.add_argument("image")
    ap.add_argument("--no-heatmap", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        pipe = get_matlab_pipeline()
    except MatlabEngineUnavailable as e:
        print(f"MATLAB backend unavailable:\n{e}", file=sys.stderr)
        sys.exit(2)

    with open(args.image, "rb") as fh:
        raw = fh.read()
    res = pipe.run(raw, with_heatmap=not args.no_heatmap)
    if res.get("heatmap_base64"):
        res["heatmap_base64"] = res["heatmap_base64"][:32] + f"... ({len(res['heatmap_base64'])} chars)"
    print(json.dumps(res, indent=2))
