# gradcam_samples — Grad-CAM eyeball fixtures

Small set of real APTOS fundus images, one per DR grade (0–4), used to
sanity-check that Grad-CAM heatmaps land on retinal structure (see the repo
`README.md` "Grad-CAM sanity check" section) and as generic sample inputs for
`matlab/test_quality_gate.m` and `scripts/measure_matlab_latency.py`.

## Labels are derived from `matlab/reference_test.csv`, not hand-assigned

Each file is a verbatim copy of a `data/aptos/cache/<id_code>.png` row from
`matlab/reference_test.csv`. The grade in the filename is that row's recorded
grade — it is **not** typed by hand and it is **not** re-derived from a model
run here. For every sample below the APTOS ground-truth grade (`true_grade`) and
the current shipped model's prediction (`pt_grade`, from `models/dr_sight.onnx`)
agree, so the filename is unambiguous.

**If you ever see a sample whose model output disagrees with its filename,
trust `matlab/reference_test.csv` and `scripts/compare_backends.py` (backend vs
backend / vs the CSV) — never the filename.** The previous fixtures here
(`val_NN_trueX_predY.png`) had hand-written prediction labels that went stale
against the shipped model and triggered a false "model is broken" alarm during a
verification pass. Filenames are a convenience, not a source of truth.

## Provenance

| file | source id_code (APTOS) | true_grade | pt_grade | pt_label | pt_confidence | pt_p_referable |
|------|------------------------|------------|----------|----------|---------------|----------------|
| `sample_grade0_ref.png` | `060e00d1e2ab` | 0 | 0 | No DR | 1.000 | 0.000 |
| `sample_grade1_ref.png` | `8f2996b8d855` | 1 | 1 | Mild NPDR | 0.951 | 0.026 |
| `sample_grade2_ref.png` | `7743f4e04a6d` | 2 | 2 | Moderate NPDR | 0.967 | 0.995 |
| `sample_grade3_ref.png` | `537e50fdf22e` | 3 | 3 | Severe NPDR | 0.990 | 1.000 |
| `sample_grade4_ref.png` | `247e98aba610` | 4 | 4 | Proliferative DR | 0.990 | 1.000 |

## Regenerating

The `*.png` here are gitignored (`.gitignore`: `tests/gradcam_samples/*.png`),
same policy as `/data/` — APTOS images are license-restricted and not committed.
To rebuild from a populated `data/aptos/cache/`:

```bash
# from dr-sight-ml/
python - <<'PY'
import csv, shutil, pathlib
want = {"060e00d1e2ab":0, "8f2996b8d855":1, "7743f4e04a6d":2,
        "537e50fdf22e":3, "247e98aba610":4}
rows = {r["id_code"]: r for r in csv.DictReader(open("matlab/reference_test.csv"))}
out = pathlib.Path("tests/gradcam_samples")
for cid, g in want.items():
    shutil.copy(f"data/aptos/cache/{cid}.png", out / f"sample_grade{g}_ref.png")
PY
```

Pick any other row from `matlab/reference_test.csv` where `true_grade ==
pt_grade` if you want different exemplars — keep the filename grade sourced from
the CSV column, not typed by hand.
