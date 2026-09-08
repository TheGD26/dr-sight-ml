# DR-Sight - MATLAB DR grading + image-quality gate + retinal segmentation (SIH PS26038)

MATLAB-side diabetic-retinopathy severity grading, running the trained DR-Sight
model (`models/best.pt`, MobileNetV3-Small, 5-class) via ONNX, plus an
Image Processing Toolbox port of the pre-classifier image-quality gate
("Image Quality Assessment and Enhancement" requirement) and two classical,
training-free retinal-structure segmenters (optic disc, vessels).

## Files

| File | Purpose |
|------|---------|
| `../scripts/export_onnx.py` | Export `models/best.pt` -> `models/dr_sight.onnx` (dynamic batch, input `input`, output `logits`). Prints the exact input shape + ImageNet mean/std MATLAB must match. |
| `../scripts/make_reference_csv.py` | Dump per-image PyTorch predictions to `matlab/reference_test.csv` for the validation step. |
| `grade_dr.m` | Grade one fundus image: returns `grade` (0-4), `label`, `confidence`, `probabilities`, `p_referable` (softmax mass on grades >= 2), `referable`, `referral_escalated`, `referral_action`, `uncertain`. Labels / referral actions / thresholds mirror `src/config.py`. |
| `validate_grading.m` | Run `grade_dr` over the reference CSV and check the ONNX-imported model agrees with PyTorch within tolerance. |
| `quality_gate.m` | Port of `src/quality/image_quality.py`. Blur (variance of Laplacian), exposure (mean brightness + black/white clip fractions) and field-of-view (largest bright blob after Otsu) checks with the same reject-reason strings; returns `usable`, `reason`, `scores`, and for borderline-but-usable images an `enhanced_image` (flat-field + CLAHE + light denoise). Thresholds mirror `src/config.py::QualityConfig`. |
| `test_quality_gate.m` | Runs `quality_gate` over APTOS cache / Grad-CAM sample images plus a few synthesised degraded variants, prints a verdict table, and saves before/after figures to `matlab/output/quality_samples/`. |
| `segment_optic_disc.m` | Locate the optic disc as the largest bright, roughly circular blob in the green/red channel (FOV mask -> vessel-suppressing close -> strong Gaussian blur -> top-percentile threshold -> `regionprops`). Returns `found`, `centroid`, `radius`, `circularity`, `area_fraction`, disc `mask`, `overlay` (ROI drawn) and a close-up `crop`. Brightness/geometry heuristic, no training data. |
| `segment_vessels.m` | Segment the retinal vasculature: FOV mask -> green channel + CLAHE + top-hat flat-field -> `fibermetric` multi-scale ridge filter (matched-filter-bank fallback) -> `imbinarize` (adaptive/Otsu) -> `bwskel` centreline. Returns the binary `mask`, `skeleton`, `vessel_density` (fraction of in-FOV pixels), `skeleton_length`, `mean_width_px` and an `overlay`. |
| `SegmentationDemo.m` | Section-by-section demo of both segmenters on a sample fundus image with before/after figures (save as `.mlx` for a Live Script). Ends with an explicit implementation-status block: microaneurysm / exudate / haemorrhage / neovascularisation detection are **not implemented** and documented as future work needing per-lesion annotated data (IDRiD) and dedicated CNN/U-Net models. |
| `DRScreeningDemo.m` | Section-by-section demo (save as `.mlx` for a Live Script). |

## Requirements

- MATLAB with **Deep Learning Toolbox** and the **Deep Learning Toolbox
  Converter for ONNX Model Format** (provides `importNetworkFromONNX`) for the
  grading path, and the **Image Processing Toolbox** for `quality_gate.m`
  (`imfilter`, `graythresh`/`imbinarize`, `bwareafilt`, `adapthisteq`,
  `rgb2lab`, `imgaussfilt`; uses `imflatfield`/`imbilatfilt` when present).
- **Image Processing Toolbox** + **Computer Vision Toolbox** for the
  segmenters: `fibermetric`, `bwskel`, `regionprops`, `adapthisteq`,
  `imtophat`, `imbinarize` (IPT) and `insertShape`/`insertMarker`/`insertText`
  (CVT, used only to draw the optic-disc overlay). `segment_vessels.m` falls
  back to a built-in matched-filter bank if `fibermetric` is missing.
- Python env from `requirements.txt` plus `onnx` and `onnxruntime`
  (`uv pip install onnx onnxruntime`) for the two export scripts.

## Usage

From the repo root (`dr-sight-ml/`):

```bash
python scripts/export_onnx.py
python scripts/make_reference_csv.py --split test      # writes matlab/reference_test.csv
```

Then in MATLAB:

```matlab
cd matlab
result = grade_dr("../data/aptos/cache/0024cdab0c1e.png")
validate_grading()                       % uses reference_test.csv next to this file

q = quality_gate("../data/aptos/cache/0024cdab0c1e.png")   % usable / reason / scores / enhanced_image
test_quality_gate                        % verdict table + before/after figures under output/quality_samples/

disc = segment_optic_disc("../data/aptos/cache/0024cdab0c1e.png", Show=true)  % centroid / radius / overlay
ves  = segment_vessels("../data/aptos/cache/0024cdab0c1e.png", Show=true)     % mask / skeleton / vessel_density
SegmentationDemo                          % full before/after walkthrough + implementation-status block
```

### Retinal segmentation scope (PS26038)

PS26038 lists six retinal-analysis tasks. This prototype implements **two**
properly - **optic disc / fovea localisation** (`segment_optic_disc.m`; the
fovea is ~2.5 disc-diameters temporal to the disc centre) and **vessel
segmentation** (`segment_vessels.m`). The other four - **microaneurysm
detection, exudate segmentation, haemorrhage classification, neovascularisation
detection** - are **not implemented** and are documented (in code, and printed
to the console by `SegmentationDemo.m`) as future work requiring per-lesion
annotated training data (e.g. IDRiD segmentation masks) and dedicated CNN/U-Net
models. They are deliberately left out rather than approximated so results are
never faked.

### Quality gate vs. the Python reference

`quality_gate.m` keeps the Python decision order (FOV fill, then exposure mean,
then black/white clip fraction, then blur) and the same operator-facing reason
strings. Scores are computed with Image Processing Toolbox equivalents
(`imfilter` + `fspecial('laplacian',0)` for the Laplacian, population variance to
match `numpy.var`; `imgaussfilt(sigma=1.4)` for the 7x7 `cv2.GaussianBlur`;
`graythresh`/`imbinarize` for Otsu), so absolute numbers track OpenCV closely but
are not bit-identical (border handling, resampling). Enhancement only runs on
images that *pass* the gate but look borderline - outright rejects are returned
unenhanced, exactly as the Python contract implies.

### Preprocessing (must match `src/data/dataset.py`)

1. RGB, `uint8` [0, 255]
2. Letterbox: resize longest side to 224 (bilinear, no antialiasing), centre-pad
   to 224x224 with 0. For the 512x512 square APTOS cache images this is a plain
   resize to 224.
3. `x = x/255; x = (x - mean) / std` with `mean = [0.485 0.456 0.406]`,
   `std = [0.229 0.224 0.225]`
4. `NCHW`, `float32`, batch axis 1

`dr_sight.onnx` outputs raw **logits**; `grade_dr.m` applies softmax.

## Notes

- `reference_test.csv` is a generated artifact (contains absolute image paths
  from the machine that ran the script). Regenerate it locally; pass
  `imageDir=...` to `validate_grading` to resolve images by `id_code` instead.
- `models/dr_sight.onnx` is gitignored like the rest of `models/` - regenerate
  with `export_onnx.py`.
- Prototype for Smart India Hackathon 2026, **not** a certified medical device.
