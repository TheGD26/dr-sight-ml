function result = screen_image(imageInput, net, opts)
%SCREEN_IMAGE  End-to-end DR screening for one fundus image (SIH PS26038, Team Diasight).
%
%   result = SCREEN_IMAGE(imagePath)
%   result = SCREEN_IMAGE(imageMatrix)
%   result = SCREEN_IMAGE(..., net)
%   result = SCREEN_IMAGE(..., net, Name=Value)
%
%   The MATLAB orchestrator that chains quality_gate.m -> grade_dr.m ->
%   explain_gradcam.m into a single call, reproducing the five stages of
%   dr-sight-ml/src/inference/pipeline.py::DRPipeline.run() so that the FastAPI
%   backend can run the ENTIRE screening computation inside MATLAB and keep
%   Python as a thin HTTP/auth/CORS layer only.
%
%   INPUTS
%     imageInput : path to an image file, OR an HxWx3 uint8 RGB array (so this
%                  works both from a temp file written by Python and from an
%                  in-memory frame).
%     net        : (optional) a pre-imported dlnetwork from
%                  importNetworkFromONNX(...). Pass this in from a long-lived
%                  caller (see load_screening_net.m / matlab_pipeline.py) so the
%                  ONNX import - which costs several seconds - happens ONCE per
%                  process, not once per request. If omitted or [], the model
%                  next to this file (../models/dr_sight.onnx) is imported and
%                  cached via load_screening_net().
%
%   NAME-VALUE OPTIONS
%     WithHeatmap          (logical) run explain_gradcam.m for the Grad-CAM
%                          overlay + focus regions (default true). Set false for
%                          a faster health-check / batch-grading path - then
%                          heatmap_base64 and affected_regions come back empty,
%                          exactly like DRPipeline.run(with_heatmap=False).
%     FeatureLayer         (string) Grad-CAM target layer override, forwarded to
%                          explain_gradcam.m ("" = auto-select).
%     ReferableThreshold   (double) P(grade >= 2) at/above which `referable` is
%                          true. Default 0.5 - the value of
%                          PipelineConfig.referable_threshold in src/config.py
%                          (env DR_REFERABLE_THRESHOLD, default "0.5"). MATLAB
%                          cannot import the Python config module, so the number
%                          is duplicated here; keep it in sync with
%                          src/config.py if that default changes.
%     UncertaintyThreshold (double) top-1 softmax below which `uncertain` is
%                          true (mandatory human review). Default 0.6 -
%                          PipelineConfig.uncertainty_threshold in src/config.py
%                          (env DR_UNCERTAINTY_THRESHOLD, default "0.6").
%     BlurMinVariance, ExposureMinMean, ExposureMaxMean, FovMinFill,
%     FovMaxFill, ClipFracMax  (double) override the matching quality_gate.m /
%                          QualityConfig thresholds. Leave as NaN to keep the
%                          quality_gate.m defaults.
%
%   OUTPUT  result : struct whose fields match, key-for-key and type-for-type,
%   PredictionResult.to_dict() in src/inference/pipeline.py so the Python bridge
%   can hand it to the frontend unchanged:
%
%     usable_image       logical
%     rejection_reason   char   ("" when usable)
%     grade              double 0-4        ([] when the image was rejected)
%     label              char              ("" when rejected)
%     confidence         double            ([] when rejected)   softmax of `grade`, round 4
%     uncertain          logical           (false when rejected)
%     referable          logical           ([] when rejected)
%     p_referable        double            ([] when rejected)   round 4
%     referral_escalated logical           (false when rejected)
%     referral_action    char              ("" when rejected)
%     heatmap_base64     char   base64 PNG of the Grad-CAM overlay
%                               ("" when rejected or WithHeatmap=false)
%     affected_regions   struct array (label,x,y,radius,intensity) - identical
%                               shape to pipeline.py::cam_regions(); 0x0 when
%                               rejected or WithHeatmap=false
%     probabilities      1x5 double, softmax over grades 0..4, each round 4
%                               ([] when rejected)
%     quality_scores     struct - the `scores` struct from quality_gate.m
%                               (blur_var, mean_brightness, dark_clip_frac,
%                               bright_clip_frac, fov_fraction)
%     disclaimer         char - the standing screening-aid disclaimer, copied
%                               verbatim from src/config.py::DISCLAIMER
%
%   The Python bridge (src/inference/matlab_pipeline.py) converts [] -> None and
%   "" -> None for the nullable fields so the JSON is a byte-for-byte drop-in
%   for the PyTorch pipeline's response.
%
%   Smart India Hackathon 2026 prototype, NOT a certified medical device.
%   Screening aid only - every result must be confirmed by an ophthalmologist.

arguments
    imageInput
    net = []
    opts.WithHeatmap (1,1) logical = true
    opts.FeatureLayer (1,1) string = ""
    opts.ReferableThreshold (1,1) double = 0.5   % src/config.py PipelineConfig.referable_threshold
    opts.UncertaintyThreshold (1,1) double = 0.6 % src/config.py PipelineConfig.uncertainty_threshold
    opts.BlurMinVariance (1,1) double = NaN
    opts.ExposureMinMean (1,1) double = NaN
    opts.ExposureMaxMean (1,1) double = NaN
    opts.FovMinFill (1,1) double = NaN
    opts.FovMaxFill (1,1) double = NaN
    opts.ClipFracMax (1,1) double = NaN
end

here = fileparts(mfilename("fullpath"));
addpath(here);   % so quality_gate / grade_dr / explain_gradcam resolve when
                 % screen_image is called with a different working directory
                 % (e.g. from the MATLAB Engine started by Python).

% --- constants duplicated from src/config.py (MATLAB can't import the module) - %
REFERABLE_DR_MIN_GRADE = 2;                        % src/config.py REFERABLE_DR_MIN_GRADE
GRADE_LABELS = ["No DR", "Mild NPDR", "Moderate NPDR", ...
                "Severe NPDR", "Proliferative DR"];
REFERRAL_ACTIONS = [ ...                            % src/config.py REFERRAL_ACTIONS
    "No referral. Routine annual screening."; ...
    "No urgent referral. Re-screen in 9-12 months."; ...
    "Refer to ophthalmologist within ~6 months."; ...
    "Refer within ~1 month - treat as priority."; ...
    "Urgent referral within days."];
% src/config.py DISCLAIMER (two string literals concatenated in the source):
DISCLAIMER = "AI screening aid only - not a diagnosis. Confirm with an ophthalmologist.";

% --- read / normalise input to uint8 RGB -------------------------------- %
if ischar(imageInput) || isstring(imageInput)
    rgb = imread(char(imageInput));
else
    rgb = imageInput;
end
rgb = toRGBUint8(rgb);

% ===================================================================== %
% [1] image-quality gate  (mirrors DRPipeline.run step 1)
% ===================================================================== %
qcfg = buildQualityCfg(opts);
q = quality_gate(rgb, qcfg);

if ~q.usable
    % Same shape as PredictionResult(usable_image=False, rejection_reason=...,
    % quality_scores=...) - every grading/heatmap field empty/false.
    result = emptyResult(DISCLAIMER);
    result.usable_image     = false;
    result.rejection_reason = char(q.reason);
    result.quality_scores   = q.scores;
    result = orderfields(result, resultFieldOrder());
    return
end

% ===================================================================== %
% [2] choose the image the model sees.
%     NOTE: DRPipeline.run() grades the ORIGINAL image - it does
%     `rgb = np.asarray(pil)` straight after the gate and never looks at an
%     enhanced copy (the Python quality gate, src/quality/image_quality.py,
%     has no enhancement stage at all; only quality_gate.m produces
%     `enhanced_image`). To stay numerically faithful to the validated
%     89.7% / 93.8% pipeline we grade the ORIGINAL here too, and deliberately
%     ignore q.enhanced_image. If the Python pipeline is ever changed to grade
%     the enhanced image, change the next line to:
%         if q.enhanced; gradingImage = q.enhanced_image; else; gradingImage = rgb; end
% ===================================================================== %
gradingImage = rgb;

% ===================================================================== %
% [3] DR grading  (mirrors DRPipeline.run step 3: model forward -> softmax)
% ===================================================================== %
% Resolve the network only now - a rejected image never needs the model, so
% standalone callers that pass no `net` don't pay the ONNX import on rejects.
if isempty(net)
    net = load_screening_net();
end
g = grade_dr(gradingImage, net);
probs = double(g.probabilities(:)).';               % 1x5 row
[~, argmaxIdx] = max(probs);
grade = argmaxIdx - 1;                              % 0-based, model point estimate
confidence = probs(argmaxIdx);

% ===================================================================== %
% [4] Grad-CAM overlay + focus regions  (DRPipeline.run step 4)
% ===================================================================== %
heatmapB64 = "";
regions = emptyRegions();
if opts.WithHeatmap
    rep = explain_gradcam(gradingImage, net, g, ...
        FeatureLayer = opts.FeatureLayer, ...
        ShowFigure   = false);
    heatmapB64 = pngBase64(rep.overlay);            % base64 PNG string, no data: prefix
    regions    = rep.regions;                       % label,x,y,radius,intensity
end

% ===================================================================== %
% [5] referral decision + uncertainty  (mirrors referable_decision() exactly)
%     p_referable = softmax mass on grades >= min_grade
%     referable   = p_referable >= threshold
%     escalated   = referable AND argmax grade < min_grade
%     the reported `grade` stays the argmax; only the referral action escalates
% ===================================================================== %
pReferable = sum(probs((REFERABLE_DR_MIN_GRADE + 1):end));
referable  = pReferable >= opts.ReferableThreshold;
escalated  = referable && grade < REFERABLE_DR_MIN_GRADE;
uncertain  = confidence < opts.UncertaintyThreshold;

actionGrade = grade;
if escalated
    actionGrade = REFERABLE_DR_MIN_GRADE;
end

% --- assemble the PredictionResult-shaped struct ----------------------- %
result = emptyResult(DISCLAIMER);
result.usable_image       = true;
result.rejection_reason   = "";                     % None on the Python side
result.grade              = double(grade);
result.label              = char(GRADE_LABELS(grade + 1));
result.confidence         = round(confidence, 4);   % pipeline.py rounds to 4
result.uncertain          = logical(uncertain);
result.referable          = logical(referable);
result.p_referable        = round(pReferable, 4);
result.referral_escalated = logical(escalated);
result.referral_action    = char(REFERRAL_ACTIONS(actionGrade + 1));
result.heatmap_base64     = char(heatmapB64);
result.affected_regions   = regions;
result.probabilities      = round(probs, 4);        % 1x5 double, each rounded to 4
result.quality_scores     = q.scores;
result.disclaimer         = char(DISCLAIMER);

result = orderfields(result, resultFieldOrder());
end


% ======================================================================= %
function order = resultFieldOrder()
%RESULTFIELDORDER  Field order of PredictionResult in src/inference/pipeline.py.
order = {'usable_image', 'rejection_reason', 'grade', 'label', 'confidence', ...
    'uncertain', 'referable', 'p_referable', 'referral_escalated', ...
    'referral_action', 'heatmap_base64', 'affected_regions', 'probabilities', ...
    'quality_scores', 'disclaimer'};
end


% ======================================================================= %
function r = emptyResult(disclaimer)
%EMPTYRESULT  A struct with every PredictionResult field at its "rejected"
%   default: [] for nullable numerics, "" for nullable strings, false for the
%   two boolean flags that default False in the dataclass.
r = struct( ...
    "usable_image",       false, ...
    "rejection_reason",   "", ...
    "grade",              [], ...
    "label",              "", ...
    "confidence",         [], ...
    "uncertain",          false, ...
    "referable",          [], ...
    "p_referable",        [], ...
    "referral_escalated", false, ...
    "referral_action",    "", ...
    "heatmap_base64",      "", ...
    "affected_regions",   emptyRegions(), ...
    "probabilities",      [], ...
    "quality_scores",     struct(), ...
    "disclaimer",         char(disclaimer));
% struct() would broadcast a struct-array value; emptyRegions() is 0x0 so it
% is stored as-is, but guard anyway.
r.affected_regions = emptyRegions();
end


% ======================================================================= %
function s = emptyRegions()
%EMPTYREGIONS  0x0 struct array with cam_regions()'s fields.
s = struct("label", {}, "x", {}, "y", {}, "radius", {}, "intensity", {});
end


% ======================================================================= %
function cfg = buildQualityCfg(opts)
%BUILDQUALITYCFG  Turn the non-NaN quality overrides into a quality_gate.m cfg
%   struct. Missing fields fall back to quality_gate.m's QualityConfig defaults.
cfg = struct();
map = struct( ...
    "BlurMinVariance",  "blur_min_variance", ...
    "ExposureMinMean",  "exposure_min_mean", ...
    "ExposureMaxMean",  "exposure_max_mean", ...
    "FovMinFill",       "fov_min_fill", ...
    "FovMaxFill",       "fov_max_fill", ...
    "ClipFracMax",      "clip_frac_max");
src = fieldnames(map);
for i = 1:numel(src)
    v = opts.(src{i});
    if ~isnan(v)
        cfg.(map.(src{i})) = v;
    end
end
end


% ======================================================================= %
function rgb = toRGBUint8(img)
%TORGBUINT8  Coerce any input to an HxWx3 uint8 RGB array (mirrors _load_image
%   + _as_rgb_array on the Python side).
if ndims(img) == 2                                     %#ok<ISMAT> grayscale
    img = repmat(img, 1, 1, 3);
elseif size(img, 3) == 1
    img = repmat(img, 1, 1, 3);
elseif size(img, 3) == 4                               % drop alpha
    img = img(:, :, 1:3);
end
if ~isa(img, "uint8")
    img = im2uint8(img);
end
rgb = img;
end


% ======================================================================= %
function s = pngBase64(overlayUint8)
%PNGBASE64  PNG-encode an HxWx3 uint8 image and base64 it, matching
%   src/explain/gradcam.py::to_b64_png (raw base64 ASCII, no data: URI prefix,
%   no line breaks). imwrite has no in-memory PNG sink, so a temp file is the
%   simplest reliable path; it is removed immediately.
tmp = tempname + ".png";   % string scalar ([tempname, ".png"] would be a 1x2 string array)
cleaner = onCleanup(@() removeIfExists(tmp));
imwrite(overlayUint8, tmp);
fid = fopen(tmp, "r");
assert(fid > 0, "screen_image:pngRead", "Cannot re-read temp PNG %s", tmp);
bytes = fread(fid, Inf, "*uint8");
fclose(fid);
try
    s = string(matlab.net.base64encode(bytes));
catch
    s = string(matlab.net.base64encode(char(bytes')));  %#ok<CHARTEN>
end
end


% ======================================================================= %
function removeIfExists(p)
if isfile(p)
    delete(p);
end
end
