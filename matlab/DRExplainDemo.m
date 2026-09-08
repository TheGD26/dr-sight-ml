%% DR-Sight - Explainability demo  (SIH PS26038, Team Diasight)
% End-to-end: import the ONNX model, grade one fundus image with |grade_dr|,
% then run |explain_gradcam| to produce the Grad-CAM attention map, the
% ranked regions of interest, and the one-glance annotated referral report.
%
% This file is a cell-formatted script. Open it in MATLAB and either:
%   * run section-by-section with Ctrl+Enter (figures + console output), or
%   * File > Save As... > "DRExplainDemo.mlx" to get the Live Script view
%     (report + heatmap render inline under each section) for the judge demo.
%   This matches how matlab/DRScreeningDemo.m is shipped.
%
% Prerequisites (run once, from the repo root, in a terminal):
%     python scripts/export_onnx.py            % -> models/dr_sight.onnx
%
% Requires: Deep Learning Toolbox + "Deep Learning Toolbox Converter for ONNX
% Model Format" (importNetworkFromONNX, gradCAM) and Image Processing Toolbox
% (bwconncomp/regionprops/viscircles).
%
% NOT a certified medical device. Screening aid only - confirm with an
% ophthalmologist.

%% 1. Locate the repo and add this folder to the path
thisDir = fileparts(matlab.desktop.editor.getActiveFilename);
if isempty(thisDir); thisDir = pwd; end
repoRoot = fullfile(thisDir, "..");
addpath(thisDir);

onnxPath = fullfile(repoRoot, "models", "dr_sight.onnx");
assert(isfile(onnxPath), ...
    "Missing %s - run:  python scripts/export_onnx.py", onnxPath);

%% 2. Import the ONNX model once
% NCHW input "input" -> data format "BCSS" (batch, channel, spatial, spatial).
net = importNetworkFromONNX(onnxPath, InputDataFormats="BCSS");
if ~net.Initialized
    net = initialize(net);
end
disp(net)

%% 3. Pick a sample fundus image
% Any image from the APTOS cache; swap in your own path for the live demo.
sampleDir = fullfile(repoRoot, "data", "aptos", "cache");
listing   = dir(fullfile(sampleDir, "*.png"));
assert(~isempty(listing), "No sample images under %s", sampleDir);

imgPath = fullfile(listing(1).folder, listing(1).name);   % <-- change index / path
fprintf("Sample image: %s\n", imgPath);

figure("Name", "Input fundus image", "Color", "w");
imshow(imread(imgPath));
title("Input fundus image", "Interpreter", "none");

%% 4. Grade the image (grade_dr.m -> src/inference/pipeline.py parity)
result = grade_dr(imgPath, net);
disp(result)

fprintf("\nGrade %d - %s   (confidence %.1f%%)\n", ...
    result.grade, result.label, 100 * result.confidence);
fprintf("P(referable DR, grade>=2) = %.1f%%  ->  referable: %s\n", ...
    100 * result.p_referable, string(result.referable));
fprintf("Referral action: %s\n", result.referral_action);
if result.uncertain
    fprintf("** low confidence - flag for mandatory human review **\n");
end

%% 5. Explain the prediction (Grad-CAM + ROIs + annotated report)
% Pass the grade_dr result through so the two stay consistent. The report
% figure is created here; the struct is returned for programmatic use / JSON.
[report, reportFig] = explain_gradcam(imgPath, net, result);

% In a Live Script this displays the report figure inline right here.
% (Uncomment to bring it forward when running as a plain script.)
% figure(reportFig);

%% 6. Inspect the regions of interest
% Same shape as pipeline.py::cam_regions(): label, x/y (percent of W/H),
% radius (percent of W, clamped 2-25), intensity (mean CAM in blob, 0-1).
if isempty(report.regions)
    disp("No distinct model-focus region above 0.55 * max activation.");
else
    roiTable = struct2table(report.regions);
    disp(roiTable)
end

%% 7. Side-by-side: input vs Grad-CAM overlay
figure("Name", "Grad-CAM overlay", "Color", "w", "Position", [100 100 980 520]);
tiledlayout(1, 2, "TileSpacing", "compact", "Padding", "compact");

nexttile;
imshow(imresize(imread(imgPath), [224 224]));
title("Input (letterboxed to 224)");

nexttile;
imshow(report.overlay);
hold on
[H, W, ~] = size(report.overlay);
for i = 1:numel(report.regions)
    r = report.regions(i);
    viscircles([r.x/100*W, r.y/100*H], max(r.radius/100*W, 6), ...
        "Color", "w", "LineWidth", 1.5, "EnhanceVisibility", false);
    text(r.x/100*W, r.y/100*H, sprintf("%d", i), "Color", "w", ...
        "FontWeight", "bold", "HorizontalAlignment", "center");
end
hold off
title(sprintf("Grad-CAM (class %d) - layer: %s", ...
    report.target_class, report.feature_layer), "Interpreter", "none");

%% 8. Write the annotated report to disk (PDF + JSON)
outDir = fullfile(thisDir, "explain_reports");
if ~isfolder(outDir); mkdir(outDir); end
[~, stem] = fileparts(imgPath);

explain_gradcam(imgPath, net, result, ...
    ReportPath = fullfile(outDir, stem + "_report.pdf"), ShowFigure = false);
explain_gradcam(imgPath, net, result, ...
    ReportPath = fullfile(outDir, stem + "_report.json"), ShowFigure = false);

fprintf("\nWrote report PDF + JSON (+ overlay PNG) to:\n  %s\n", outDir);

%% 9. (Optional) confidence-calibration hook
% The reported confidence is the raw softmax top-1. explain_gradcam accepts a
% logit Temperature for post-hoc calibration (fit offline on a held-out set
% via a reliability curve / ECE - not done in this prototype). T = 1 is a
% no-op; T > 1 softens over-confident scores.
calDemo = explain_gradcam(imgPath, net, result, Temperature = 1.5, ShowFigure = false);
fprintf("raw confidence      : %.1f%%\n", 100 * report.confidence);
fprintf("confidence @ T=1.5  : %.1f%%\n", 100 * calDemo.confidence);

%% 10. Batch a folder into individual reports (screening throughput view)
files = dir(fullfile(sampleDir, "*.png"));
files = files(1:min(8, numel(files)));
rows  = table('Size', [numel(files) 5], ...
    'VariableTypes', {'string','double','string','double','double'}, ...
    'VariableNames', {'image','grade','label','confidence','n_roi'});
for k = 1:numel(files)
    p = fullfile(files(k).folder, files(k).name);
    g = grade_dr(char(p), net);
    e = explain_gradcam(char(p), net, g, ShowFigure = false);
    rows(k, :) = {string(files(k).name), e.grade, e.label, e.confidence, numel(e.regions)};
end
disp(rows)

%% Notes for the demo
% * "< 30 s review" - the report figure (section 5) is the deliverable to put
%   in front of a judge: big grade/label + confidence at the top, heatmap with
%   numbered focus regions in the middle, referral action + 2-line summary at
%   the bottom, disclaimer always visible.
% * The Grad-CAM target layer is auto-selected (last spatial layer before
%   global pooling; MobileNetV3-Small's conv_head sits *after* the pool and is
%   useless for a CAM). Override with explain_gradcam(..., FeatureLayer="...").
% * ROIs are grounded entirely in the model's own Grad-CAM - no lesion
%   detection is claimed. Per-lesion segmentation is separate future work
%   (needs IDRiD-style pixel masks + dedicated U-Net models).
