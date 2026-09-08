function [report, fig] = explain_gradcam(imageInput, net, gradeResult, opts)
%EXPLAIN_GRADCAM  Grad-CAM attention map + annotated referral report (SIH PS26038).
%
%   report            = EXPLAIN_GRADCAM(imagePath)
%   report            = EXPLAIN_GRADCAM(imageMatrix, net)
%   report            = EXPLAIN_GRADCAM(img, net, gradeResult)
%   [report, fig]     = EXPLAIN_GRADCAM(..., Name=Value)
%
%   The MATLAB counterpart of dr-sight-ml/src/explain/gradcam.py. Runs the
%   Deep Learning Toolbox built-in GRADCAM() on the DR-Sight network imported
%   from models/dr_sight.onnx (MobileNetV3-Small, 5-class), computes the class
%   activation map for the predicted grade, blends it as a jet heatmap over the
%   fundus image, extracts up to four "model focus" regions of interest, and
%   assembles a one-glance annotated report (big grade/label, heatmap, 2-3 line
%   summary, referral action) meant to be read by an ophthalmologist in well
%   under 30 seconds.
%
%   INPUTS
%     imageInput  : path to an image file, OR an HxWx3 uint8/RGB array.
%     net         : (optional) dlnetwork from importNetworkFromONNX. If omitted,
%                   ../models/dr_sight.onnx is imported once and cached.
%     gradeResult : (optional) the struct returned by grade_dr(). If omitted it
%                   is computed here via grade_dr(imageInput, net).
%
%   NAME-VALUE OPTIONS
%     FeatureLayer  (string)  Grad-CAM target layer name. "" (default) ->
%                             auto-select the last spatial layer before global
%                             pooling (see pickFeatureLayer below).
%     ReportPath    (string)  "" (default) writes no file. A path ending in
%                             ".pdf" exports the report figure to PDF (via
%                             MATLAB Report Generator when available, else
%                             exportgraphics). A path ending in ".json" writes a
%                             JSON report plus a "<stem>_overlay.png".
%     Alpha         (double)  heatmap blend weight (default 0.5, matches
%                             PipelineConfig.gradcam_alpha).
%     RelThreshold  (double)  ROI mask = cam >= RelThreshold*max(cam)
%                             (default 0.55, matches cam_regions()).
%     MaxRegions    (double)  max ROIs to return (default 4).
%     Temperature   (double)  logit temperature for the reported confidence
%                             (default 1.0 = raw softmax). A hook for post-hoc
%                             confidence calibration; >1 softens, <1 sharpens.
%                             Fit it offline on a held-out set (reliability
%                             curve / ECE) - not done in this prototype.
%     ShowFigure    (logical) create and show the report figure (default true).
%     UseReportGenerator (logical) allow the Report Generator PDF path when the
%                             toolbox is licensed (default true).
%
%   OUTPUT  report struct with fields:
%     grade, label, confidence, probabilities, p_referable, referable,
%     referral_escalated, referral_action, uncertain   (mirror grade_dr)
%     target_class     : class index the CAM was computed for (= grade)
%     feature_layer    : network layer the CAM was taken from
%     cam              : HxW double in [0,1], the normalised activation map
%     overlay          : HxWx3 uint8, jet heatmap alpha-blended on the image
%     regions          : 1xN struct array (label, x, y, radius, intensity) with
%                        x/y the blob centroid as percent of width/height,
%                        radius the area-equivalent radius as percent of width
%                        (clamped 2-25), intensity the mean CAM in the blob
%                        (0-1) - identical shape to pipeline.py::cam_regions().
%     disclaimer       : the standing screening-aid disclaimer.
%
%   Grading scale, referral actions, referable minimum grade, ImageNet
%   normalisation and the decision thresholds all mirror src/config.py - keep
%   them in sync with grade_dr.m if that file changes.
%
%   Smart India Hackathon 2026 prototype (Team Diasight). NOT a certified
%   medical device. Every grade must be confirmed by an ophthalmologist.

arguments
    imageInput
    net = getDefaultNet()
    gradeResult struct = struct([])
    opts.FeatureLayer (1,1) string = ""
    opts.ReportPath (1,1) string = ""
    opts.Alpha (1,1) double {mustBePositive} = 0.5
    opts.RelThreshold (1,1) double {mustBePositive} = 0.55
    opts.MaxRegions (1,1) double {mustBeInteger, mustBePositive} = 4
    opts.Temperature (1,1) double {mustBePositive} = 1.0
    opts.ShowFigure (1,1) logical = true
    opts.UseReportGenerator (1,1) logical = true
end

here = fileparts(mfilename("fullpath"));
addpath(here);   % so grade_dr.m is reachable when called directly

% --- constants, mirrored from src/config.py -------------------------------- %
INPUT_SIZE    = 224;
IMAGENET_MEAN = reshape([0.485, 0.456, 0.406], 1, 1, 3);
IMAGENET_STD  = reshape([0.229, 0.224, 0.225], 1, 1, 3);
GRADE_LABELS  = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"];
REFERABLE_DR_MIN_GRADE = 2;
REFERABLE_THRESHOLD    = 0.5;
UNCERTAINTY_THRESHOLD  = 0.6;
DISCLAIMER = "AI screening aid only - not a diagnosis. Confirm with an ophthalmologist.";
REFERRAL_ACTIONS = referralActionsMap();   % containers.Map, 0..4 -> action string

% --- read image ---------------------------------------------------------- %
if ischar(imageInput) || isstring(imageInput)
    img = imread(char(imageInput));
else
    img = imageInput;
end
if ndims(img) == 2                                     %#ok<ISMAT> grayscale
    img = repmat(img, 1, 1, 3);
end
if size(img, 3) == 4
    img = img(:, :, 1:3);
end
img = im2uint8(img);

% --- preprocess: letterbox to 224 (the image the model actually sees) ----- %
canvas = letterboxResize(img, INPUT_SIZE);                        % uint8 HxWx3
x      = normalizeImageNet(canvas, IMAGENET_MEAN, IMAGENET_STD);  % single HxWx3

% NCHW batch-1, matching the exported ONNX input "input".
X   = reshape(permute(x, [3, 1, 2]), [1, 3, INPUT_SIZE, INPUT_SIZE]);
dlX = dlarray(single(X), "BCSS");

% --- forward pass (own copy so Temperature can act on the logits) --------- %
logits = predict(net, dlX);
logits = double(gather(extractdata(logits)));
logits = logits(:).';                                  % 1x5 row
probs  = softmaxRow(logits ./ opts.Temperature);
[confidence, idx] = max(probs);
grade = idx - 1;

if ~isempty(fieldnames(gradeResult)) && isfield(gradeResult, "grade") ...
        && gradeResult.grade ~= grade
    warning("explain_gradcam:gradeMismatch", ...
        "grade_dr() said grade %d but this pass said %d (Temperature=%.3g). " + ...
        "Explaining grade %d.", gradeResult.grade, grade, opts.Temperature, grade);
end

pReferable = sum(probs((REFERABLE_DR_MIN_GRADE + 1):end));
referable  = pReferable >= REFERABLE_THRESHOLD;
escalated  = referable && grade < REFERABLE_DR_MIN_GRADE;
actionGrade = grade;
if escalated
    actionGrade = REFERABLE_DR_MIN_GRADE;
end

% --- Grad-CAM ---------------------------------------------------------- %
if strlength(opts.FeatureLayer) > 0
    featureLayer = opts.FeatureLayer;
else
    featureLayer = pickFeatureLayer(net, dlX);
end

classIdx     = grade + 1;
reductionFcn = @(y) y(classIdx);   % score = logit of the predicted grade
try
    scoreMap = gradCAM(net, dlX, reductionFcn, ...
        FeatureLayer = char(featureLayer), ...
        OutputUpsampling = "bicubic");
catch err
    names = string({net.Layers.Name});
    error("explain_gradcam:gradcamFailed", ...
        "gradCAM failed on FeatureLayer='%s' (%s).\n" + ...
        "Pass an explicit FeatureLayer=... . Candidate spatial layers:\n%s", ...
        featureLayer, err.message, strjoin("  " + names, newline));
end
if isa(scoreMap, "dlarray")
    scoreMap = extractdata(scoreMap);
end
scoreMap = double(gather(scoreMap));
scoreMap = squeeze(scoreMap);

% normalise to [0,1] exactly like pytorch-grad-cam's per-image min-max scaling
lo = min(scoreMap(:));
hi = max(scoreMap(:));
if hi > lo
    cam = (scoreMap - lo) ./ (hi - lo);
else
    cam = zeros(size(scoreMap));
end
if ~isequal(size(cam), [INPUT_SIZE, INPUT_SIZE])
    cam = imresize(cam, [INPUT_SIZE, INPUT_SIZE]);
    cam = min(max(cam, 0), 1);
end

% --- overlay + regions of interest ---------------------------------------- %
overlay = buildOverlay(canvas, cam, opts.Alpha);
regions = camRegions(cam, opts.MaxRegions, opts.RelThreshold);

% --- assemble report ---------------------------------------------------- %
report = struct( ...
    "grade",              grade, ...
    "label",              GRADE_LABELS(idx), ...
    "confidence",         confidence, ...
    "probabilities",      probs, ...
    "p_referable",        pReferable, ...
    "referable",          referable, ...
    "referral_escalated", escalated, ...
    "referral_action",    string(REFERRAL_ACTIONS(actionGrade)), ...
    "uncertain",          confidence < UNCERTAINTY_THRESHOLD, ...
    "target_class",       grade, ...
    "feature_layer",      string(featureLayer), ...
    "temperature",        opts.Temperature, ...
    "cam",                cam, ...
    "overlay",            overlay, ...
    "regions",            regions, ...
    "disclaimer",         DISCLAIMER);

printSummary(report);

% --- figure + optional file ------------------------------------------------ %
fig = gobjects(0);
needFig = opts.ShowFigure || endsWith(lower(opts.ReportPath), ".pdf");
if needFig
    fig = renderReportFigure(report);
    if ~opts.ShowFigure
        set(fig, "Visible", "off");
    end
end

if strlength(opts.ReportPath) > 0
    writeReport(report, opts.ReportPath, fig, opts.UseReportGenerator);
end
end


% ======================================================================= %
% Region-of-interest extraction - mirrors src/inference/pipeline.py::cam_regions
% ======================================================================= %
function regions = camRegions(cam, maxRegions, relThreshold)
regions = struct("label", {}, "x", {}, "y", {}, "radius", {}, "intensity", {});
if isempty(cam)
    return
end
cam = double(cam);
cam(~isfinite(cam)) = 0;
peak = max(cam(:));
if isempty(peak) || peak <= 1e-6
    return
end
[h, w] = size(cam);
mask = cam >= relThreshold * peak;
if ~any(mask(:))
    return
end

cc = bwconncomp(mask, 8);
stats = regionprops(cc, "Area", "Centroid", "PixelIdxList");
minArea = max(9.0, 0.002 * h * w);   % ignore specks

cand   = struct("label", {}, "x", {}, "y", {}, "radius", {}, "intensity", {});
scores = [];
for i = 1:numel(stats)
    area = double(stats(i).Area);
    if area < minArea
        continue
    end
    c  = stats(i).Centroid;                 % [x y], geometric (unweighted)
    cx = c(1);
    cy = c(2);
    intensity = mean(cam(stats(i).PixelIdxList));
    radiusPx  = sqrt(area / pi);

    r = struct();
    r.label     = "";                       % filled after ranking
    r.x         = round(cx / w * 100.0, 1);
    r.y         = round(cy / h * 100.0, 1);
    r.radius    = round(min(25.0, max(2.0, radiusPx / w * 100.0)), 1);
    r.intensity = round(min(1.0, max(0.0, intensity)), 3);

    cand(end + 1)   = r;              %#ok<AGROW>
    scores(end + 1) = area * intensity; %#ok<AGROW>
end
if isempty(cand)
    return
end

[~, order] = sort(scores, "descend");
k = min(maxRegions, numel(order));
regions = cand(order(1:k));
for rank = 1:k
    regions(rank).label = "Model focus region " + rank;
end
regions = orderfields(regions, {'label', 'x', 'y', 'radius', 'intensity'});
end


% ======================================================================= %
function overlay = buildOverlay(baseRGB, cam, alpha)
%BUILDOVERLAY  (1-alpha)*image + alpha*JET(cam), matches gradcam.py::overlay_cam.
base   = im2double(baseRGB);
camIdx = gray2ind(mat2gray(cam), 256);
heat   = ind2rgb(camIdx, jet(256));                 % double [0,1]
if ~isequal(size(heat, 1:2), size(base, 1:2))
    heat = imresize(heat, [size(base, 1), size(base, 2)]);
end
blended = (1 - alpha) * base + alpha * heat;
overlay = im2uint8(min(max(blended, 0), 1));
end


% ======================================================================= %
function name = pickFeatureLayer(net, dlX)
%PICKFEATURELAYER  Last convolutional / activation layer that still has a
%   spatial map, i.e. the feature block feeding global pooling. For the
%   ONNX-imported MobileNetV3-Small this is the HardSwish after
%   blocks.5.0.conv, just before GlobalAveragePool (conv_head sits AFTER the
%   pool in MobileNetV3-Small, so it is 1x1 and useless for a CAM - same
%   reasoning as DRModel.gradcam_target_layer() in src/model/architecture.py).
layers = net.Layers;

% 1) find a global pooling layer, take its source layer via the connection table
isGap = arrayfun(@(L) contains(class(L), "GlobalAveragePooling2DLayer") || ...
                      contains(class(L), "GlobalMaxPooling2DLayer") || ...
                      contains(lower(string(L.Name)), "globalaveragepool") || ...
                      contains(lower(string(L.Name)), "globalmaxpool"), layers);
gapIdx = find(isGap, 1);
if ~isempty(gapIdx)
    gapName = string(layers(gapIdx).Name);
    conns   = net.Connections;
    dst     = string(conns.Destination);
    row     = find(dst == gapName | startsWith(dst, gapName + "/"), 1);
    if ~isempty(row)
        src = string(conns.Source(row));
        src = extractBefore(src + "/", "/");   % strip "/out" port suffix
        if strlength(src) > 0 && hasSpatialOutput(net, src, dlX)
            name = src;
            return
        end
    end
end

% 2) probe backwards for the last layer with a >1x1 spatial output
for i = numel(layers):-1:1
    nm = string(layers(i).Name);
    c  = string(class(layers(i)));
    looksSpatial = contains(c, ["Convolution2DLayer", "ReLULayer", ...
        "ClippedReLULayer", "SwishLayer", "BatchNormalizationLayer", ...
        "FunctionLayer", "GroupNormalizationLayer"]) || ...
        contains(lower(nm), ["hardswish", "swish", "relu", "conv", "act"]);
    if looksSpatial && hasSpatialOutput(net, nm, dlX)
        name = nm;
        return
    end
end

error("explain_gradcam:noFeatureLayer", ...
    "Could not auto-select a Grad-CAM feature layer. Pass FeatureLayer=... .\n" + ...
    "Network layers:\n%s", strjoin("  " + string({layers.Name}), newline));
end


% ======================================================================= %
function tf = hasSpatialOutput(net, layerName, dlX)
tf = false;
try
    a  = predict(net, dlX, Outputs = char(layerName));
    sz = size(a, 1:2);
    tf = numel(sz) >= 2 && all(sz > 1);
catch
    tf = false;
end
end


% ======================================================================= %
function fig = renderReportFigure(report)
%RENDERREPORTFIGURE  One-glance layout: big grade/label, heatmap with numbered
%   ROIs, referral banner, 2-3 line summary. Built for a <30 s read.
sev = severityColor(report.grade);

fig = figure("Name", "DR-Sight - Explainability Report", "Color", "w", ...
    "Position", [80, 80, 900, 820]);
tl = tiledlayout(fig, 4, 1, "TileSpacing", "compact", "Padding", "compact");
title(tl, "DR-Sight  -  Explainable DR Screening Report", ...
    "FontWeight", "bold", "FontSize", 13);

% -- header --
axH = nexttile(tl);
axis(axH, [0 1 0 1]); axis(axH, "off");
text(axH, 0.00, 0.70, sprintf("GRADE %d", report.grade), ...
    "FontSize", 34, "FontWeight", "bold", "Color", sev);
text(axH, 0.00, 0.24, report.label, "FontSize", 22, "Color", sev);
text(axH, 1.00, 0.70, sprintf("%.0f%%", 100 * report.confidence), ...
    "FontSize", 30, "FontWeight", "bold", "HorizontalAlignment", "right");
lbl = "model confidence (raw softmax)";
if report.temperature ~= 1
    lbl = sprintf("model confidence (T=%.2g)", report.temperature);
end
text(axH, 1.00, 0.24, lbl, "FontSize", 11, "Color", [.4 .4 .4], ...
    "HorizontalAlignment", "right");
if report.uncertain
    text(axH, 0.00, -0.05, "LOW CONFIDENCE - mandatory human review", ...
        "FontSize", 12, "FontWeight", "bold", "Color", [0.75 0.2 0.0]);
end

% -- heatmap with numbered ROIs (spans two tiles) --
axImg = nexttile(tl, [2 1]);
imshow(report.overlay, "Parent", axImg);
hold(axImg, "on");
[H, W, ~] = size(report.overlay);
for i = 1:numel(report.regions)
    r  = report.regions(i);
    px = r.x / 100 * W;
    py = r.y / 100 * H;
    pr = max(r.radius / 100 * W, 6);
    drawCircle(axImg, px, py, pr);
    text(axImg, px, py, sprintf("%d", i), "Parent", axImg, ...
        "Color", "w", "FontWeight", "bold", "FontSize", 12, ...
        "HorizontalAlignment", "center", "VerticalAlignment", "middle");
end
hold(axImg, "off");
title(axImg, sprintf("Grad-CAM for class %d (%s)  -  %d model-focus region(s)   [layer: %s]", ...
    report.target_class, report.label, numel(report.regions), report.feature_layer), ...
    "Interpreter", "none", "FontSize", 10);

% -- footer: referral banner + summary --
axF = nexttile(tl);
axis(axF, [0 1 0 1]); axis(axF, "off");
rectangle(axF, "Position", [0 0.62 1 0.38], ...
    "FaceColor", 1 - 0.16 * (1 - sev), "EdgeColor", "none");   % tint of sev
text(axF, 0.01, 0.81, "REFERRAL:  " + report.referral_action, ...
    "FontSize", 15, "FontWeight", "bold", "Color", sev * 0.7);

escNote = "";
if report.referral_escalated
    escNote = "  (escalated by P(referable) threshold, not by argmax grade)";
end
roiTxt = "no distinct focus region above threshold";
if ~isempty(report.regions)
    roiTxt = strjoin(arrayfun(@(k) sprintf("#%d (%.0f%%,%.0f%%) int %.2f", k, ...
        report.regions(k).x, report.regions(k).y, report.regions(k).intensity), ...
        1:numel(report.regions), "UniformOutput", false), "   ");
end
summary = sprintf([ ...
    "P(referable DR, grade >= 2) = %.0f%%   ->   referable: %s%s\n", ...
    "Model focus: %s\n", ...
    "%s"], ...
    100 * report.p_referable, upper(string(report.referable)), escNote, ...
    roiTxt, report.disclaimer);
text(axF, 0.01, 0.42, summary, "FontSize", 10.5, "VerticalAlignment", "top", ...
    "Interpreter", "none");
end


% ======================================================================= %
function drawCircle(ax, cx, cy, r)
try
    viscircles(ax, [cx, cy], r, "Color", "w", "LineWidth", 1.5, ...
        "EnhanceVisibility", false);
catch
    rectangle(ax, "Position", [cx - r, cy - r, 2 * r, 2 * r], ...
        "Curvature", [1 1], "EdgeColor", "w", "LineWidth", 1.5);
end
end


% ======================================================================= %
function c = severityColor(grade)
palette = [ ...
    0.13 0.55 0.13; ...   % 0 No DR         - green
    0.60 0.60 0.10; ...   % 1 Mild NPDR     - olive
    0.90 0.55 0.00; ...   % 2 Moderate NPDR - orange
    0.85 0.33 0.10; ...   % 3 Severe NPDR   - dark orange
    0.75 0.10 0.10];      % 4 Proliferative - red
c = palette(min(max(grade, 0), 4) + 1, :);
end


% ======================================================================= %
function writeReport(report, reportPath, fig, useReportGen)
reportPath = string(reportPath);
lower_ = lower(reportPath);

if endsWith(lower_, ".json")
    [p, n] = fileparts(reportPath);
    pngPath = fullfile(p, n + "_overlay.png");
    imwrite(report.overlay, pngPath);

    j = struct();
    j.grade             = report.grade;
    j.label             = report.label;
    j.confidence        = report.confidence;
    j.probabilities     = report.probabilities;
    j.p_referable       = report.p_referable;
    j.referable         = report.referable;
    j.referral_escalated = report.referral_escalated;
    j.referral_action   = report.referral_action;
    j.uncertain         = report.uncertain;
    j.target_class      = report.target_class;
    j.feature_layer     = report.feature_layer;
    j.temperature       = report.temperature;
    j.regions           = report.regions;
    j.overlay_png       = pngPath;
    j.overlay_png_base64 = base64OfFile(pngPath);
    j.disclaimer        = report.disclaimer;

    try
        txt = jsonencode(j, "PrettyPrint", true);
    catch
        txt = jsonencode(j);
    end
    fid = fopen(reportPath, "w");
    assert(fid > 0, "explain_gradcam:jsonOpen", "Cannot write %s", reportPath);
    fwrite(fid, txt, "char");
    fclose(fid);
    fprintf("  report JSON  -> %s\n  overlay PNG  -> %s\n", reportPath, pngPath);
    return
end

if endsWith(lower_, ".pdf")
    if useReportGen && tryReportGenerator(report, reportPath)
        fprintf("  report PDF   -> %s   (MATLAB Report Generator)\n", reportPath);
        return
    end
    if isempty(fig) || ~isgraphics(fig)
        fig = renderReportFigure(report);
    end
    try
        exportgraphics(fig, reportPath, "ContentType", "vector", ...
            "BackgroundColor", "white");
    catch
        print(fig, char(reportPath), "-dpdf", "-bestfit");
    end
    fprintf("  report PDF   -> %s   (figure export)\n", reportPath);
    return
end

warning("explain_gradcam:unknownReport", ...
    "ReportPath '%s' has no .pdf/.json extension - nothing written.", reportPath);
end


% ======================================================================= %
function ok = tryReportGenerator(report, pdfPath)
ok = false;
if exist("mlreportgen.report.Report", "class") ~= 8
    return
end
try
    import mlreportgen.report.*
    import mlreportgen.dom.*

    stem = char(erase(string(pdfPath), ".pdf" + textBoundary("end")));
    rpt  = Report(stem, "pdf");

    tp = TitlePage();
    tp.Title    = "DR-Sight - Explainable DR Screening Report";
    tp.Subtitle = sprintf("Grade %d  -  %s  (%.0f%% confidence)", ...
        report.grade, report.label, 100 * report.confidence);
    tp.Author   = "DR-Sight prototype (SIH PS26038, Team Diasight)";
    add(rpt, tp);

    tmpPng = [tempname, ".png"];
    imwrite(report.overlay, tmpPng);
    img = Image(tmpPng);
    img.Style = {ScaleToFit(true)};
    add(rpt, img);

    pRef = Paragraph(sprintf("Referral: %s", report.referral_action));
    pRef.Bold = true;
    pRef.FontSize = "14pt";
    add(rpt, pRef);

    add(rpt, Paragraph(sprintf( ...
        "P(referable DR, grade >= 2) = %.0f%%    referable: %s    uncertain: %s", ...
        100 * report.p_referable, string(report.referable), string(report.uncertain))));

    if ~isempty(report.regions)
        add(rpt, Heading2("Model focus regions"));
        for k = 1:numel(report.regions)
            r = report.regions(k);
            add(rpt, Paragraph(sprintf( ...
                "%d. %s - centroid (%.0f%%, %.0f%%), radius %.1f%% w, intensity %.2f", ...
                k, r.label, r.x, r.y, r.radius, r.intensity)));
        end
    end

    disc = Paragraph(report.disclaimer);
    disc.Italic = true;
    add(rpt, disc);

    close(rpt);
    ok = true;
catch
    ok = false;
end
end


% ======================================================================= %
function s = base64OfFile(path)
fid = fopen(path, "r");
if fid < 0
    s = "";
    return
end
bytes = fread(fid, Inf, "*uint8");
fclose(fid);
try
    s = string(matlab.net.base64encode(bytes));
catch
    s = string(matlab.net.base64encode(char(bytes')));  %#ok<CHARTEN>
end
end


% ======================================================================= %
function m = referralActionsMap()
%REFERRALACTIONSMAP  src/config.py::REFERRAL_ACTIONS as a containers.Map.
m = containers.Map("KeyType", "double", "ValueType", "char");
m(0) = 'No referral. Routine annual screening.';
m(1) = 'No urgent referral. Re-screen in 9-12 months.';
m(2) = 'Refer to ophthalmologist within ~6 months.';
m(3) = 'Refer within ~1 month - treat as priority.';
m(4) = 'Urgent referral within days.';
end


% ======================================================================= %
function canvas = letterboxResize(img, inputSize)
%LETTERBOXRESIZE  LongestMaxSize + centre PadIfNeeded (value 0), as in
%   src/data/dataset.py::build_transforms(train=False) and grade_dr.m.
[h, w, ~] = size(img);
s  = inputSize / max(h, w);
nh = round(h * s);
nw = round(w * s);
resized = imresize(img, [nh, nw], "bilinear", "Antialiasing", false);
padT = floor((inputSize - nh) / 2);
padL = floor((inputSize - nw) / 2);
canvas = zeros(inputSize, inputSize, 3, "uint8");
canvas((padT + 1):(padT + nh), (padL + 1):(padL + nw), :) = resized;
end


% ======================================================================= %
function out = normalizeImageNet(canvas, meanRGB, stdRGB)
out = single(canvas) / 255;
out = (out - single(meanRGB)) ./ single(stdRGB);
end


% ======================================================================= %
function p = softmaxRow(logits)
z = logits - max(logits);
e = exp(z);
p = e / sum(e);
end


% ======================================================================= %
function printSummary(report)
fprintf("\n--- DR-Sight explainability ------------------------------------\n");
fprintf("  Grade %d  -  %s   (confidence %.1f%%)\n", ...
    report.grade, report.label, 100 * report.confidence);
fprintf("  P(referable, grade>=2) = %.1f%%   referable: %s%s\n", ...
    100 * report.p_referable, string(report.referable), ...
    ternary(report.referral_escalated, "  [escalated by threshold]", ""));
fprintf("  Referral: %s\n", report.referral_action);
if report.uncertain
    fprintf("  ** low confidence - flag for mandatory human review **\n");
end
fprintf("  Grad-CAM layer: %s   focus regions: %d\n", ...
    report.feature_layer, numel(report.regions));
for k = 1:numel(report.regions)
    r = report.regions(k);
    fprintf("    %d. (%.0f%%, %.0f%%)  r=%.1f%%  intensity=%.2f\n", ...
        k, r.x, r.y, r.radius, r.intensity);
end
fprintf("---------------------------------------------------------------\n");
end


% ======================================================================= %
function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end


% ======================================================================= %
function net = getDefaultNet()
%GETDEFAULTNET  Import ../models/dr_sight.onnx once and cache it (mirrors
%   grade_dr.m so explain_gradcam can be called on its own).
persistent cachedNet
if ~isempty(cachedNet)
    net = cachedNet;
    return
end
here = fileparts(mfilename("fullpath"));
onnxPath = fullfile(here, "..", "models", "dr_sight.onnx");
if ~isfile(onnxPath)
    error("explain_gradcam:onnxMissing", ...
        ["Cannot find %s.\nExport it first from the repo root:\n" ...
         "    python scripts/export_onnx.py"], onnxPath);
end
try
    cachedNet = importNetworkFromONNX(onnxPath, InputDataFormats = "BCSS");
catch err
    error("explain_gradcam:importFailed", ...
        ["importNetworkFromONNX failed (%s).\n" ...
         "Needs the Deep Learning Toolbox Converter for ONNX Model Format."], ...
        err.message);
end
if ~cachedNet.Initialized
    cachedNet = initialize(cachedNet);
end
net = cachedNet;
end
