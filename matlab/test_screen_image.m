%% test_screen_image - end-to-end sanity check for screen_image.m
% SIH PS26038, Team Diasight.
%
% Runs the full MATLAB screening orchestrator (quality gate -> grade ->
% Grad-CAM -> referral) over a few real APTOS cache images and pretty-prints
% the PredictionResult-shaped struct, so the MATLAB path can be eyeballed
% directly in MATLAB before it is wired up behind FastAPI.
%
% Run from MATLAB:  >> test_screen_image
%
% Requires: Deep Learning Toolbox + "Deep Learning Toolbox Converter for ONNX
% Model Format", Image Processing Toolbox, Computer Vision Toolbox.

%% Locate the repo and wire up paths (same sample dir as DRScreeningDemo.m)
thisDir = fileparts(mfilename("fullpath"));
if isempty(thisDir); thisDir = pwd; end
addpath(thisDir);
repoRoot = fullfile(thisDir, "..");

sampleDir = fullfile(repoRoot, "data", "aptos", "cache");
listing   = dir(fullfile(sampleDir, "*.png"));
assert(~isempty(listing), "No sample images under %s", sampleDir);

nSamples = min(3, numel(listing));
paths = strings(nSamples, 1);
for i = 1:nSamples
    paths(i) = string(fullfile(listing(i).folder, listing(i).name));
end

%% Import the ONNX network ONCE and reuse it (mirrors the server path)
net = load_screening_net();
fprintf("Imported dr_sight.onnx -> %s\n\n", class(net));

%% Screen each sample and pretty-print the result
for i = 1:nSamples
    fprintf("================================================================\n");
    fprintf(" [%d/%d]  %s\n", i, nSamples, paths(i));
    fprintf("================================================================\n");

    t = tic;
    result = screen_image(paths(i), net);            % WithHeatmap defaults true
    fprintf("  screen_image() took %.2f s\n\n", toc(t));

    printResult(result);
    fprintf("\n");
end

%% Also exercise the fast path (WithHeatmap = false)
fprintf("================================================================\n");
fprintf(" WithHeatmap = false  (health-check / batch-grading path)\n");
fprintf("================================================================\n");
result = screen_image(paths(1), net, WithHeatmap = false);
printResult(result);

%% Confirm the field set matches PredictionResult.to_dict()
expected = ["usable_image","rejection_reason","grade","label","confidence", ...
    "uncertain","referable","p_referable","referral_escalated","referral_action", ...
    "heatmap_base64","affected_regions","probabilities","quality_scores","disclaimer"];
got = string(fieldnames(result)).';
assert(isequal(got, expected), ...
    "Field mismatch vs PredictionResult:\n  expected: %s\n  got:      %s", ...
    strjoin(expected, ", "), strjoin(got, ", "));
fprintf("\nOK - result fields match PredictionResult.to_dict() exactly.\n");


% ======================================================================= %
function printResult(r)
%PRINTRESULT  Pretty-print one screen_image() struct.
fprintf("  usable_image ........ %s\n", tf(r.usable_image));
if ~r.usable_image
    fprintf("  rejection_reason .... %s\n", r.rejection_reason);
    printScores(r.quality_scores);
    fprintf("  disclaimer .......... %s\n", r.disclaimer);
    return
end
fprintf("  grade / label ....... %d  (%s)\n", r.grade, r.label);
fprintf("  confidence .......... %.4f%s\n", r.confidence, ...
    ternary(r.uncertain, "   ** UNCERTAIN - human review **", ""));
fprintf("  probabilities ....... [%s]\n", strjoin(compose("%.4f", r.probabilities(:).'), "  "));
fprintf("  p_referable ......... %.4f\n", r.p_referable);
fprintf("  referable ........... %s%s\n", tf(r.referable), ...
    ternary(r.referral_escalated, "   (escalated by threshold, not argmax)", ""));
fprintf("  referral_action ..... %s\n", r.referral_action);
if isempty(r.heatmap_base64) || strlength(string(r.heatmap_base64)) == 0
    fprintf("  heatmap_base64 ...... <none>\n");
else
    b = char(r.heatmap_base64);
    fprintf("  heatmap_base64 ...... %s... (%d chars)\n", b(1:min(32, numel(b))), numel(b));
end
fprintf("  affected_regions .... %d region(s)\n", numel(r.affected_regions));
for k = 1:numel(r.affected_regions)
    a = r.affected_regions(k);
    fprintf("      %d. %-22s (x=%.1f%% y=%.1f%%) r=%.1f%% intensity=%.3f\n", ...
        k, a.label, a.x, a.y, a.radius, a.intensity);
end
printScores(r.quality_scores);
fprintf("  disclaimer .......... %s\n", r.disclaimer);
end


% ======================================================================= %
function printScores(s)
if isempty(s) || isempty(fieldnames(s))
    fprintf("  quality_scores ...... <none>\n");
    return
end
f = fieldnames(s);
parts = strings(1, numel(f));
for i = 1:numel(f)
    parts(i) = sprintf("%s=%.4g", f{i}, s.(f{i}));
end
fprintf("  quality_scores ...... %s\n", strjoin(parts, "  "));
end


% ======================================================================= %
function s = tf(v)
if v; s = "true"; else; s = "false"; end
end


% ======================================================================= %
function out = ternary(cond, a, b)
if cond; out = a; else; out = b; end
end
