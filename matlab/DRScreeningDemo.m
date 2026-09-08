%% DR-Sight - MATLAB screening demo (SIH PS26038, Team Diasight)
% Interactive walkthrough of the MATLAB DR severity grading pipeline built on
% the PyTorch model exported to ONNX (models/dr_sight.onnx).
%
% Open this file in MATLAB and either run it section-by-section (Ctrl+Enter)
% or save it as a Live Script (Save As -> *.mlx) for the richer view.
%
% Prerequisites (run once, from the repo root, in a terminal):
%     python scripts/export_onnx.py            % -> models/dr_sight.onnx
%     python scripts/make_reference_csv.py --split test   % -> matlab/reference_test.csv
%
% Requires: Deep Learning Toolbox + "Deep Learning Toolbox Converter for ONNX
% Model Format" (for importNetworkFromONNX).
%
% NOT a certified medical device. Screening aid only - confirm with an
% ophthalmologist.

%% 1. Locate the repo and add this folder to the path
thisDir  = fileparts(matlab.desktop.editor.getActiveFilename);
if isempty(thisDir); thisDir = pwd; end
repoRoot = fullfile(thisDir, "..");
addpath(thisDir);

onnxPath = fullfile(repoRoot, "models", "dr_sight.onnx");
assert(isfile(onnxPath), ...
    "Missing %s - run:  python scripts/export_onnx.py", onnxPath);

%% 2. Import the ONNX model
% NCHW input "input" -> data format "BCSS" (batch, channel, spatial, spatial).
net = importNetworkFromONNX(onnxPath, InputDataFormats="BCSS");
if ~net.Initialized
    net = initialize(net);
end
disp(net)

%% 3. Grade a single fundus image
% Pick any image from the APTOS cache; swap in your own path here.
sampleDir = fullfile(repoRoot, "data", "aptos", "cache");
listing   = dir(fullfile(sampleDir, "*.png"));
assert(~isempty(listing), "No sample images under %s", sampleDir);
imgPath = fullfile(listing(1).folder, listing(1).name);

result = grade_dr(imgPath, net);
disp(result)

fprintf("\nGrade %d - %s   (confidence %.1f%%)\n", ...
    result.grade, result.label, 100 * result.confidence);
fprintf("P(referable DR, grade>=2) = %.1f%%  ->  referable: %s\n", ...
    100 * result.p_referable, string(result.referable));
if result.referral_escalated
    fprintf("   (referral escalated by threshold, not by argmax grade)\n");
end
fprintf("Referral action: %s\n", result.referral_action);
if result.uncertain
    fprintf("** low confidence - flag for mandatory human review **\n");
end

%% 4. Show the image with its grade
figure("Name", "DR-Sight grade", "Color", "w");
imshow(imread(imgPath));
title(sprintf("%s  (%.0f%%)  |  %s", ...
    result.label, 100 * result.confidence, result.referral_action), ...
    "Interpreter", "none");

%% 5. Bar chart of the class probabilities
labels = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"];
figure("Name", "Class probabilities", "Color", "w");
bar(categorical(labels, labels), result.probabilities);
ylabel("softmax probability"); ylim([0 1]);
title("DR-Sight class probabilities");

%% 6. Batch a folder of images
files = dir(fullfile(sampleDir, "*.png"));
files = files(1:min(12, numel(files)));
names   = strings(numel(files), 1);
grades  = zeros(numel(files), 1);
gLabels = strings(numel(files), 1);
conf    = zeros(numel(files), 1);
pRef    = zeros(numel(files), 1);
for k = 1:numel(files)
    p = fullfile(files(k).folder, files(k).name);
    r = grade_dr(char(p), net);
    names(k)   = string(files(k).name);
    grades(k)  = r.grade;
    gLabels(k) = r.label;
    conf(k)    = r.confidence;
    pRef(k)    = r.p_referable;
end
batchTable = table(names, grades, gLabels, conf, pRef, ...
    VariableNames=["image", "grade", "label", "confidence", "p_referable"]);
disp(batchTable)

%% 7. Validate the ONNX import against the original PyTorch predictions
% Compares grade_dr.m output to matlab/reference_test.csv (from
% scripts/make_reference_csv.py). Small floating-point drift is expected.
refCsv = fullfile(thisDir, "reference_test.csv");
if isfile(refCsv)
    summary = validate_grading(refCsv);
    disp(summary)
else
    fprintf("Skip: %s not found. Generate it with\n", refCsv);
    fprintf("    python scripts/make_reference_csv.py --split test\n");
end
