function result = grade_dr(imageInput, net)
%GRADE_DR  Diabetic-retinopathy severity grading from a fundus image (SIH PS26038).
%
%   result = GRADE_DR(imagePath)
%   result = GRADE_DR(imageMatrix)
%   result = GRADE_DR(..., net)
%
%   Runs the DR-Sight model exported from PyTorch (models/dr_sight.onnx, a
%   MobileNetV3-Small 5-class classifier) on one fundus photo and returns the
%   International Clinical DR severity grade plus a screening referral decision.
%
%   INPUTS
%     imageInput : path to an image file, OR an HxWx3 uint8/RGB image array.
%     net        : (optional) dlnetwork from importNetworkFromONNX. If omitted,
%                  the ONNX model next to this file (../models/dr_sight.onnx) is
%                  imported once and cached for subsequent calls.
%
%   OUTPUT  result struct with fields:
%     grade             : integer 0-4  (argmax of the softmax)
%     label             : "No DR" | "Mild NPDR" | "Moderate NPDR" |
%                         "Severe NPDR" | "Proliferative DR"
%     confidence        : softmax probability of the predicted grade
%     probabilities     : 1x5 softmax vector (grades 0..4)
%     p_referable       : softmax mass on grades >= 2 (referable DR)
%     referable         : logical, p_referable >= referable_threshold
%     referral_escalated: true when `referable` is driven by the threshold
%                         while the argmax grade is < 2
%     referral_action   : recommended referral timeframe (string)
%     uncertain         : logical, confidence < uncertainty_threshold
%                         (flag for mandatory human review)
%
%   The grading scale, referral actions, the referable-DR minimum grade, the
%   ImageNet normalisation constants and the two decision thresholds below all
%   mirror dr-sight-ml/src/config.py -- keep them in sync if that file changes.
%
%   This is a Smart India Hackathon 2026 prototype, NOT a certified medical
%   device. Every grade must be confirmed by an ophthalmologist.

arguments
    imageInput
    net = getDefaultNet()
end

% --- constants, mirrored from src/config.py -------------------------------- %
INPUT_SIZE    = 224;                              % BACKBONES["edge"]["input_size"]
IMAGENET_MEAN = reshape([0.485, 0.456, 0.406], 1, 1, 3);
IMAGENET_STD  = reshape([0.229, 0.224, 0.225], 1, 1, 3);

GRADE_LABELS = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"];
REFERRAL_ACTIONS = [ ...
    "No referral. Routine annual screening."; ...
    "No urgent referral. Re-screen in 9-12 months."; ...
    "Refer to ophthalmologist within ~6 months."; ...
    "Refer within ~1 month - treat as priority."; ...
    "Urgent referral within days."];
REFERABLE_DR_MIN_GRADE = 2;      % grades >= this are "referable DR"
REFERABLE_THRESHOLD    = 0.5;    % PipelineConfig.referable_threshold default
UNCERTAINTY_THRESHOLD  = 0.6;    % PipelineConfig.uncertainty_threshold default

% --- read image --------------------------------------------------------- %
if ischar(imageInput) || isstring(imageInput)
    img = imread(char(imageInput));
else
    img = imageInput;
end
if ndims(img) == 2                                     %#ok<ISMAT> grayscale
    img = repmat(img, 1, 1, 3);
end
if size(img, 3) == 4                                   % drop alpha
    img = img(:, :, 1:3);
end
img = im2uint8(img);

% --- preprocess: letterbox resize + ImageNet normalise ----------------- %
x = preprocessFundus(img, INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD);  % HxWx3 single

% NCHW with a batch axis of 1, matching the exported ONNX input "input".
X = reshape(permute(x, [3, 1, 2]), [1, 3, INPUT_SIZE, INPUT_SIZE]);
dlX = dlarray(single(X), "BCSS");

% --- inference -------------------------------------------------------- %
logits = predict(net, dlX);
logits = double(gather(extractdata(logits)));
logits = logits(:).';                                  % 1x5 row

probs = softmaxRow(logits);
[confidence, idx] = max(probs);
grade = idx - 1;                                        % 0-based

pReferable = sum(probs((REFERABLE_DR_MIN_GRADE + 1):end));
referable  = pReferable >= REFERABLE_THRESHOLD;
escalated  = referable && grade < REFERABLE_DR_MIN_GRADE;

% `grade` stays the argmax point estimate; only the referral action escalates
% when the threshold (not the argmax) triggers "referable" -- same rule as
% src/inference/pipeline.py::referable_decision.
actionGrade = grade;
if escalated
    actionGrade = REFERABLE_DR_MIN_GRADE;
end

result = struct( ...
    "grade",              grade, ...
    "label",              GRADE_LABELS(idx), ...
    "confidence",         confidence, ...
    "probabilities",      probs, ...
    "p_referable",        pReferable, ...
    "referable",          referable, ...
    "referral_escalated", escalated, ...
    "referral_action",    REFERRAL_ACTIONS(actionGrade + 1), ...
    "uncertain",          confidence < UNCERTAINTY_THRESHOLD);
end


% ======================================================================= %
function out = preprocessFundus(img, inputSize, meanRGB, stdRGB)
%PREPROCESSFUNDUS  Reproduce the albumentations validation transform from
%   src/data/dataset.py::build_transforms(train=False):
%       LongestMaxSize(inputSize)  -> bilinear, no antialiasing (cv2 INTER_LINEAR)
%       PadIfNeeded(inputSize, inputSize, centre, BORDER_CONSTANT value 0)
%       Normalize(mean, std, max_pixel_value=255)
%   For the 512x512 square APTOS cache images this is just a resize to 224.

[h, w, ~] = size(img);
s  = inputSize / max(h, w);
nh = round(h * s);
nw = round(w * s);

resized = imresize(img, [nh, nw], "bilinear", "Antialiasing", false);

padT = floor((inputSize - nh) / 2);
padL = floor((inputSize - nw) / 2);
canvas = zeros(inputSize, inputSize, 3, "uint8");
canvas((padT + 1):(padT + nh), (padL + 1):(padL + nw), :) = resized;

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
function net = getDefaultNet()
%GETDEFAULTNET  Import ../models/dr_sight.onnx once and cache it.
persistent cachedNet
if ~isempty(cachedNet)
    net = cachedNet;
    return
end

here = fileparts(mfilename("fullpath"));
onnxPath = fullfile(here, "..", "models", "dr_sight.onnx");
if ~isfile(onnxPath)
    error("grade_dr:onnxMissing", ...
        ["Cannot find %s.\nExport it first from the repo root:\n" ...
         "    python scripts/export_onnx.py"], onnxPath);
end

try
    % NCHW input -> "BCSS" (batch, channel, spatial, spatial).
    cachedNet = importNetworkFromONNX(onnxPath, InputDataFormats="BCSS");
catch err
    error("grade_dr:importFailed", ...
        ["importNetworkFromONNX failed (%s).\n" ...
         "Needs the Deep Learning Toolbox Converter for ONNX Model Format."], ...
        err.message);
end

if ~cachedNet.Initialized
    cachedNet = initialize(cachedNet);
end
net = cachedNet;
end
