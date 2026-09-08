function result = segment_vessels(imageInput, opts)
%SEGMENT_VESSELS  Segment the retinal vasculature in a fundus photo (SIH PS26038).
%
%   result = SEGMENT_VESSELS(imagePath)
%   result = SEGMENT_VESSELS(imageMatrix)
%   result = SEGMENT_VESSELS(..., Name = Value)
%
%   Classical, training-free vessel segmentation with the Image Processing
%   Toolbox:
%
%     1. field-of-view (FOV) mask, eroded so the bright FOV rim is excluded;
%     2. green channel + CLAHE (ADAPTHISTEQ) + top-hat background flattening
%        to even out illumination and lift thin vessels;
%     3. ridge enhancement with FIBERMETRIC (a multi-scale Frangi-type
%        vesselness filter). If FIBERMETRIC is unavailable the code falls
%        back to a rotating Gaussian matched filter (MATCHEDFILTERBANK below);
%     4. threshold (IMBINARIZE, adaptive or Otsu) and clean small specks;
%     5. thin to a 1-px centreline skeleton with BWSKEL.
%
%   Vessel density is reported as the fraction of in-FOV pixels labelled
%   vessel -- a coarse global marker that rises with neovascularisation and
%   falls with capillary drop-out, but is NOT a validated clinical measure
%   here.
%
%   NAME-VALUE OPTIONS
%     ThicknessRange  vessel widths in px for FIBERMETRIC (default 1:7,
%                     auto-scaled by image size)
%     Method          "fibermetric" (default) | "matchedfilter"
%     Threshold       "adaptive" (default) | "otsu"
%     Sensitivity     IMBINARIZE adaptive sensitivity (default 0.55)
%     MinObjectPx     drop vessel blobs smaller than this (default: scale/12)
%     Show            pop up a diagnostic figure (default false)
%
%   OUTPUT  result struct with fields:
%     mask            logical HxW vessel mask
%     skeleton        logical HxW 1-px centreline (BWSKEL)
%     vessel_density  nnz(mask) / nnz(fov_mask)
%     skeleton_length total centreline length in px (nnz(skeleton))
%     mean_width_px   mask area / skeleton length (rough average calibre)
%     method          enhancement method actually used
%     fov_mask        logical HxW field-of-view mask
%     enhanced        double HxW vesselness response (0..1)
%     overlay         RGB input with the vessel mask painted green
%
%   Smart India Hackathon 2026 prototype -- screening aid only, NOT a
%   certified medical device.

arguments
    imageInput
    opts.ThicknessRange {mustBeVector, mustBePositive} = 1:7
    opts.Method (1,1) string {mustBeMember(opts.Method, ["fibermetric", "matchedfilter"])} = "fibermetric"
    opts.Threshold (1,1) string {mustBeMember(opts.Threshold, ["adaptive", "otsu"])} = "adaptive"
    opts.Sensitivity (1,1) double {mustBeInRange(opts.Sensitivity, 0, 1)} = 0.55
    opts.MinObjectPx double {mustBeScalarOrEmpty} = []
    opts.Show (1,1) logical = false
end

rgb = readRgb(imageInput);
[H, W, ~] = size(rgb);
scale = min(H, W);

% --- 1. field-of-view mask --------------------------------------------- %
fovMask = fieldOfViewMask(rgb);
if nnz(fovMask) == 0
    fovMask = true(H, W);
end
% Extra erosion: keep the analysis well inside the aperture.
fovInner = imerode(fovMask, strel("disk", max(4, round(scale / 60))));
if nnz(fovInner) == 0
    fovInner = fovMask;
end

% --- 2. green channel, illumination-flattened, contrast-boosted -------- %
g = im2double(rgb(:, :, 2));
lims = stretchlim(g(fovInner), [0.01 0.99]);
if lims(2) <= lims(1)
    lims = [0; 1];
end
g = imadjust(g, lims, []);
gEq = adapthisteq(g, "NumTiles", [8 8], "ClipLimit", 0.01);

% Top-hat on the complement pulls out the dark tubular vessels while
% removing slow background drift (optic disc, uneven flash).
seBG = strel("disk", max(6, round(scale / 30)));
vesselsUp = imtophat(imcomplement(gEq), seBG);
vesselsUp = mat2gray(vesselsUp);

% --- 3. ridge / vesselness enhancement -------------------------------- %
tRange = unique(round(opts.ThicknessRange * scale / 512));
tRange = tRange(tRange >= 1);
if isempty(tRange)
    tRange = 1:7;
end

method = opts.Method;
enhanced = [];
if method == "fibermetric"
    try
        % Vessels are already bright in `vesselsUp`, so ObjectPolarity 'bright'.
        enhanced = fibermetric(vesselsUp, tRange, ...
            "ObjectPolarity", "bright", "StructureSensitivity", 0.05);
    catch
        method = "matchedfilter";      % toolbox/function not available
    end
end
if method == "matchedfilter"
    enhanced = matchedFilterBank(vesselsUp, tRange);
end
enhanced = mat2gray(enhanced);
enhanced(~fovInner) = 0;

% --- 4. threshold + clean ------------------------------------------- %
if opts.Threshold == "adaptive"
    bw = imbinarize(enhanced, "adaptive", ...
        "ForegroundPolarity", "bright", "Sensitivity", opts.Sensitivity);
else
    lvl = graythresh(enhanced(fovInner));
    bw = enhanced >= lvl;
end
bw = bw & fovInner;

minPx = opts.MinObjectPx;
if isempty(minPx)
    minPx = max(15, round(scale / 12));
end
bw = bwareaopen(bw, round(minPx));
bw = bwmorph(bw, "spur", 3);
bw = bwareaopen(bw, round(minPx / 2));

% --- 5. skeletonise ----------------------------------------------- %
skel = bwskel(bw, "MinBranchLength", max(4, round(scale / 100)));

% --- metrics ---------------------------------------------------- %
fovCount = nnz(fovInner);
vesselDensity = nnz(bw) / fovCount;
skelLen = nnz(skel);
if skelLen > 0
    meanWidth = nnz(bw) / skelLen;
else
    meanWidth = 0;
end

% --- overlay -------------------------------------------------- %
overlay = im2uint8(rgb);
overlay = imoverlay(overlay, bw, "green");
overlay = imoverlay(overlay, skel, "red");

result = struct( ...
    "mask",            bw, ...
    "skeleton",        skel, ...
    "vessel_density",  vesselDensity, ...
    "skeleton_length", skelLen, ...
    "mean_width_px",   meanWidth, ...
    "method",          method, ...
    "fov_mask",        fovMask, ...
    "enhanced",        enhanced, ...
    "overlay",         overlay);

if opts.Show
    showDiagnostics(rgb, gEq, enhanced, bw, skel, result);
end
end


% ===================================================================== %
function rgb = readRgb(imageInput)
if ischar(imageInput) || isstring(imageInput)
    rgb = imread(char(imageInput));
else
    rgb = imageInput;
end
if ndims(rgb) == 2                          %#ok<ISMAT> grayscale
    rgb = repmat(rgb, 1, 1, 3);
end
if size(rgb, 3) == 4
    rgb = rgb(:, :, 1:3);
end
rgb = im2uint8(rgb);
end


% ===================================================================== %
function mask = fieldOfViewMask(rgb)
%FIELDOFVIEWMASK  The illuminated retinal circle, black surround removed.
gray = rgb2gray(rgb);
lvl = graythresh(gray);
lvl = max(lvl, 10 / 255);
mask = imbinarize(gray, lvl);
mask = imfill(mask, "holes");
mask = imopen(mask, strel("disk", 5));
cc = bwconncomp(mask);
if cc.NumObjects == 0
    mask = false(size(gray));
    return
end
numPix = cellfun(@numel, cc.PixelIdxList);
[~, k] = max(numPix);
mask = false(size(gray));
mask(cc.PixelIdxList{k}) = true;
end


% ===================================================================== %
function response = matchedFilterBank(img, widths)
%MATCHEDFILTERBANK  Fallback vessel enhancer: a Gaussian-profile line kernel
%   rotated through 0..165 deg, keeping the per-pixel maximum response.
%   Approximates Chaudhuri et al. (1989) matched filtering.
angles = 0:15:165;
response = zeros(size(img));
for wIdx = 1:numel(widths)
    sigma = max(1, widths(wIdx) / 2);
    L = max(9, 2 * ceil(3 * sigma) + 1);      % kernel length along the vessel
    half = floor(L / 2);
    [xx, yy] = meshgrid(-half:half, -half:half);
    prof = -exp(-(xx.^2) / (2 * sigma^2));    % dark-line profile
    prof = prof .* (abs(yy) <= ceil(1.5 * sigma));
    prof = prof - mean(prof(:));              % zero-mean -> flat bg cancels
    for a = angles
        k = imrotate(prof, a, "bilinear", "crop");
        k = k - mean(k(:));
        response = max(response, imfilter(img, k, "replicate", "same"));
    end
end
response = max(response, 0);
end


% ===================================================================== %
function showDiagnostics(rgb, gEq, enhanced, bw, skel, result)
figure("Name", "segment_vessels", "Color", "w");
tiledlayout(2, 3, "TileSpacing", "compact", "Padding", "compact");

nexttile; imshow(rgb);            title("input");
nexttile; imshow(gEq, []);        title("green + CLAHE");
nexttile; imshow(enhanced, []);   title(result.method + " vesselness");
nexttile; imshow(bw);             title("vessel mask");
nexttile; imshow(skel);           title("skeleton (bwskel)");
nexttile; imshow(result.overlay); title("overlay");

sgtitle(sprintf("vessel density %.1f%%   centreline %d px   mean width %.1f px", ...
    100 * result.vessel_density, result.skeleton_length, result.mean_width_px));
end
