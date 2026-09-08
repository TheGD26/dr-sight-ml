function result = segment_optic_disc(imageInput, opts)
%SEGMENT_OPTIC_DISC  Locate the optic disc in a fundus photo (SIH PS26038).
%
%   result = SEGMENT_OPTIC_DISC(imagePath)
%   result = SEGMENT_OPTIC_DISC(imageMatrix)
%   result = SEGMENT_OPTIC_DISC(..., Name = Value)
%
%   The optic disc is normally the largest, brightest, roughly circular
%   high-contrast structure in a retinal image. This function finds it with a
%   classical Image Processing Toolbox pipeline (no training data required):
%
%     1. build a field-of-view (FOV) mask so the black camera surround is
%        never mistaken for signal;
%     2. take the green (default) or red channel -- both show the disc as a
%        bright blob, green keeps the best overall fundus contrast;
%     3. morphologically close the channel to swallow the dark vessels that
%        cross the disc, then apply a strong Gaussian blur so the disc becomes
%        one smooth bright hill;
%     4. threshold at a top percentile of the in-FOV intensities and take the
%        largest connected component via REGIONPROPS;
%     5. report that blob as a circular ROI (centroid + radius) and draw it.
%
%   This is a geometry/brightness heuristic, not a learned segmenter: a large
%   bright lesion (e.g. a big exudate patch) or heavy glare can pull the
%   estimate off the true disc. Use `result.circularity` and
%   `result.area_fraction` to sanity-check the hit.
%
%   NAME-VALUE OPTIONS
%     Channel         "green" (default) | "red"
%     Percentile      brightness percentile for the threshold (default 99)
%     BlurSigma       Gaussian sigma in px (default: min(H,W)/28, auto)
%     SuppressVessels close the channel to remove vessels first (default true)
%     MinCircularity  below this, `found` is set false (default 0.35)
%     Show            pop up a diagnostic figure (default false)
%
%   OUTPUT  result struct with fields:
%     found          logical -- a plausible disc was located
%     centroid       [x y] disc centre in pixels ([NaN NaN] if not found)
%     radius         disc radius in pixels (NaN if not found)
%     bbox           [x y w h] bounding box of the detected blob
%     circularity    4*pi*Area / Perimeter^2  (1.0 == perfect circle)
%     area_fraction  disc area as a fraction of the FOV area
%     channel        channel actually used
%     mask           logical HxW mask of the detected disc blob
%     fov_mask       logical HxW field-of-view mask
%     overlay        RGB image with the disc ROI drawn on it
%     crop           RGB close-up crop around the disc (3x radius box)
%
%   Smart India Hackathon 2026 prototype -- screening aid only, NOT a
%   certified medical device.

arguments
    imageInput
    opts.Channel (1,1) string {mustBeMember(opts.Channel, ["green", "red"])} = "green"
    opts.Percentile (1,1) double {mustBeInRange(opts.Percentile, 50, 99.99)} = 99
    opts.BlurSigma double {mustBeScalarOrEmpty} = []
    opts.SuppressVessels (1,1) logical = true
    opts.MinCircularity (1,1) double {mustBeInRange(opts.MinCircularity, 0, 1)} = 0.35
    opts.Show (1,1) logical = false
end

rgb = readRgb(imageInput);
[H, W, ~] = size(rgb);
scale = min(H, W);

% --- 1. field-of-view mask ------------------------------------------------ %
fovMask = fieldOfViewMask(rgb);
fovArea = nnz(fovMask);
if fovArea == 0
    fovMask = true(H, W);       % degenerate: treat whole frame as FOV
    fovArea = H * W;
end

% --- 2. channel --------------------------------------------------------- %
if opts.Channel == "green"
    chan = rgb(:, :, 2);
else
    chan = rgb(:, :, 1);
end
chan = im2double(chan);

% --- 3. suppress vessels, then blur to a single bright hill ------------- %
work = chan;
if opts.SuppressVessels
    seClose = strel("disk", max(3, round(scale / 60)));
    work = imclose(work, seClose);         % dark vessels get filled in
end

sigma = opts.BlurSigma;
if isempty(sigma)
    sigma = max(3, scale / 28);
end
blur = imgaussfilt(work, sigma);

% Only consider in-FOV pixels: the threshold is driven purely by real retina.
thresh = prctile(blur(fovMask), opts.Percentile);

bright = (blur >= thresh) & fovMask;
bright = imfill(bright, "holes");
bright = imopen(bright, strel("disk", max(2, round(scale / 120))));

% --- 4. largest bright blob = optic disc candidate --------------------- %
cc = bwconncomp(bright);
result = emptyResult(opts.Channel, fovMask, rgb);
if cc.NumObjects == 0
    if opts.Show
        showDiagnostics(rgb, chan, blur, bright, result);
    end
    return
end

stats = regionprops(cc, "Area", "Centroid", "BoundingBox", ...
    "EquivDiameter", "Perimeter", "MajorAxisLength", "MinorAxisLength");
[~, kBest] = max([stats.Area]);
best = stats(kBest);

discMask = false(H, W);
discMask(cc.PixelIdxList{kBest}) = true;

centroid = best.Centroid;                       % [x y]
% Radius: average the equivalent-area radius and the axis-based radius so a
% slightly elliptical blob still gives a sensible circle.
rArea = best.EquivDiameter / 2;
rAxis = (best.MajorAxisLength + best.MinorAxisLength) / 4;
radius = mean([rArea, rAxis]);

if best.Perimeter > 0
    circularity = 4 * pi * best.Area / best.Perimeter^2;
else
    circularity = 0;
end
circularity = min(circularity, 1);             % discretisation can overshoot
areaFraction = best.Area / fovArea;

found = circularity >= opts.MinCircularity && areaFraction < 0.25;

% --- 5. draw + crop --------------------------------------------------- %
overlay = im2uint8(rgb);
overlay = insertShape(overlay, "circle", [centroid, radius], ...
    "Color", "yellow", "LineWidth", max(2, round(scale / 200)));
overlay = insertMarker(overlay, centroid, "x", "Color", "yellow", "Size", 10);
if ~found
    overlay = insertText(overlay, [10, 10], "low-confidence disc hit", ...
        "TextColor", "red", "BoxOpacity", 0.4);
end

crop = cropAround(im2uint8(rgb), centroid, radius, 3.0);

result.found = found;
result.centroid = centroid;
result.radius = radius;
result.bbox = best.BoundingBox;
result.circularity = circularity;
result.area_fraction = areaFraction;
result.mask = discMask;
result.overlay = overlay;
result.crop = crop;

if opts.Show
    showDiagnostics(rgb, chan, blur, bright, result);
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
lvl = max(lvl, 10 / 255);                   % guard against all-dark inputs
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
% Pull the border in a little -- the bright FOV rim is a ridge artefact.
mask = imerode(mask, strel("disk", max(3, round(min(size(gray)) / 100))));
end


% ===================================================================== %
function crop = cropAround(rgb, centroid, radius, factor)
[H, W, ~] = size(rgb);
half = max(round(radius * factor), 10);
cx = round(centroid(1));
cy = round(centroid(2));
x1 = max(1, cx - half); x2 = min(W, cx + half);
y1 = max(1, cy - half); y2 = min(H, cy + half);
crop = rgb(y1:y2, x1:x2, :);
end


% ===================================================================== %
function result = emptyResult(channel, fovMask, rgb)
result = struct( ...
    "found",         false, ...
    "centroid",      [NaN, NaN], ...
    "radius",        NaN, ...
    "bbox",          [NaN, NaN, NaN, NaN], ...
    "circularity",   NaN, ...
    "area_fraction", NaN, ...
    "channel",       channel, ...
    "mask",          false(size(fovMask)), ...
    "fov_mask",      fovMask, ...
    "overlay",       im2uint8(rgb), ...
    "crop",          im2uint8(rgb));
end


% ===================================================================== %
function showDiagnostics(rgb, chan, blur, bright, result)
figure("Name", "segment_optic_disc", "Color", "w");
tiledlayout(2, 3, "TileSpacing", "compact", "Padding", "compact");

nexttile; imshow(rgb);            title("input");
nexttile; imshow(chan, []);       title(result.channel + " channel");
nexttile; imshow(blur, []);       title("vessel-suppressed + blurred");
nexttile; imshow(bright);         title("top-percentile blob");
nexttile; imshow(result.overlay); title("disc ROI");
nexttile; imshow(result.crop);    title("crop");

if result.found
    sgtitle(sprintf("disc @ (%.0f, %.0f)  r = %.0f px  circularity %.2f", ...
        result.centroid(1), result.centroid(2), result.radius, result.circularity));
else
    sgtitle("optic disc not confidently located");
end
end
