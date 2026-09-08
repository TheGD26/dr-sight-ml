function result = quality_gate(imageInput, cfg)
%QUALITY_GATE  Image-quality assessment + enhancement for fundus photos (SIH PS26038).
%
%   result = QUALITY_GATE(imagePath)
%   result = QUALITY_GATE(imageMatrix)
%   result = QUALITY_GATE(..., cfg)
%
%   MATLAB port of dr-sight-ml/src/quality/image_quality.py::assess_quality,
%   built on the Image Processing Toolbox. Runs *before* any image reaches the
%   classifier: a plausible-looking heat-map on a blurry or badly-lit photo is
%   worse than returning nothing, so unusable images are rejected up front with
%   a reason the operator can act on.
%
%   Checks (first failing check wins, ordered by how fatal it is):
%     * fov      - the illuminated retinal disc must fill a sane fraction of the
%                  frame (largest bright connected component after Otsu).
%     * exposure - mean brightness must sit inside a band, and not too many
%                  pixels may be clipped to pure black / pure white.
%     * blur     - variance of the Laplacian (sharp images have high variance).
%
%   Adaptive enhancement: when the image PASSES the gate but looks borderline
%   (mild blur, uneven illumination, low contrast, some clipping) an enhanced
%   copy is produced - illumination flattening, CLAHE, light denoise - and
%   returned in result.enhanced_image. Outright rejects are not enhanced.
%
%   INPUTS
%     imageInput : path to an image file, OR an HxWxC uint8/RGB image array.
%     cfg        : (optional) struct overriding any of the threshold fields
%                  below. Missing fields fall back to the defaults, which
%                  mirror dr-sight-ml/src/config.py::QualityConfig:
%                    blur_min_variance  80.0   (env DR_BLUR_MIN_VAR)
%                    exposure_min_mean  25.0   (env DR_EXPOSURE_MIN)
%                    exposure_max_mean  230.0  (env DR_EXPOSURE_MAX)
%                    fov_min_fill       0.20   (env DR_FOV_MIN_FILL)
%                    fov_max_fill       0.95   (env DR_FOV_MAX_FILL)
%                    clip_frac_max      0.35   (env DR_CLIP_FRAC_MAX)
%
%   OUTPUT  result struct with fields:
%     usable         : logical, true when no check failed.
%     reason         : "" when usable, otherwise the operator-facing message
%                      (kept string-for-string comparable with the Python gate).
%     scores         : struct with
%                        blur_var         variance of the Laplacian
%                        mean_brightness  mean of the grayscale image (0-255)
%                        dark_clip_frac   fraction of pixels <= 5
%                        bright_clip_frac fraction of pixels >= 250
%                        fov_fraction     largest bright blob / frame area
%     enhanced       : logical, true when an enhanced copy was produced.
%     enhanced_image : uint8 HxWx3 enhanced image when `enhanced`, else [].
%
%   This is a Smart India Hackathon 2026 prototype, NOT a certified medical
%   device. Screening aid only - confirm with an ophthalmologist.

arguments
    imageInput
    cfg struct = struct()
end

cfg = applyDefaults(cfg);

% --- read / normalise to uint8 RGB ------------------------------------- %
if ischar(imageInput) || isstring(imageInput)
    rgb = imread(char(imageInput));
else
    rgb = imageInput;
end
rgb = toRGBUint8(rgb);
gray = rgb2gray(rgb);

% --- scores ----------------------------------------------------------- %
blurVar   = blurScore(gray);
expo      = exposureScores(gray);
fovFrac   = fovFillFraction(gray);

scores = struct( ...
    "blur_var",         blurVar, ...
    "mean_brightness",  expo.mean_brightness, ...
    "dark_clip_frac",   expo.dark_clip_frac, ...
    "bright_clip_frac", expo.bright_clip_frac, ...
    "fov_fraction",     fovFrac);

% --- decision (first failing check wins, same order as the Python gate) - %
reason = "";
if fovFrac < cfg.fov_min_fill
    reason = sprintf( ...
        "Retina fills only %.0f%% of the frame (need >= %.0f%%). Move closer / re-centre.", ...
        fovFrac * 100, cfg.fov_min_fill * 100);
elseif fovFrac > cfg.fov_max_fill
    reason = "No dark surround visible - image looks cropped or is not a full fundus photo.";
elseif expo.mean_brightness < cfg.exposure_min_mean
    reason = sprintf( ...
        "Underexposed (mean brightness %.0f). Increase illumination.", ...
        expo.mean_brightness);
elseif expo.mean_brightness > cfg.exposure_max_mean
    reason = sprintf( ...
        "Overexposed (mean brightness %.0f). Reduce flash / illumination.", ...
        expo.mean_brightness);
elseif expo.dark_clip_frac > cfg.clip_frac_max
    reason = sprintf( ...
        "%.0f%% of pixels are pure black - under-illuminated or wrong field.", ...
        expo.dark_clip_frac * 100);
elseif expo.bright_clip_frac > cfg.clip_frac_max
    reason = sprintf( ...
        "%.0f%% of pixels are blown out - flash glare or overexposure.", ...
        expo.bright_clip_frac * 100);
elseif blurVar < cfg.blur_min_variance
    reason = sprintf( ...
        "Image too blurry (sharpness %.0f < %.0f). Hold steady / refocus.", ...
        blurVar, cfg.blur_min_variance);
end

reason = string(reason);          % sprintf yields char; normalise to string
usable = strlength(reason) == 0;

% --- adaptive enhancement for borderline-but-usable images ------------ %
enhancedImage = [];
doEnhance = usable && isBorderline(blurVar, expo, fovFrac, cfg);
if doEnhance
    enhancedImage = enhanceFundus(rgb);
end

result = struct( ...
    "usable",         usable, ...
    "reason",         reason, ...
    "scores",         scores, ...
    "enhanced",       ~isempty(enhancedImage), ...
    "enhanced_image", enhancedImage);
end


% ======================================================================= %
function cfg = applyDefaults(cfg)
%APPLYDEFAULTS  Fill missing threshold fields from QualityConfig defaults.
defaults = struct( ...
    "blur_min_variance", 80.0, ...
    "exposure_min_mean", 25.0, ...
    "exposure_max_mean", 230.0, ...
    "fov_min_fill",      0.20, ...
    "fov_max_fill",      0.95, ...
    "clip_frac_max",     0.35);
f = fieldnames(defaults);
for i = 1:numel(f)
    if ~isfield(cfg, f{i}) || isempty(cfg.(f{i}))
        cfg.(f{i}) = defaults.(f{i});
    end
end
end


% ======================================================================= %
function rgb = toRGBUint8(img)
%TORGBUINT8  Coerce any input to an HxWx3 uint8 RGB array (mirrors _as_rgb_array).
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
function v = blurScore(gray)
%BLURSCORE  Variance of the Laplacian. Higher = sharper.
%   fspecial('laplacian', 0) is the 4-neighbour kernel [0 1 0; 1 -4 1; 0 1 0],
%   matching cv2.Laplacian(..., ksize=1). Population variance (normalise by N)
%   to match numpy's ndarray.var().
k   = fspecial("laplacian", 0);
lap = imfilter(double(gray), k, "replicate");
v   = var(lap(:), 1);
end


% ======================================================================= %
function s = exposureScores(gray)
%EXPOSURESCORES  Mean brightness + clipped-pixel fractions (mirrors exposure_scores).
g = double(gray);
s = struct( ...
    "mean_brightness",  mean(g(:)), ...
    "dark_clip_frac",   mean(gray(:) <= 5), ...
    "bright_clip_frac", mean(gray(:) >= 250));
end


% ======================================================================= %
function areaFrac = fovFillFraction(gray)
%FOVFILLFRACTION  Fraction of the frame the illuminated retina covers.
%   Gaussian-blur, Otsu-threshold, keep the largest bright connected
%   component, compare its area to the frame. cv2.GaussianBlur with a 7x7
%   kernel uses sigma = 0.3*((7-1)*0.5 - 1) + 0.8 = 1.4.
[h, w] = size(gray);
blurred = imgaussfilt(gray, 1.4, "FilterSize", 7);

level = graythresh(blurred);            % Otsu, normalised 0..1
mask  = imbinarize(blurred, level);

if ~any(mask(:))
    areaFrac = 0.0;
    return;
end

mask = bwareafilt(mask, 1);             % largest connected component only
areaFrac = nnz(mask) / (h * w);
end


% ======================================================================= %
function tf = isBorderline(blurVar, expo, fovFrac, cfg)
%ISBORDERLINE  Usable, but close enough to a failure that enhancement helps.
%   Any one of: sharpness within 2x of the blur floor, mean brightness in the
%   outer thirds of the valid band, a noticeable clipped-pixel fraction, or a
%   tight field of view.
nearBlur = blurVar < 2 * cfg.blur_min_variance;
dimOrHot = expo.mean_brightness < 60 || expo.mean_brightness > 200;
someClip = expo.dark_clip_frac > 0.15 || expo.bright_clip_frac > 0.15;
tightFov = fovFrac < cfg.fov_min_fill * 1.5;
tf = nearBlur || dimOrHot || someClip || tightFov;
end


% ======================================================================= %
function out = enhanceFundus(rgb)
%ENHANCEFUNDUS  Illumination flattening -> CLAHE on L* -> light denoise.
%   Every step degrades gracefully if an optional function is unavailable.
img = im2double(rgb);
[h, w, ~] = size(img);

% --- 1. illumination normalisation --------------------------------------
% Prefer imflatfield (Retinex-style flat-field correction); fall back to
% dividing by a heavily low-pass version of the image.
sigma = max(15, round(min(h, w) / 6));
if exist("imflatfield", "file") == 2
    flat = im2double(imflatfield(im2uint8(img), sigma));
else
    lowpass = imgaussfilt(img, sigma);
    lowpass = max(lowpass, 1e-3);
    flat = img ./ lowpass;
    flat = flat .* mean(lowpass(:));          % restore a sane overall level
    flat = min(max(flat, 0), 1);
end

% --- 2. CLAHE on the L* channel ---------------------------------------
lab = rgb2lab(flat);
L   = lab(:, :, 1) / 100;                     % adapthisteq wants [0,1]
L   = adapthisteq(L, "ClipLimit", 0.01, "Distribution", "rayleigh", ...
                  "NumTiles", [8 8]);
lab(:, :, 1) = L * 100;
claheRGB = lab2rgb(lab);
claheRGB = min(max(claheRGB, 0), 1);

% --- 3. light denoise -----------------------------------------------
% Edge-preserving if available, otherwise a small Gaussian.
if exist("imbilatfilt", "file") == 2
    den = imbilatfilt(claheRGB, 0.02, 3);
else
    den = imgaussfilt(claheRGB, 0.5);
end

out = im2uint8(min(max(den, 0), 1));
end
