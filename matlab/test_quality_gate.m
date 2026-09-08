%% test_quality_gate - exercise quality_gate.m over sample fundus images
% SIH PS26038, Team Diasight. Runs the image-quality gate over a handful of
% real APTOS cache images plus a few deliberately degraded variants (blur,
% under-exposure, tiny field of view), prints a summary table, and for every
% image the gate chose to enhance saves a before/after figure under
%   matlab/output/quality_samples/
%
% Run from MATLAB:  >> test_quality_gate
%
% Requires: Image Processing Toolbox.

%% Locate the repo and wire up paths
thisDir = fileparts(mfilename("fullpath"));
if isempty(thisDir); thisDir = pwd; end
addpath(thisDir);
repoRoot = fullfile(thisDir, "..");

outDir = fullfile(thisDir, "output", "quality_samples");
if ~exist(outDir, "dir"); mkdir(outDir); end

%% Collect sample images
candidates = strings(0, 1);
cacheDir = fullfile(repoRoot, "data", "aptos", "cache");
gcamDir  = fullfile(repoRoot, "tests", "gradcam_samples");

listing = dir(fullfile(cacheDir, "*.png"));
for i = 1:min(4, numel(listing))
    candidates(end + 1, 1) = string(fullfile(listing(i).folder, listing(i).name)); %#ok<SAGROW>
end
listing = dir(fullfile(gcamDir, "*.png"));
for i = 1:min(2, numel(listing))
    candidates(end + 1, 1) = string(fullfile(listing(i).folder, listing(i).name)); %#ok<SAGROW>
end

assert(~isempty(candidates), ...
    "No sample images found under %s or %s", cacheDir, gcamDir);

%% Build the work list: (name, image) pairs
names  = strings(0, 1);
images = {};

for i = 1:numel(candidates)
    [~, base, ~] = fileparts(candidates(i));
    names(end + 1, 1)  = base;                 %#ok<SAGROW>
    images{end + 1, 1} = imread(char(candidates(i))); %#ok<SAGROW>
end

% Degraded variants from the first clean image, to exercise the reject and
% enhancement paths deterministically.
clean = im2uint8(images{1});
names(end + 1, 1)  = names(1) + "__blurred";
images{end + 1, 1} = imgaussfilt(clean, 6);           %#ok<SAGROW> too soft
names(end + 1, 1)  = names(1) + "__dark";
images{end + 1, 1} = im2uint8(im2double(clean) * 0.18); %#ok<SAGROW> underexposed
names(end + 1, 1)  = names(1) + "__lowcontrast";
lc = im2double(clean); lc = 0.45 + (lc - mean(lc(:))) * 0.25;
images{end + 1, 1} = im2uint8(min(max(lc, 0), 1));    %#ok<SAGROW> borderline
names(end + 1, 1)  = names(1) + "__tinyfov";
tiny = zeros(size(clean), "uint8");
[h, w, ~] = size(clean);
cy = round(h / 2); cx = round(w / 2); r = round(min(h, w) * 0.12);
[X, Y] = meshgrid(1:w, 1:h);
disc = ((X - cx).^2 + (Y - cy).^2) <= r^2;
tiny(repmat(disc, 1, 1, 3)) = clean(repmat(disc, 1, 1, 3));
images{end + 1, 1} = tiny;                            %#ok<SAGROW> off-frame retina

%% Run the gate and report
fprintf("\n%-28s  %-6s  %-8s  %6s  %6s  %6s  %6s  %-8s\n", ...
    "image", "usable", "enhanced", "blur", "mean", "darkF", "fovF", "");
fprintf("%s\n", repmat('-', 1, 96));

results = cell(numel(images), 1);
for i = 1:numel(images)
    r = quality_gate(images{i});
    results{i} = r;
    s = r.scores;
    fprintf("%-28s  %-6s  %-8s  %6.0f  %6.1f  %6.2f  %6.2f  %s\n", ...
        names(i), string(r.usable), string(r.enhanced), ...
        s.blur_var, s.mean_brightness, s.dark_clip_frac, s.fov_fraction, ...
        r.reason);
end
fprintf("\n");

%% Save before/after figures for every enhanced image
nSaved = 0;
for i = 1:numel(images)
    r = results{i};
    if ~r.enhanced; continue; end

    fig = figure("Visible", "off", "Position", [100 100 1100 560]);

    subplot(1, 2, 1);
    imshow(im2uint8(images{i}));
    title(sprintf("original  -  blur %.0f  mean %.0f", ...
        r.scores.blur_var, r.scores.mean_brightness), "Interpreter", "none");

    subplot(1, 2, 2);
    imshow(r.enhanced_image);
    title("enhanced  -  flat-field + CLAHE + denoise", "Interpreter", "none");

    sgtitle(sprintf("%s   (usable = %s)", names(i), string(r.usable)), ...
        "Interpreter", "none");

    outPath = fullfile(outDir, names(i) + "_beforeafter.png");
    exportgraphics(fig, outPath, "Resolution", 150);
    close(fig);
    nSaved = nSaved + 1;
    fprintf("saved %s\n", outPath);
end

%% Contact-sheet montage of every original with its verdict
fig = figure("Visible", "off", "Position", [100 100 1200 800]);
montage(cellfun(@im2uint8, images, "UniformOutput", false), "Size", [NaN 4]);
title(sprintf("quality_gate verdicts  (%d images, %d enhanced, %d saved)", ...
    numel(images), sum(cellfun(@(r) r.enhanced, results)), nSaved), ...
    "Interpreter", "none");
montagePath = fullfile(outDir, "contact_sheet.png");
exportgraphics(fig, montagePath, "Resolution", 150);
close(fig);
fprintf("saved %s\n\n", montagePath);

fprintf("done: %d images assessed, %d before/after figures under %s\n", ...
    numel(images), nSaved, outDir);
