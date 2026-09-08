%% DR-Sight - retinal structure segmentation demo (SIH PS26038, Team Diasight)
% Two classical, training-free segmentation components for fundus images:
%
%   * segment_optic_disc.m  - optic disc localisation (centroid + radius)
%   * segment_vessels.m     - retinal vessel mask + vessel density
%
% Both use only the Image Processing Toolbox (fibermetric, adapthisteq,
% regionprops, imbinarize, bwskel, ...). Run this file section-by-section
% (Ctrl+Enter) or save it as a Live Script (Save As -> SegmentationDemo.mlx)
% for the richer inline-figure view.
%
% SCOPE NOTE - PS26038 lists SIX retinal analysis tasks. This prototype
% implements TWO of them properly (optic disc, vessels). The remaining four
% (microaneurysms, exudates, haemorrhages, neovascularisation) are documented
% as future work at the bottom of this script and are NOT faked.
%
% NOT a certified medical device. Screening aid only - confirm with an
% ophthalmologist.

%% 1. Locate the repo and pick a sample fundus image
thisDir = fileparts(matlab.desktop.editor.getActiveFilename);
if isempty(thisDir); thisDir = pwd; end
repoRoot = fullfile(thisDir, "..");
addpath(thisDir);

sampleDir = fullfile(repoRoot, "data", "aptos", "cache");
listing = dir(fullfile(sampleDir, "*.png"));
assert(~isempty(listing), "No sample images under %s", sampleDir);

% Swap in your own fundus image path here if you like.
imgPath = fullfile(listing(1).folder, listing(1).name);
rgb = imread(imgPath);
fprintf("Sample image: %s  (%d x %d)\n", imgPath, size(rgb, 2), size(rgb, 1));

figure("Name", "input", "Color", "w");
tiledlayout(1, 4, "TileSpacing", "compact", "Padding", "compact");
nexttile; imshow(rgb);            title("RGB");
nexttile; imshow(rgb(:, :, 1));   title("red channel");
nexttile; imshow(rgb(:, :, 2));   title("green channel");
nexttile; imshow(rgb(:, :, 3));   title("blue channel");

%% 2. Optic disc localisation - before / after
discGreen = segment_optic_disc(imgPath, Channel = "green", Show = true);

fprintf("\n--- optic disc (green channel) ---\n");
if discGreen.found
    fprintf("centroid      : (%.1f, %.1f) px\n", discGreen.centroid);
    fprintf("radius        : %.1f px\n", discGreen.radius);
    fprintf("circularity   : %.2f  (1.0 = perfect circle)\n", discGreen.circularity);
    fprintf("area fraction : %.3f of FOV\n", discGreen.area_fraction);
else
    fprintf("disc not confidently located (circularity %.2f) - see overlay\n", ...
        discGreen.circularity);
end

% Red channel often shows the disc with even higher contrast - compare.
discRed = segment_optic_disc(imgPath, Channel = "red");

figure("Name", "optic disc - before / after", "Color", "w");
tiledlayout(1, 3, "TileSpacing", "compact", "Padding", "compact");
nexttile; imshow(rgb);               title("before");
nexttile; imshow(discGreen.overlay); title("after - green-channel ROI");
nexttile; imshow(discRed.overlay);   title("after - red-channel ROI");

%% 3. Vessel segmentation - before / after
ves = segment_vessels(imgPath, Show = true);

fprintf("\n--- retinal vessels ---\n");
fprintf("method          : %s\n", ves.method);
fprintf("vessel density  : %.2f%% of in-FOV pixels\n", 100 * ves.vessel_density);
fprintf("centreline length: %d px\n", ves.skeleton_length);
fprintf("mean calibre    : %.1f px\n", ves.mean_width_px);

figure("Name", "vessels - before / after", "Color", "w");
tiledlayout(2, 2, "TileSpacing", "compact", "Padding", "compact");
nexttile; imshow(rgb);          title("before");
nexttile; imshow(ves.enhanced); title("vesselness response");
nexttile; imshow(ves.mask);     title("after - binary vessel mask");
nexttile; imshow(ves.overlay);  title("overlay (green mask, red skeleton)");

%% 4. Combined view
combined = discGreen.overlay;
combined = imoverlay(combined, ves.skeleton, "cyan");
figure("Name", "combined", "Color", "w");
imshow(combined);
title(sprintf("optic disc r=%.0f px  |  vessel density %.1f%%", ...
    discGreen.radius, 100 * ves.vessel_density), "Interpreter", "none");

%% 5. Batch a few images (sanity check the pipeline is stable)
files = listing(1:min(6, numel(listing)));
names = strings(numel(files), 1);
discFound = false(numel(files), 1);
discR = zeros(numel(files), 1);
vDens = zeros(numel(files), 1);
for k = 1:numel(files)
    p = fullfile(files(k).folder, files(k).name);
    d = segment_optic_disc(p);
    v = segment_vessels(p);
    names(k) = string(files(k).name);
    discFound(k) = d.found;
    discR(k) = d.radius;
    vDens(k) = v.vessel_density;
end
batchTable = table(names, discFound, discR, vDens, ...
    VariableNames = ["image", "disc_found", "disc_radius_px", "vessel_density"]);
disp(batchTable);

%% 6. Scope statement - what this prototype does NOT do
% ---------------------------------------------------------------------------
% PS26038 - RETINAL ANALYSIS TASKS: IMPLEMENTATION STATUS
% ---------------------------------------------------------------------------
%   [DONE]  Optic disc / fovea localisation
%             -> segment_optic_disc.m (disc centroid + radius; fovea can be
%                derived as ~2.5 disc-diameters temporal to the disc centre)
%   [DONE]  Retinal vessel segmentation
%             -> segment_vessels.m (fibermetric ridge filter + bwskel)
%
%   [NOT IMPLEMENTED IN THIS PROTOTYPE - DOCUMENTED AS FUTURE WORK]
%     - Microaneurysm detection
%     - Exudate segmentation
%     - Haemorrhage classification
%     - Neovascularisation detection
%
%   These four are per-lesion tasks with small, low-contrast, highly variable
%   targets. Doing them at clinical quality requires per-lesion annotated
%   training data (e.g. the IDRiD segmentation masks, or e-ophtha / DIARETDB1)
%   and dedicated CNN / U-Net segmentation models - not the classical
%   morphology pipeline used above. They are intentionally left unimplemented
%   here rather than approximated, so results are never faked. See the project
%   roadmap for the planned annotated-data + U-Net track.
% ---------------------------------------------------------------------------

fprintf("\n==================================================================\n");
fprintf(" PS26038 retinal analysis - implementation status\n");
fprintf("==================================================================\n");
fprintf("  [DONE] Optic disc / fovea localisation   -> segment_optic_disc.m\n");
fprintf("  [DONE] Retinal vessel segmentation       -> segment_vessels.m\n");
fprintf("\n");
fprintf("  NOT implemented in this prototype - documented as future work\n");
fprintf("  requiring per-lesion annotated training data (e.g. IDRiD\n");
fprintf("  segmentation masks) and dedicated CNN/U-Net models:\n");
fprintf("    - Microaneurysm detection\n");
fprintf("    - Exudate segmentation\n");
fprintf("    - Haemorrhage classification\n");
fprintf("    - Neovascularisation detection\n");
fprintf("==================================================================\n");
