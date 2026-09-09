function net = load_screening_net(onnxPath)
%LOAD_SCREENING_NET  Import the DR-Sight ONNX model once and cache it (SIH PS26038).
%
%   net = LOAD_SCREENING_NET()
%   net = LOAD_SCREENING_NET(onnxPath)
%
%   importNetworkFromONNX costs several seconds, so a long-lived caller (the
%   FastAPI backend via src/inference/matlab_pipeline.py, or screen_image.m in
%   standalone use) should import the network ONCE and reuse the handle across
%   every request. This helper does that: the first call imports and initialises
%   models/dr_sight.onnx, later calls return the cached dlnetwork for free.
%
%   INPUT
%     onnxPath : (optional) path to the .onnx file. Default: ../models/dr_sight.onnx
%                next to this file. Passing a different path bypasses / refreshes
%                the cache for that path.
%
%   OUTPUT
%     net : an initialised dlnetwork, input data format "BCSS" (batch, channel,
%           spatial, spatial), output = raw logits (softmax is applied downstream
%           in grade_dr.m / explain_gradcam.m).
%
%   Requires the Deep Learning Toolbox + "Deep Learning Toolbox Converter for
%   ONNX Model Format" support package.

arguments
    onnxPath (1,1) string = defaultOnnxPath()
end

persistent CACHE_PATH CACHE_NET
if ~isempty(CACHE_NET) && strcmp(CACHE_PATH, onnxPath)
    net = CACHE_NET;
    return
end

if ~isfile(onnxPath)
    error("load_screening_net:onnxMissing", ...
        "Cannot find %s.\nExport it once from the repo root:\n    python scripts/export_onnx.py", ...
        onnxPath);
end

% importNetworkFromONNX auto-generates custom layers for ONNX ops it can't map
% to built-ins (MobileNetV3's HardSwish / HardSigmoid) and writes them to a
% "+dr_sight" namespace folder in the CURRENT directory. Pin the current
% directory to this file's folder for the call so that (a) it never litters
% whatever directory the MATLAB Engine started in (the repo root, for the
% FastAPI bridge) and (b) the package lands next to screen_image.m, which
% callers already put on the path. The caller's cwd is restored immediately
% after. matlab/+dr_sight is git-ignored - it is regenerated on first import.
here = fileparts(mfilename("fullpath"));
restoreCwd = withCwd(here);   %#ok<NASGU>  onCleanup, fires on any exit path
try
    net = importNetworkFromONNX(onnxPath, ...
        InputDataFormats = "BCSS", Namespace = "dr_sight");
catch err
    error("load_screening_net:importFailed", ...
        "importNetworkFromONNX failed (%s).\nInstall the 'Deep Learning Toolbox Converter for ONNX Model Format' support package (Add-On Explorer).", ...
        err.message);
end
clear restoreCwd            % restore cwd before initialize / addpath / return
addpath(here);              % so the generated +dr_sight namespace resolves
if ~net.Initialized
    net = initialize(net);
end

CACHE_PATH = onnxPath;
CACHE_NET  = net;
end


% ======================================================================= %
function p = defaultOnnxPath()
here = fileparts(mfilename("fullpath"));
p = string(fullfile(here, "..", "models", "dr_sight.onnx"));
end


% ======================================================================= %
function guard = withCwd(targetDir)
%WITHCWD  cd to targetDir and return an onCleanup that cd's back.
origDir = cd(targetDir);
guard = onCleanup(@() cd(origDir));
end
