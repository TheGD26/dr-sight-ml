function summary = validate_grading(referenceCsv, opts)
%VALIDATE_GRADING  Sanity-check the ONNX-imported DR-Sight model against PyTorch.
%
%   validate_grading()
%   validate_grading(referenceCsv)
%   summary = validate_grading(referenceCsv, imageDir=..., probTol=..., ...
%                              gradeAgreeMin=..., verbose=...)
%
%   Runs grade_dr.m over every image listed in a reference CSV produced by
%       python scripts/make_reference_csv.py
%   and compares the MATLAB/ONNX output against the original PyTorch
%   predictions in that CSV. Small floating-point differences are expected
%   (OpenCV vs imresize, ONNX Runtime vs libtorch); a grade flip or a large
%   probability gap is not.
%
%   INPUTS
%     referenceCsv : path to the reference CSV. Default: reference_test.csv
%                    next to this file. Required columns:
%                      id_code, abs_path, true_grade,
%                      pt_grade, pt_p_referable, pt_p0..pt_p4
%     opts.imageDir : if set, images are looked up as fullfile(imageDir,
%                     id_code + ".png") instead of using the abs_path column
%                     (useful when the CSV was generated on another machine).
%     opts.probTol       : max allowed |Δ| on any softmax probability. Default 0.02.
%     opts.gradeAgreeMin : min fraction of images whose argmax grade must match.
%                          Default 0.98.
%     opts.verbose       : print a row per image (default false; mismatches are
%                          always printed).
%
%   OUTPUT  summary struct: n, grade_agree, max_prob_diff, mean_prob_diff,
%     max_p_referable_diff, referable_agree, passed (logical), mismatches (table).
%
%   A non-zero exit is signalled via error() when passed == false, so this can
%   gate a CI step.

arguments
    referenceCsv (1,1) string = defaultCsv()
    opts.imageDir (1,1) string = ""
    opts.probTol (1,1) double = 0.02
    opts.gradeAgreeMin (1,1) double = 0.98
    opts.verbose (1,1) logical = false
end

if ~isfile(referenceCsv)
    error("validate_grading:csvMissing", ...
        ["No reference CSV at %s.\nGenerate it from the repo root:\n" ...
         "    python scripts/make_reference_csv.py --split test"], referenceCsv);
end

ref = readtable(referenceCsv, TextType="string");
n = height(ref);
fprintf("Loaded %d reference rows from %s\n", n, referenceCsv);

here = fileparts(mfilename("fullpath"));
onnxPath = fullfile(here, "..", "models", "dr_sight.onnx");
net = importNetworkFromONNX(onnxPath, InputDataFormats="BCSS");
if ~net.Initialized
    net = initialize(net);
end

ptProbCols = "pt_p" + string(0:4);

mlGrade      = zeros(n, 1);
mlPref       = zeros(n, 1);
maxProbDiff  = zeros(n, 1);
prefDiff     = zeros(n, 1);
gradeMatch   = false(n, 1);

for i = 1:n
    if opts.imageDir ~= ""
        imgPath = fullfile(opts.imageDir, ref.id_code(i) + ".png");
    else
        imgPath = ref.abs_path(i);
    end
    if ~isfile(imgPath)
        error("validate_grading:imgMissing", "Image not found: %s", imgPath);
    end

    r = grade_dr(char(imgPath), net);

    ptProbs = arrayfun(@(c) ref.(c)(i), ptProbCols);   % 1x5 double
    mlGrade(i)     = r.grade;
    mlPref(i)      = r.p_referable;
    maxProbDiff(i) = max(abs(r.probabilities(:).' - ptProbs(:).'));
    prefDiff(i)    = abs(r.p_referable - ref.pt_p_referable(i));
    gradeMatch(i)  = (r.grade == ref.pt_grade(i));

    if opts.verbose || ~gradeMatch(i) || maxProbDiff(i) > opts.probTol
        tag = "  ok";
        if ~gradeMatch(i);                 tag = "GRADE"; end
        if maxProbDiff(i) > opts.probTol;  tag = tag + " PROB"; end
        fprintf("[%s] %-14s  pt=%d ml=%d  maxΔp=%.4g  Δp_ref=%.4g\n", ...
            tag, ref.id_code(i), ref.pt_grade(i), r.grade, maxProbDiff(i), prefDiff(i));
    end
end

gradeAgree     = mean(gradeMatch);
referableAgree = mean((mlPref >= 0.5) == (ref.pt_p_referable >= 0.5));

mismRows = ~gradeMatch | (maxProbDiff > opts.probTol);
mismatches = table(ref.id_code(mismRows), ref.pt_grade(mismRows), mlGrade(mismRows), ...
    maxProbDiff(mismRows), prefDiff(mismRows), ...
    VariableNames=["id_code", "pt_grade", "ml_grade", "max_prob_diff", "p_referable_diff"]);

passed = (gradeAgree >= opts.gradeAgreeMin) && (max(maxProbDiff) <= opts.probTol);

fprintf("\n================ ONNX vs PyTorch validation ================\n");
fprintf("  images ................ %d\n", n);
fprintf("  grade agreement ....... %.3f  (min required %.3f)\n", gradeAgree, opts.gradeAgreeMin);
fprintf("  referable agreement ... %.3f\n", referableAgree);
fprintf("  max |Δ probability| ... %.4g  (tol %.4g)\n", max(maxProbDiff), opts.probTol);
fprintf("  mean |Δ probability| .. %.4g\n", mean(maxProbDiff));
fprintf("  max |Δ p_referable| ... %.4g\n", max(prefDiff));
fprintf("  result ................ %s\n", string(passed) + repmat("  <-- PASS", 1, passed));
fprintf("===========================================================\n");

if ~isempty(mismatches)
    fprintf("\nMismatches:\n");
    disp(mismatches);
end

summary = struct( ...
    "n", n, ...
    "grade_agree", gradeAgree, ...
    "max_prob_diff", max(maxProbDiff), ...
    "mean_prob_diff", mean(maxProbDiff), ...
    "max_p_referable_diff", max(prefDiff), ...
    "referable_agree", referableAgree, ...
    "passed", passed, ...
    "mismatches", mismatches);

if ~passed
    error("validate_grading:mismatch", ...
        "ONNX-imported model disagrees with PyTorch beyond tolerance (see summary above).");
end
end


% ======================================================================= %
function p = defaultCsv()
here = fileparts(mfilename("fullpath"));
p = string(fullfile(here, "reference_test.csv"));
end
