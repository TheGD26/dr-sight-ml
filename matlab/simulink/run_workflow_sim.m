%RUN_WORKFLOW_SIM  Drive dr_screening_workflow.slx and produce the plots a
%                  judge needs to evaluate the resource-allocation claim.
%
%   Run from anywhere:  >> run_workflow_sim
%
%   Figures (also saved as PNG next to this file):
%     1. fig_queues.png          - stage backlog over time at the 100k/yr target
%     2. fig_queues_stress.png   - same, at a volume where the review stage
%                                  backlogs, so the bottleneck is visible
%     3. fig_wait_hist.png       - distribution of estimated patient turnaround
%     4. fig_reviewer_capacity.png - max sustainable annual volume vs # reviewers
%                                    (the "how many ophthalmologists?" answer)

clear; clc;
here = fileparts(mfilename('fullpath'));
cd(here);

assert(license('test','Simulink') == 1, ...
    'Simulink is required. Open MATLAB with Simulink and re-run.');

mdl = 'dr_screening_workflow';
if exist(fullfile(here, [mdl '.slx']), 'file') == 0
    fprintf('Model not found - building it from build_dr_screening_workflow.m ...\n');
    build_dr_screening_workflow(mdl);
end

%% ===================== PARAMETERS ====================================
annual_volume      = 100000;        % district programme target (patients/year)
patients_per_day   = annual_volume/365;
images_per_patient = 2;             % one macula-centred image per eye
image_MB           = 0.35;          % compressed fundus JPEG
bandwidth_Mbps     = 5;             % rural clinic uplink
n_cameras          = 6;             % screening camps uploading in parallel
% Measured steady-state wall-clock time of one screen_image.m call through the
% live MATLAB engine (quality gate + ONNX grade + Grad-CAM), excluding engine
% start / ONNX import / warm-up. 24 timed calls on distinct APTOS fundus images:
% mean 1.06 s, range 0.68-1.54 s. See scripts/measure_matlab_latency.py.
ai_time_s          = 1.06;           % quality gate + inference per image (measured)
% ai_time_s        = 2.0;            % pre-measurement placeholder (typical-CPU estimate)
n_ai_workers       = 2;             % inference worker processes
referral_frac      = 0.30;          % share of images escalated to a human
review_time_s      = 120;           % ophthalmologist grade + read + sign, per case
n_reviewers        = 2;             % baseline panel size

sim_days           = 45;            % horizon (long enough for backlog to show)
fixed_step         = 5;             % solver step (s)
Tdrain             = 60;            % backlog relief time constant (s)
log_decim          = 60;            % logging decimation

% ---- capacities (images/s), for annotation & the analytic checks -----
upload_time_s = image_MB*8 / bandwidth_Mbps;
cap_upload    = n_cameras    / upload_time_s;
cap_ai        = n_ai_workers / ai_time_s;
cap_review    = @(nr) nr     / review_time_s;                % referred images/s

% system-wide ceiling in patients/year, given each stage's capacity
cap_to_annual = @(c_imgs) c_imgs*86400/images_per_patient*365;
ceil_upload   = cap_to_annual(cap_upload);
ceil_ai       = cap_to_annual(cap_ai);
ceil_review   = @(nr) cap_to_annual(cap_review(nr)/max(referral_frac,eps));

fprintf('\n--- capacity vs the 100,000 patients/year target ---\n');
fprintf('  upload stage ceiling : %10.0f patients/yr\n', ceil_upload);
fprintf('  AI stage ceiling     : %10.0f patients/yr\n', ceil_ai);
fprintf('  review ceiling (2 rev): %9.0f patients/yr\n', ceil_review(2));
fprintf('  binding stage        : %s\n\n', bottleneck(ceil_upload, ceil_ai, ceil_review(n_reviewers)));

%% ===================== BASELINE RUN (100k/yr) =======================
out = run_once(mdl);
plot_queues(out, sprintf('Stage backlog at the target (%d reviewers, %.0f patients/day)', ...
            n_reviewers, patients_per_day), fullfile(here,'fig_queues.png'));

% ---- estimated patient turnaround distribution ---------------------
qU  = out.q_upload.Data; qA = out.q_ai.Data; qR = out.q_review.Data;
wait_min = ( (qU./cap_upload + upload_time_s) ...
           + (qA./cap_ai     + ai_time_s) ...
           + referral_frac.*(qR./max(cap_review(n_reviewers),eps) + review_time_s) ) / 60;

f = figure('Name','Wait-time distribution','Color','w','Position',[100 100 720 400]);
histogram(wait_min, 40, 'Normalization','probability'); grid on;
xlabel('estimated time from capture to signed report (minutes)');
ylabel('fraction of simulated time');
title(sprintf('Patient turnaround at the target\nmedian %.1f min   |   90th pct %.1f min', ...
      median(wait_min), pctl(wait_min,90)));
saveas(f, fullfile(here,'fig_wait_hist.png'));

%% ===================== REVIEWER CAPACITY SWEEP =====================
% For each panel size, find the largest annual volume the *whole pipeline*
% sustains without an ever-growing backlog. Verified by simulation:
% bisect on annual_volume, re-running the model each step.
sweep_reviewers = 1:6;
max_annual      = zeros(size(sweep_reviewers));
base_nr = n_reviewers; base_vol = annual_volume;

for k = 1:numel(sweep_reviewers)
    n_reviewers = sweep_reviewers(k);
    lo = 1e4; hi = 5e6;                         % patients/year search bracket
    for it = 1:12
        annual_volume    = 0.5*(lo+hi);
        patients_per_day = annual_volume/365;
        if is_stable(run_once(mdl))
            lo = annual_volume;
        else
            hi = annual_volume;
        end
    end
    max_annual(k) = lo;
    fprintf('  %d reviewer(s): sustains up to %8.0f patients/year\n', ...
            sweep_reviewers(k), max_annual(k));
end
n_reviewers = base_nr; annual_volume = base_vol; patients_per_day = annual_volume/365;

min_reviewers = find(max_annual >= 100000, 1);

f = figure('Name','Reviewer capacity','Color','w','Position',[120 120 800 440]);
bar(sweep_reviewers, max_annual, 0.6, 'FaceColor',[0.30 0.55 0.85], 'EdgeColor','none');
hold on; grid on;
ylim([0, max(max_annual)*1.15]);
yline(100000, '--', '100,000 patients/year target', 'LineWidth',1.6, ...
      'Color',[0.1 0.1 0.1], 'LabelHorizontalAlignment','left');
xlabel('number of ophthalmologist-reviewers');
ylabel(sprintf('max sustainable volume (patients/year)\n[AI/upload stages cap the pipeline at ~%.1fM/yr]', ...
       min([ceil_upload ceil_ai])/1e6));
if ~isempty(min_reviewers)
    title(sprintf('%d reviewer(s) sustain 100,000 patients/year; each adds ~%.0fk/yr', ...
          min_reviewers, mean(diff(max_annual))/1e3));
else
    title('No tested panel size sustains 100,000 patients/year - check other stages');
end
saveas(f, fullfile(here,'fig_reviewer_capacity.png'));

%% ===================== STRESSED RUN (review backlogs) =============
% Pick a volume ~1.5x what the baseline panel can sustain, so fig_queues_stress
% shows the review queue climbing without bound.
annual_volume    = 1.5 * max_annual(base_nr);
patients_per_day = annual_volume/365;
plot_queues(run_once(mdl), ...
    sprintf('Stage backlog at %.0f patients/year (%d reviewers) - review stage saturates', ...
            annual_volume, base_nr), fullfile(here,'fig_queues_stress.png'));
annual_volume = base_vol; patients_per_day = annual_volume/365;

fprintf('\n==============================================================\n');
if ~isempty(min_reviewers)
    fprintf(' RESULT: %d ophthalmologist-reviewer(s) sustain 100,000 patients/year\n', min_reviewers);
    fprintf('         (%.0f s/case, %.0f%% referral rate). The whole pipeline is\n', ...
            review_time_s, 100*referral_frac);
    fprintf('         ultimately capped at ~%.0f patients/year by the %s stage.\n', ...
            min([ceil_upload ceil_ai]), bottleneck(ceil_upload, ceil_ai, inf));
else
    fprintf(' RESULT: review stage is not the binding constraint at 100k/yr.\n');
end
fprintf('==============================================================\n');
fprintf('Figures written to %s\n', here);

%% ===================== helpers =====================================
function out = run_once(mdl)
    % This file is a script: its parameter assignments already live in the
    % base workspace, where the model resolves block expressions. The sweeps
    % just reassign n_reviewers / patients_per_day before calling here.
    out = sim(Simulink.SimulationInput(mdl));
end

function tf = is_stable(out)
    % backlog is bounded if no stage queue is still growing over the last
    % 40% of the run
    tf = ~(growing(out.q_upload) || growing(out.q_ai) || growing(out.q_review));
    function g = growing(ts)
        n = numel(ts.Time); i = max(2, round(0.6*n));
        tt = ts.Time(i:end); tt = tt - tt(1);   % centre for conditioning
        p = polyfit(tt, ts.Data(i:end), 1);
        g = p(1) > 1e-4;                         % images/s
    end
end

function plot_queues(out, ttl, pngpath)
    td = out.q_review.Time/86400;
    f = figure('Color','w','Position',[80 80 800 430]);
    plot(td, out.q_upload.Data, 'LineWidth',1.4); hold on;
    plot(td, out.q_ai.Data,     'LineWidth',1.4);
    plot(td, out.q_review.Data, 'LineWidth',1.4);
    grid on; xlabel('time (days)'); ylabel('backlog (images waiting)');
    legend({'acquisition + upload','quality gate + AI','ophthalmologist review'}, ...
           'Location','northwest');
    title(ttl);
    saveas(f, pngpath);
end

function s = bottleneck(u, a, r)
    [~, i] = min([u a r]); names = {'upload','AI','review'};
    s = names{i};
end

function q = pctl(x, p)
    x = sort(x(~isnan(x)));
    if isempty(x); q = NaN; return; end
    q = interp1(linspace(0,100,numel(x))', x, p, 'linear', 'extrap');
end
