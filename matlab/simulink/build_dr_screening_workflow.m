function mdlPath = build_dr_screening_workflow(mdl)
%BUILD_DR_SCREENING_WORKFLOW  Programmatically build dr_screening_workflow.slx.
%
%   mdlPath = BUILD_DR_SCREENING_WORKFLOW()
%   mdlPath = BUILD_DR_SCREENING_WORKFLOW('my_model_name')
%
%   Constructs a scoped-down discrete-event model of the district-level
%   telemedicine DR-screening pipeline (SIH PS26038) and saves it as
%   dr_screening_workflow.slx next to this file.
%
%   The .slx is generated from code (not hand-drawn) so it can be rebuilt on
%   any machine with base Simulink -- no SimEvents, no add-on toolboxes.
%   run_workflow_sim.m calls this automatically if the .slx is missing.
%
%   MODEL  (deterministic "fluid queue" approximation)
%   -------------------------------------------------------------------
%   A screening programme is a chain of capacity-limited stages. Each
%   stage's backlog is a fill level Q(t) (unit: images):
%
%       dQ/dt        = inflow(t) - served(t)
%       served(t)    = min( inflow(t) + Q/Tdrain , stage_capacity )
%       Q >= 0                                 (queue can't go negative)
%
%   * inflow < capacity, Q = 0  -> serve exactly what arrives, no backlog
%   * inflow > capacity         -> served saturates at capacity, Q grows unbounded
%   * Q/Tdrain term             -> a transient backlog drains once capacity
%                                  frees up (Tdrain = relief time const, s)
%
%   STAGES (each = Constant capacity + Sum + MinMax + Gain + Discrete-Time
%   Integrator with a lower limit of 0; all standard Simulink blocks)
%     1. Acquisition + upload : capacity = n_cameras / upload_time
%                               upload_time = image_MB*8 / bandwidth_Mbps
%     2. Quality gate + AI     : capacity = n_ai_workers / ai_time_s
%     3. Ophthalmologist review: capacity = n_reviewers / review_time_s
%                               (only referral_frac of images reach here)
%
%   Physical parameters are plain base-workspace variables so they can be
%   swept from run_workflow_sim.m. Defaults are assigned here for any that
%   are undefined, so the model also opens and runs stand-alone.

    if nargin < 1 || isempty(mdl)
        mdl = 'dr_screening_workflow';
    end

    here    = fileparts(mfilename('fullpath'));
    mdlPath = fullfile(here, [mdl '.slx']);

    assert(license('test','Simulink') == 1, ...
        'Simulink is required to build this model.');

    assign_default_params();

    if bdIsLoaded(mdl); close_system(mdl, 0); end
    load_system('simulink');
    new_system(mdl);

    set_param(mdl, 'SolverType', 'Fixed-step');
    set_param(mdl, 'Solver',     'FixedStepDiscrete');
    set_param(mdl, 'FixedStep',  'fixed_step');
    set_param(mdl, 'StopTime',   'sim_days*86400');
    set_param(mdl, 'SaveOutput', 'on');
    set_param(mdl, 'SaveTime',   'on');
    set_param(mdl, 'SaveFormat', 'Dataset');
    set_param(mdl, 'ReturnWorkspaceOutputs', 'on');

    % ---- patient / image arrival stream ------------------------------
    add_block('built-in/Constant', [mdl '/Arrivals'], ...
        'Position', xywh(40, 90, 90, 40), ...
        'Value', 'patients_per_day*images_per_patient/86400');

    % ---- three capacity-limited queue stages ------------------------
    add_queue_stage(mdl, 'Upload', ...
        'n_cameras / (image_MB*8 / bandwidth_Mbps)', 240);
    add_queue_stage(mdl, 'AI', ...
        'n_ai_workers / ai_time_s', 620);
    add_queue_stage(mdl, 'Review', ...
        'n_reviewers / review_time_s', 1040);

    % only referral_frac of AI-processed images are escalated to a human
    add_block('built-in/Gain', [mdl '/ReferralFilter'], ...
        'Position', xywh(960, 95, 40, 30), 'Gain', 'referral_frac');

    % ---- logging (To Workspace, decimated) -------------------------
    tw = @(name, y) add_block('simulink/Sinks/To Workspace', [mdl '/' name], ...
        'Position', xywh(1280, y, 90, 34), 'VariableName', name, ...
        'SaveFormat', 'Timeseries', 'SampleTime', '-1', ...
        'Decimation', 'log_decim', 'MaxDataPoints', 'inf');
    tw('arrival_rate',  40);
    tw('q_upload',      110);
    tw('served_upload', 170);
    tw('q_ai',          230);
    tw('served_ai',     290);
    tw('q_review',      350);
    tw('served_review', 410);

    % ---- wiring: each stage's inflow feeds BOTH its demand sum and its
    %      dQ sum, so the queue integrates (inflow - served) correctly. ----
    L = @(a, b) add_line(mdl, a, b, 'autorouting', 'on');

    feed(L, 'Arrivals/1',      'Upload');
    L('Arrivals/1',            'arrival_rate/1');

    feed(L, 'Upload_served/1', 'AI');
    L('Upload_served/1',       'served_upload/1');
    L('Upload_Q/1',            'q_upload/1');

    L('AI_served/1',           'ReferralFilter/1');
    L('AI_served/1',           'served_ai/1');
    L('AI_Q/1',                'q_ai/1');

    feed(L, 'ReferralFilter/1', 'Review');
    L('Review_served/1',       'served_review/1');
    L('Review_Q/1',            'q_review/1');

    save_system(mdl, mdlPath);
    close_system(mdl, 0);
    fprintf('Built %s\n', mdlPath);
end

% ======================================================================
function feed(L, srcPort, stage)
%FEED  Route one inflow signal into a stage's demand sum and dQ sum.
    L(srcPort, sprintf('%s_demand/1', stage));
    L(srcPort, sprintf('%s_dQ/1',     stage));
end

% ======================================================================
function add_queue_stage(mdl, name, capExpr, x)
%ADD_QUEUE_STAGE  Flat set of standard blocks implementing one fluid queue.
%   Named ports for the parent to wire:
%     <name>_demand/1 , <name>_dQ/1   inflow sinks (parent branches inflow here)
%     <name>_served/1                 service-rate source
%     <name>_Q/1                      queue-length source
    B = @(bl, suffix, pos, varargin) add_block(bl, ...
        sprintf('%s/%s_%s', mdl, name, suffix), 'Position', pos, varargin{:});

    % demand = inflow + Q/Tdrain      (port1 = inflow, port2 = drain term)
    B('built-in/Sum', 'demand', xywh(x, 88, 32, 32), 'Inputs', '++');

    B('built-in/Constant', 'cap', xywh(x, 210, 150, 34), 'Value', capExpr);

    % served = min(demand, capacity)
    B('simulink/Math Operations/MinMax', 'served', xywh(x+110, 92, 44, 44), ...
        'Function', 'min', 'Inputs', '2');

    % dQ = inflow - served            (port1 = +inflow, port2 = -served)
    B('built-in/Sum', 'dQ', xywh(x+110, 210, 32, 32), 'Inputs', '+-');

    % Q = integral(dQ), clamped at 0
    B('simulink/Discrete/Discrete-Time Integrator', 'Q', ...
        xywh(x+200, 205, 70, 44), ...
        'gainval', '1', 'SampleTime', 'fixed_step', ...
        'IntegratorMethod', 'Integration: Forward Euler', ...
        'LimitOutput', 'on', 'LowerSaturationLimit', '0', ...
        'UpperSaturationLimit', 'inf', 'InitialCondition', '0');

    % Q/Tdrain feedback into the demand sum
    B('built-in/Gain', 'drain', xywh(x+110, 300, 44, 30), 'Gain', '1/Tdrain');

    l = @(a, b) add_line(mdl, sprintf('%s_%s', name, a), ...
                              sprintf('%s_%s', name, b), 'autorouting', 'on');
    l('demand/1', 'served/1');
    l('cap/1',    'served/2');
    l('served/1', 'dQ/2');
    l('dQ/1',     'Q/1');
    l('Q/1',      'drain/1');
    l('drain/1',  'demand/2');
end

% ======================================================================
function p = xywh(x, y, w, h)
    p = [x, y, x + w, y + h];
end

% ======================================================================
function assign_default_params()
%ASSIGN_DEFAULT_PARAMS  Define any missing model parameters in base ws.
%   Baseline: a district programme screening 100,000 patients/year.
    d = struct( ...
        'patients_per_day',   100000/365, ...  % ~274 screenings/day
        'images_per_patient', 2,          ...  % one macula-centred image per eye
        'image_MB',           0.35,       ...  % compressed fundus JPEG
        'bandwidth_Mbps',     5,          ...  % rural clinic uplink
        'n_cameras',          6,          ...  % camps operating in parallel
        'ai_time_s',          1.06,       ...  % measured screen_image.m call (was 2.0 est.)
        'n_ai_workers',       2,          ...  % inference worker processes
        'referral_frac',      0.30,       ...  % share escalated to a human
        'review_time_s',      120,        ...  % grade + read + sign report
        'n_reviewers',        2,          ...  % ophthalmologists on the panel
        'sim_days',           45,         ...  % horizon long enough to see backlog
        'fixed_step',         5,          ...  % solver step (s)
        'Tdrain',             60,         ...  % backlog relief time constant (s)
        'log_decim',          60);             % keep every 60th sample (~300 s)

    f = fieldnames(d);
    for i = 1:numel(f)
        if evalin('base', sprintf('~exist(''%s'',''var'')', f{i}))
            assignin('base', f{i}, d.(f{i}));
        end
    end
end
