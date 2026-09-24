function run_variable_length(options)
%RUN_VARIABLE_LENGTH Train the variable-length model on mixed sequence lengths.
%
%   RUN_VARIABLE_LENGTH trains the closed-loop (parallel) NARX of
%   MAKE_VARLEN_NARX, S^1 = 2 with D = 1 and D = 4 feedback taps, for poslin
%   and tansig, on each length range in LENGTHS. Every training and test
%   sequence has its own length L, drawn from the range (MAKE_VARLEN_DATASET).
%   Each range gets its own folder, results_T<min>-<max>, holding one
%   checkpoint per run plus RESULTS.MAT, SUMMARY.CSV and SUMMARY.XLSX in the
%   format of RUN_ADDING.
%
%   MIXED LENGTHS. Sequences are zero-padded to the longest length. The
%   target y(L) sits at each sequence's own L with error weight MAXLENGTH,
%   and every other step has weight 0. MATLAB's weighted MSE averages over
%   all MAXLENGTH x Q entries, so the training loss is exactly the final-time
%   MSE, mean over q of (y(L_q) - a^2(L_q))^2. The network is never told L;
%   the output is simply read at t = L_q. It is causal, so the padding after
%   L_q cannot reach a^2(L_q).
%
%   TRAINING follows RUN_ADDING: trainlm in blocks of 100 epochs until
%   training MSE <= 1e-6 ("goal"), 2000 epochs ("budget"), or 4 blocks in a
%   row each improving training MSE by less than 0.5% ("stall"). No
%   validation split. Solved means held-out MSE < 0.01. Run k uses weight
%   seed 100 + k and its own dataset.
%
%   Runs are trained in a PARFOR, so an open pool trains them side by side.
%   A finished run is loaded from its checkpoint instead of retrained, and
%   runs already in a folder's RESULTS.MAT but not requested in this call
%   are kept, so a configuration can be added later without retraining.
%
%   NAME-VALUE OPTIONS
%     Lengths         One [min max] range per row. Default [150 200; 300 400].
%     ActivationFcns  Default ["poslin" "tansig"].
%     Delays          D. Default [1 4].
%     Runs            Default 1:5.
%
%   See also MAKE_VARLEN_NARX, MAKE_VARLEN_DATASET, PLOT_VARLEN_HEATMAPS, RUN_ADDING.

arguments
    options.Lengths (:,2) double = [150 200; 300 400]
    options.ActivationFcns (1,:) string = ["poslin" "tansig"]
    options.Delays (1,:) double = [1 4]
    options.Runs (1,:) double = 1:5
end

here = fileparts(mfilename('fullpath'));
[r, a, d, k] = ndgrid(1:size(options.Lengths, 1), 1:numel(options.ActivationFcns), ...
    options.Delays, options.Runs);
jobMin = options.Lengths(r(:), 1);
jobMax = options.Lengths(r(:), 2);
jobFcn = options.ActivationFcns(a(:));
jobDelay = d(:);
jobRun = k(:);
for range = options.Lengths.'
    folder = fullfile(here, results_folder(range(1), range(2)));
    if ~isfolder(folder), mkdir(folder); end
end

rows = cell(numel(jobRun), 1);
parfor j = 1:numel(jobRun)
    rows{j} = train_run(here, jobMin(j), jobMax(j), jobFcn(j), jobDelay(j), jobRun(j));
end
rows = [rows{:}];

for range = options.Lengths.'
    mine = [rows.minLength] == range(1) & [rows.maxLength] == range(2);
    result.table = as_cells(struct2table(rows(mine), 'AsArray', true));
    folder = fullfile(here, results_folder(range(1), range(2)));
    file = fullfile(folder, 'results.mat');
    if isfile(file)                    % keep earlier runs not trained in this call
        old = load(file, 'result').result.table;
        key = {'activationFcn', 'delay', 'run'};
        old = old(~ismember(old(:, key), result.table(:, key)), :);
        result.table = sortrows([old; result.table], key);
    end
    save(file, 'result');
    export(folder, result.table);
    report(result.table);
end
end


function row = train_run(here, minLength, maxLength, activationFcn, delay, run)
%TRAIN_RUN Train (or resume) one run and return its summary row.
goal = 1e-6; budget = 2000; blockEpochs = 100; stallBlocks = 4; stallTolerance = 0.005;

dataset = make_varlen_dataset(minLength, maxLength, 5000, 5000, run);
pTrain = adding_input_cells(dataset.train);
yTrain = repmat({zeros(1, 5000)}, 1, maxLength);
errorWeights = yTrain;
for L = unique(dataset.train.lengths)
    q = dataset.train.lengths == L;
    yTrain{L}(q) = dataset.train.yFinal(q);
    errorWeights{L}(q) = maxLength;
end

tag = sprintf('T%d-%d_%s_D%d_run%02d', minLength, maxLength, activationFcn, delay, run);
checkpoint = fullfile(here, results_folder(minLength, maxLength), tag + ".mat");
if isfile(checkpoint)
    state = load(checkpoint, 'state').state;
else
    net = make_varlen_narx(dataset.train, activationFcn, delay, 100 + run);
    net.trainParam.goal = goal;
    state.net = net;
    state.initial = weights_of(net);
    state.epochsDone = 0;
    state.seconds = 0;
    state.gradient = NaN;
    state.stop = "not started";
    state.stalled = 0;
    state.curveEpoch = 0;
    state.curveTrainMSE = score(net, dataset.train);
    state.curveTestMSE = score(net, dataset.test);
end

while state.curveTrainMSE(end) > goal && state.epochsDone < budget && ...
        state.stalled < stallBlocks
    state.net.trainParam.epochs = min(blockEpochs, budget - state.epochsDone);
    timer = tic;
    [state.net, record] = train(state.net, pTrain, yTrain, [], [], errorWeights, ...
        'useParallel', 'no', 'useGPU', 'no');
    state.seconds = state.seconds + toc(timer);
    state.epochsDone = state.epochsDone + record.epoch(end);
    state.stop = string(record.stop);
    state.gradient = record.gradient(end);

    state.curveEpoch(end + 1) = state.epochsDone;
    state.curveTrainMSE(end + 1) = score(state.net, dataset.train);
    state.curveTestMSE(end + 1) = score(state.net, dataset.test);
    previous = state.curveTrainMSE(end - 1);
    if (previous - state.curveTrainMSE(end)) / max(previous, eps) < stallTolerance
        state.stalled = state.stalled + 1;
    else
        state.stalled = 0;
    end
    save(checkpoint, 'state');
end

row.minLength = minLength;
row.maxLength = maxLength;
row.activationFcn = activationFcn;
row.delay = delay;
row.run = run;
row.epochs = state.epochsDone;
row.trainMSE = state.curveTrainMSE(end);
row.testMSE = state.curveTestMSE(end);
row.baselineMSE = dataset.baselineMSE;
row.ratioToBaseline = row.testMSE / dataset.baselineMSE;
row.solved = row.testMSE < 0.01;
row.gradient = state.gradient;
row.seconds = state.seconds;
if row.trainMSE <= goal
    row.stopReason = "goal";
elseif state.epochsDone >= budget
    row.stopReason = "budget";
else
    row.stopReason = "stall";
end
row.lastBlockStop = state.stop;       % MATLAB's reason for the last BLOCK, not the run
row.curveEpoch = state.curveEpoch;
row.curveTrainMSE = state.curveTrainMSE;
row.curveTestMSE = state.curveTestMSE;
row.weights = flatten(weights_of(state.net));
row.initialWeights = flatten(state.initial);
end


function name = results_folder(minLength, maxLength)
name = sprintf('results_T%d-%d', minLength, maxLength);
end


function mse = score(net, split)
%SCORE Closed-loop rollout; MSE of a^2(L_q) against y(L_q), each at its own L_q.
a2Cells = net(adding_input_cells(split));
a2 = vertcat(a2Cells{:});
a2Final = a2(sub2ind(size(a2), split.lengths, 1:numel(split.lengths)));
if all(isfinite(a2Final))
    mse = mean((split.yFinal - a2Final).^2);
else
    mse = Inf;                                   % a diverged run
end
end


function s = weights_of(net)
s = struct('IW', net.IW{1,1}, 'LW12', net.LW{1,2}, 'LW21', net.LW{2,1}, ...
    'b1', net.b{1}, 'b2', net.b{2});
end


function v = flatten(s)
%FLATTEN Column-major concatenation of every field, in WEIGHT_NAMES order.
v = [];
for f = string(fieldnames(s)).'
    v = [v, reshape(s.(f), 1, [])]; %#ok<AGROW>
end
end


function names = weight_names(delay)
%WEIGHT_NAMES One column name per entry of FLATTEN for S^1 = 2 and D taps.
sizes = struct('IW', [2 2], 'LW12', [2 delay], 'LW21', [1 2], 'b1', [2 1], 'b2', [1 1]);
names = strings(1, 0);
for f = string(fieldnames(sizes)).'
    [r, c] = ndgrid(1:sizes.(f)(1), 1:sizes.(f)(2));
    names = [names, compose("%s_%d_%d", f, r(:), c(:)).']; %#ok<AGROW>
end
end


function T = as_cells(T)
%AS_CELLS One cell per row for the vector columns, which STRUCT2TABLE turns
%   into a matrix whenever every row happens to have the same length.
for v = ["curveEpoch", "curveTrainMSE", "curveTestMSE", "weights", "initialWeights"]
    if ~iscell(T.(v)), T.(v) = num2cell(T.(v), 2); end
end
end


function export(folder, T)
%EXPORT summary.csv and summary.xlsx: a runs sheet, then weight sheets per D.
plain = removevars(T, {'curveEpoch', 'curveTrainMSE', 'curveTestMSE', ...
    'weights', 'initialWeights'});
try
    writetable(plain, fullfile(folder, 'summary.csv'));
    book = fullfile(folder, 'summary.xlsx');
    if isfile(book), delete(book); end
    writetable(plain, book, 'Sheet', 'runs');
    for delay = unique(T.delay).'
        m = T.delay == delay;
        id = T(m, {'activationFcn', 'delay', 'run', 'testMSE'});
        names = cellstr(weight_names(delay));
        writetable([id, array2table(vertcat(T.weights{m}), 'VariableNames', names)], ...
            book, 'Sheet', sprintf('weights_trained_D%d', delay));
        writetable([id, array2table(vertcat(T.initialWeights{m}), 'VariableNames', names)], ...
            book, 'Sheet', sprintf('weights_initial_D%d', delay));
    end
catch err
    % A spreadsheet open in another program must never discard finished runs.
    warning('run_variable_length:export', 'summary not written: %s', err.message);
end
end


function report(T)
%REPORT Solved counts and medians per activation and D.
fprintf('\n  T = %d..%d\n  f^1      D   solved   median testMSE   best testMSE   median epochs\n', ...
    T.minLength(1), T.maxLength(1));
for fcn = unique(T.activationFcn).'
    for delay = unique(T.delay).'
        m = T.activationFcn == fcn & T.delay == delay;
        if ~any(m), continue; end
        fprintf('  %-7s %2d   %d of %d   %12.3e   %12.3e   %8.0f\n', fcn, delay, ...
            sum(T.solved(m)), nnz(m), median(T.testMSE(m)), min(T.testMSE(m)), ...
            median(T.epochs(m)));
    end
end
fprintf('  baseline, always predicting 1: %.4f\n\n', mean(T.baselineMSE));
end
