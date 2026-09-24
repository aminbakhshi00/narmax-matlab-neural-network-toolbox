function result = run_adding(options)
%RUN_ADDING Train the adding-problem NARX and record every run.
%
%   RESULT = RUN_ADDING('ActivationFcn', F) trains the closed-loop NARX of
%   MAKE_ADDING_NARX at T = 150, 200, 300 and 400 over 15 runs each, and
%   writes one checkpoint per run plus RESULTS.MAT and SUMMARY.XLSX into a
%   results folder.
%
%   The 15 runs vary the initial weights AND the dataset; see ADDING_SEEDS.
%   Both activation functions get the same 15 (weight, dataset) pairs, so the
%   two models are compared run by run on identical data from identical
%   starting points.
%
%   TRAINING. The loss lives only at the final step: the targets are zero
%   placeholders everywhere except t = T, and the error weights are zero
%   everywhere except t = T, where the weight is T. That makes the weighted
%   MSE equal the final-time MSE, so nothing before t = T is regressed
%   against and the network is free to use those steps as memory.
%
%   HOW A RUN STOPS. There is no early stopping in the usual sense --
%   NET.DIVIDEFCN is empty, so there is no validation split and MATLAB's
%   MAX_FAIL never runs. Held-out MSE is computed only to report. Training
%   proceeds in blocks of BLOCKEPOCHS, and a run ends when any of these hits:
%
%     goal        training MSE <= GOAL (1e-6)   success; stop
%     min_grad    gradient < 1e-7               the surface is flat here
%     mu >= mu_max                              LM cannot take a useful step
%     budget      EPOCHBUDGET total epochs      hard ceiling
%     stall       STALLBLOCKS consecutive blocks improving training MSE by
%                 less than STALLTOLERANCE
%
%   The "solved" threshold of 0.01 used in the tables is a REPORTING label
%   applied after a run finishes, on held-out MSE. It never stops anything.
%   The only MSE that stops a run is the training MSE against GOAL.
%
%   STOPREASON in the results table names which of those ended the run:
%   "goal", "budget", "stall", or "interrupted" if MAXSECONDS cut the call
%   short. LASTBLOCKSTOP is MATLAB's own string for the final block, kept
%   because it distinguishes a max-mu block from a flat one -- but it is a
%   BLOCK-level message and must not be read as the run's reason.
%
%   RESUMING. Each run has its own checkpoint and each call trains for at
%   most MAXSECONDS, so call it again to continue; finished runs are loaded,
%   not retrained.
%
%   NAME-VALUE OPTIONS
%     ActivationFcn     "poslin" or "tansig". Default "poslin".
%     Lengths         Default [150 200 300 400].
%     Runs            Default 1:15.
%     NumTrain        Default 5000.
%     NumTest         Default 5000.
%     EpochBudget     Default 2000.
%     BlockEpochs     Default 100.
%     StallBlocks     Default 4.
%     StallTolerance  Default 0.005.
%     Goal            Training-MSE success criterion. Default 1e-6.
%     MaxSeconds      Wall-clock budget for one call. Default 1350.
%     OutputName      Results folder. Default "results_" + ActivationFcn.
%
%   See also MAKE_ADDING_NARX, ADDING_SEEDS, PLOT_ADDING_FIG2.

arguments
    options.ActivationFcn (1,1) string {mustBeMember(options.ActivationFcn, ...
        ["poslin", "tansig"])} = "poslin"
    options.Lengths (1,:) double {mustBePositive} = [150 200 300 400]
    options.Runs (1,:) double {mustBePositive} = 1:15
    options.NumTrain (1,1) double {mustBePositive} = 5000
    options.NumTest (1,1) double {mustBePositive} = 5000
    options.EpochBudget (1,1) double {mustBePositive} = 2000
    options.BlockEpochs (1,1) double {mustBePositive} = 100
    options.StallBlocks (1,1) double {mustBePositive} = 4
    options.StallTolerance (1,1) double = 0.005
    options.Goal (1,1) double = 1e-6
    options.MaxSeconds (1,1) double {mustBePositive} = 1350
    options.OutputName (1,1) string = ""
end

activationFcn = options.ActivationFcn;
if options.OutputName == ""
    options.OutputName = "results_" + activationFcn;
end
outputFolder = fullfile(fileparts(mfilename('fullpath')), options.OutputName);
if ~isfolder(outputFolder), mkdir(outputFolder); end

callTimer = tic;
fprintf('\nADDING PROBLEM, closed-loop NARX, f^1 = %s\n', activationFcn);
fprintf('  D = 8, S^1 = 2 (25 parameters), %d train / %d test\n', ...
    options.NumTrain, options.NumTest);
fprintf('  T %s x %d runs, budget %d epochs, this call stops after %d s\n\n', ...
    mat2str(options.Lengths), numel(options.Runs), options.EpochBudget, ...
    options.MaxSeconds);

rows = struct('sequenceLength', {}, 'activationFcn', {}, 'run', {}, ...
    'epochs', {}, 'trainMSE', {}, 'testMSE', {}, 'baselineMSE', {}, ...
    'ratioToBaseline', {}, 'solved', {}, 'gradient', {}, 'seconds', {}, ...
    'stopReason', {}, 'lastBlockStop', {}, 'complete', {}, ...
    'curveEpoch', {}, 'curveTestMSE', {}, ...
    'curveTrainMSE', {}, 'weights', {}, 'initialWeights', {});
names = struct('weights', strings(1,0), 'initialWeights', strings(1,0));
outOfTime = false;

for run = options.Runs
    for finalTime = options.Lengths
        [dataSeed, weightSeed] = adding_seeds(finalTime, run);
        dataset = generate_adding_dataset('SequenceLength', finalTime, ...
            'NumTrain', options.NumTrain, 'NumTest', options.NumTest, ...
            'Seed', dataSeed);

        pTrain = adding_input_cells(dataset.train);
        yTrain = repmat({zeros(1, options.NumTrain)}, 1, finalTime);
        yTrain{end} = dataset.train.yFinal;
        errorWeights = repmat({zeros(1, options.NumTrain)}, 1, finalTime);
        errorWeights{end} = finalTime * ones(1, options.NumTrain);

        tag = sprintf('T%03d_%s_run%02d', finalTime, activationFcn, run);
        checkpoint = fullfile(outputFolder, tag + ".mat");

        if isfile(checkpoint)
            state = load(checkpoint, 'state').state;
        else
            net = make_adding_narx('Example', dataset.train, ...
                'ActivationFcn', activationFcn, 'Seed', weightSeed);
            net.trainParam.goal = options.Goal;
            state.net = net;
            state.initial = snapshot(net);
            state.epochsDone = 0;
            state.seconds = 0;
            state.gradient = NaN;
            state.stop = "not started";
            state.stalled = 0;
            state.curveEpoch = 0;
            state.curveTestMSE = score(net, dataset.test);
            state.curveTrainMSE = score(net, dataset.train);
        end

        while state.epochsDone < options.EpochBudget && ...
                state.stalled < options.StallBlocks
            if toc(callTimer) > options.MaxSeconds, outOfTime = true; break; end
            state.net.trainParam.epochs = ...
                min(options.BlockEpochs, options.EpochBudget - state.epochsDone);

            blockTimer = tic;
            [state.net, record] = train_block(state.net, pTrain, yTrain, errorWeights);
            state.seconds = state.seconds + toc(blockTimer);
            state.epochsDone = state.epochsDone + record.epoch(end);
            state.stop = string(record.stop);
            if isfield(record, 'gradient') && ~isempty(record.gradient)
                state.gradient = record.gradient(end);
            end

            trainMSE = score(state.net, dataset.train);
            state.curveEpoch(end + 1) = state.epochsDone;
            state.curveTestMSE(end + 1) = score(state.net, dataset.test);
            state.curveTrainMSE(end + 1) = trainMSE;

            previous = state.curveTrainMSE(end - 1);
            gain = (previous - trainMSE) / max(previous, eps);
            if gain < options.StallTolerance
                state.stalled = state.stalled + 1;
            else
                state.stalled = 0;
            end
            save(checkpoint, 'state');

            if trainMSE <= options.Goal, break; end
        end

        trainMSE = score(state.net, dataset.train);
        testMSE = score(state.net, dataset.test);
        trained = struct('IW', state.net.IW{1,1}, 'LW12', state.net.LW{1,2}, ...
            'LW21', state.net.LW{2,1}, 'b1', state.net.b{1}, 'b2', state.net.b{2});

        row.sequenceLength = finalTime;
        row.activationFcn = activationFcn;
        row.run = run;
        row.epochs = state.epochsDone;
        row.trainMSE = trainMSE;
        row.testMSE = testMSE;
        row.baselineMSE = dataset.baselineMSE;
        row.ratioToBaseline = testMSE / dataset.baselineMSE;
        row.solved = testMSE < 0.01;
        row.gradient = state.gradient;
        row.seconds = state.seconds;
        % MATLAB reports why the LAST BLOCK ended, and a block's own epoch
        % limit is BLOCKEPOCHS, not the run's budget -- so "Reached maximum
        % number of epochs" from a full block says nothing about why the RUN
        % ended. Record the run-level reason separately, from the conditions
        % the loop actually tests.
        if trainMSE <= options.Goal
            row.stopReason = "goal";
        elseif state.epochsDone >= options.EpochBudget
            row.stopReason = "budget";
        elseif state.stalled >= options.StallBlocks
            row.stopReason = "stall";
        else
            row.stopReason = "interrupted";
        end
        row.lastBlockStop = state.stop;
        row.complete = row.stopReason ~= "interrupted";
        row.curveEpoch = state.curveEpoch;
        row.curveTestMSE = state.curveTestMSE;
        row.curveTrainMSE = state.curveTrainMSE;
        row.weights = flatten(trained);
        row.initialWeights = flatten(state.initial);
        % The initial snapshot holds the three weight matrices the heatmaps
        % draw; the trained one also holds the biases. Name each from the
        % struct it came from, or the two sheets disagree on their width.
        names.weights = weight_names(trained);
        names.initialWeights = weight_names(state.initial);
        rows(end + 1) = row; %#ok<AGROW>

        if row.solved, mark = 'SOLVED'; else, mark = ''; end
        fprintf('  run %2d  T%3d  %6d ep  test %9.3e (%.3fx base)  %5.0f s  %s\n', ...
            run, finalTime, state.epochsDone, testMSE, row.ratioToBaseline, ...
            state.seconds, mark);
        if outOfTime, break; end
    end
    if outOfTime, break; end
end

result.table = struct2table(rows, 'AsArray', true);
result.weightNames = names;
save(fullfile(outputFolder, 'results.mat'), 'result');
export(outputFolder, result);
report(result, options);
end


function mse = score(net, split)
%SCORE Roll the closed loop forward and read the final-time held-out MSE.
a2Cells = net(adding_input_cells(split));
a2 = vertcat(a2Cells{:});
if all(isfinite(a2), 'all')
    mse = mean((split.yFinal - a2(end, :)).^2);
else
    mse = Inf;                                   % a diverged run
end
end


function [net, record] = train_block(net, p, y, ew)
%TRAIN_BLOCK One block of training, serial if the pool refuses.
try
    [net, record] = train(net, p, y, [], [], ew, 'useParallel', 'yes', 'useGPU', 'no');
catch
    [net, record] = train(net, p, y, [], [], ew, 'useParallel', 'no', 'useGPU', 'no');
end
end


function s = snapshot(net)
%SNAPSHOT The three weight matrices the heatmaps draw.
s = struct('IW', net.IW{1,1}, 'LW12', net.LW{1,2}, 'LW21', net.LW{2,1});
end


function v = flatten(s)
%FLATTEN Column-major concatenation of every field of a weight struct.
v = [];
for f = string(fieldnames(s)).'
    v = [v, reshape(s.(f), 1, [])]; %#ok<AGROW>
end
end


function names = weight_names(s)
%WEIGHT_NAMES One column name per entry of FLATTEN(S), in the same order.
names = strings(1, 0);
for f = string(fieldnames(s)).'
    m = s.(f);
    for c = 1:size(m, 2)
        for r = 1:size(m, 1)
            names(end + 1) = sprintf('%s_%d_%d', f, r, c); %#ok<AGROW>
        end
    end
end
end


function export(outputFolder, result)
%EXPORT summary.csv and summary.xlsx, skipping the curve and weight columns.
plain = result.table;
plain = removevars(plain, {'curveEpoch', 'curveTestMSE', 'curveTrainMSE', ...
                           'weights', 'initialWeights'});
try
    writetable(plain, fullfile(outputFolder, 'summary.csv'));
    book = fullfile(outputFolder, 'summary.xlsx');
    if isfile(book), delete(book); end
    writetable(plain, book, 'Sheet', 'runs');
    writetable(weight_sheet(result, 'weights'), book, 'Sheet', 'weights_trained');
    writetable(weight_sheet(result, 'initialWeights'), book, 'Sheet', 'weights_initial');
catch err
    % A spreadsheet open in another program must never discard finished runs.
    warning('run_adding:export', 'summary not written: %s', err.message);
end
end


function sheet = weight_sheet(result, field)
%WEIGHT_SHEET One row per run, one column per weight.
values = result.table.(field);
if iscell(values)
    % STRUCT2TABLE only leaves these as a cell when the rows differ in
    % length, which they do not here; handle both so the sheet never fails.
    values = vertcat(values{:});
end
sheet = [result.table(:, {'sequenceLength', 'activationFcn', 'run', 'testMSE'}), ...
         array2table(values, 'VariableNames', cellstr(result.weightNames.(field)))];
end


function report(result, options)
%REPORT Solved counts and medians per sequence length.
T = result.table;
fprintf('\n  %d of %d requested runs recorded\n\n', height(T), ...
    numel(options.Lengths) * numel(options.Runs));
fprintf('  T      solved      median testMSE   best testMSE\n');
for L = options.Lengths
    v = T.testMSE(T.sequenceLength == L);
    if isempty(v), continue; end
    fprintf('  %3d   %2d of %2d      %12.3e   %12.3e\n', L, sum(v < 0.01), ...
        numel(v), median(v), min(v));
end
fprintf('\n  baseline, always predicting the mean: %.4f\n\n', ...
    mean(T.baselineMSE));
end
