function results = run_load_relu_comparison(options)
%RUN_LOAD_RELU_COMPARISON Does a ReLU hidden layer forecast load any better?
%
%   RESULTS = RUN_LOAD_RELU_COMPARISON() fits the load model twice over the
%   whole tapped-delay sweep, once with layer 1's default transfer function
%   `tansig` and once with `poslin`, MATLAB's ReLU, and reports the held-out
%   error of the two side by side.
%
%   Everything except f^1 is held fixed: the same training season, neurons,
%   epoch budget and horizon curriculum, the same restart seeds
%   RandomSeed + r, the same validation split to choose on, and the same
%   held-out prediction origins to score on. Restarts are fitted per delay
%   because a closed-loop NARX fitted once sometimes lands on a solution
%   whose rollout is unstable; the run with the lowest error on the
%   validation season is the one carried forward, and the held-out season is
%   never consulted to choose anything.
%
%   ReLU is initialised for ReLU: NARXMODEL draws layer 1 from
%   N(0, 2/fanIn) with zero biases whenever f^1 is `poslin`.
%
%   NEURONS defaults to 5, for both transfer functions. A sweep at 30 taps
%   over 2, 3, 5, 7 and 10 neurons, four seeds each and selected on the
%   validation season, put `tansig` at its clear minimum with 5: 49.63 MW on
%   validation against 53.68 at 7 and 64.10 at 10, and the tightest spread
%   across seeds of any count. `poslin` showed no minimum at all - 3, 5 and
%   10 neurons sat within 1.2 MW of one another on validation, well inside
%   its seed scatter - so 5 is carried over rather than chosen for it.
%
%   NAME-VALUE OPTIONS
%     Delays             Candidate tapped-delay lengths, in hours.
%                        Default [2 6 12 18 24 30 48].
%     Restarts           Networks fitted per delay and transfer function,
%                        from different initialisations. Default 5.
%     Retrain            Refit runs that already have a checkpoint.
%                        Default false, so an interrupted run resumes.
%
%   Writes one PNG figure, one results MAT file and one checkpoint per run
%   under train/load, all beside this file.
%
%   See also RUN_LOAD_DELAY_EXPERIMENT, NARXMODEL, NARXFORECAST.

arguments
    options.Delays (1,:) double {mustBePositive} = [2 6 12 18 24 30 48]
    options.Neurons (1,1) double {mustBePositive} = 5
    options.TrainK (1,1) double {mustBePositive} = 24
    options.PredictionHorizon (1,1) double {mustBePositive} = 24
    options.NumWindows (1,1) double {mustBePositive} = 2000
    options.IterPerRun (1,1) double {mustBePositive} = 300
    options.Restarts (1,1) double {mustBePositive} = 5
    options.RandomSeed (1,1) double = 2024
    options.TrainSplit = 'dry-train'
    options.ValidationSplit = 'dry-val'
    options.TestSplit = 'dry-test'
    options.Retrain (1,1) logical = true
end

scriptFolder = fileparts(mfilename('fullpath'));
projectFolder = fileparts(scriptFolder);
addpath(projectFolder);                                   % NARXmodel, secs2hms
addpath(scriptFolder);                                    % import_data_load
addpath(fullfile(projectFolder, 'Explainability'));       % narxForecast

transferFcns = {'tansig', 'poslin'};
transferFcns = {'poslin'};
% transferFcns = {'tansig'};
delays = sort(options.Delays);
neurons = options.Neurons;
trainK = options.TrainK;
predictionHorizon = options.PredictionHorizon;
nDelays = numel(delays);
nRestarts = options.Restarts;
nModels = numel(transferFcns);

% ---------------------------------------------------------------------
% 1. Prediction origins, drawn once and shared by every delay, every
%    restart and both transfer functions, so the error curves differ in
%    the model and nothing else.
% ---------------------------------------------------------------------
originDelay = max([delays, 24]);          % 24 so the naive forecast fits too
[~, ~, testOriginInfo] = sample_load_windows('Delay', originDelay, ...
    'PredictionHorizon', predictionHorizon, ...
    'NumWindows', options.NumWindows, 'Split', options.TestSplit);
origins = testOriginInfo.originIndex;
nWindows = testOriginInfo.nWindows;

[~, ~, validationOriginInfo] = sample_load_windows('Delay', originDelay, ...
    'PredictionHorizon', predictionHorizon, ...
    'NumWindows', options.NumWindows, 'Split', options.ValidationSplit);
validationOrigins = validationOriginInfo.originIndex;

% The daily-naive reference, yHat(T+k) = y(T+k-24), on both splits.
naiveMae = daily_naive_mae(options.TestSplit, origins, predictionHorizon);
naiveValidationMae = daily_naive_mae(options.ValidationSplit, ...
    validationOrigins, predictionHorizon);

fprintf('\nRELU AGAINST TANSIG ON HOURLY LOAD\n');
fprintf('  delays %s, %d neurons, horizon %d, %d restarts each\n', ...
    mat2str(delays), neurons, predictionHorizon, nRestarts);
fprintf('  fit on %s, choose on %s, score on %s (%d origins)\n\n', ...
    string(options.TrainSplit), string(options.ValidationSplit), ...
    string(options.TestSplit), nWindows);

% ---------------------------------------------------------------------
% 2. Every run of every delay, for both transfer functions.
% ---------------------------------------------------------------------
maeCurve = nan(predictionHorizon, nDelays, nRestarts, nModels);
maeLow = nan(predictionHorizon, nDelays, nRestarts, nModels);
maeHigh = nan(predictionHorizon, nDelays, nRestarts, nModels);
mapeCurve = nan(predictionHorizon, nDelays, nRestarts, nModels);
validationMae = nan(nDelays, nRestarts, nModels);
testMae = nan(nDelays, nRestarts, nModels);
trainingSeconds = nan(nDelays, nRestarts, nModels);
checkpointFile = strings(nDelays, nRestarts, nModels);
selectedRestart = zeros(nDelays, nModels);
parameterCount = zeros(1, nDelays);
deadUnits = nan(nDelays, nModels);

for delayIndex = 1:nDelays
    delay = delays(delayIndex);

    [pTest, yTestWindows] = sample_load_windows('Delay', delay, ...
        'PredictionHorizon', predictionHorizon, ...
        'Split', options.TestSplit, 'Origins', origins);
    [pValidation, yValidationWindows] = sample_load_windows('Delay', delay, ...
        'PredictionHorizon', predictionHorizon, ...
        'Split', options.ValidationSplit, 'Origins', validationOrigins);

    for modelIndex = 1:nModels
        transferFcn = transferFcns{modelIndex};
        checkpointFolder = checkpoint_folder(scriptFolder, delay, neurons, ...
            trainK, transferFcn);

        for restart = 1:nRestarts
            checkpointFile(delayIndex, restart, modelIndex) = ...
                checkpoint_file(checkpointFolder, transferFcn, restart);
            thisFile = checkpointFile(delayIndex, restart, modelIndex);

            if options.Retrain || ~isfile(thisFile)
                fprintf('TRAINING %-6s delay %2d, run %d of %d\n', ...
                    transferFcn, delay, restart, nRestarts);
                [p_pre, y_pre, ~, segmentStart] = import_data_load(options.TrainSplit);

                rng(options.RandomSeed + restart, 'twister');
                narx = NARXmodel(delay, neurons);
                narx.segmentStart = segmentStart;
                narx.hiddenTransferFcn = transferFcn;
                narx.trainAlg = 'trainlm';
                narx.earlyStoppage = true;
                narx.iterPerRun = options.IterPerRun;
                narx.iterAfterValley = 2;
                narx.iterAfterSeq = 10;
                narx.maxStep = 10;
                narx.initialTraining = 10;
                narx.zero_input_delay = false;

                trainingTimer = tic;
                narx = narx.train(p_pre, y_pre, trainK);
                trainingSeconds(delayIndex, restart, modelIndex) = ...
                    toc(trainingTimer);

                if ~isfolder(checkpointFolder)
                    mkdir(checkpointFolder);
                end
                save(thisFile, 'narx');
                fprintf('  trained in %s\n', ...
                    secs2hms(trainingSeconds(delayIndex, restart, modelIndex)));
            else
                checkpoint = load(thisFile, 'narx');
                narx = checkpoint.narx;
            end

            assert(strcmp(narx.narx.layers{1}.transferFcn, transferFcn), ...
                'Checkpoint %s has layer 1 = %s.', thisFile, ...
                narx.narx.layers{1}.transferFcn);
            parameterCount(delayIndex) = numel(getwb(narx.narx));

            [~, ~, validationMaeByHorizon] = narxForecast(narx, ...
                pValidation, yValidationWindows, predictionHorizon);
            [a2, yScored, maeByHorizon] = narxForecast(narx, pTest, ...
                yTestWindows, predictionHorizon);

            % The spread of the individual absolute errors behind each
            % MAE(k), so the figure can show how variable one forecast is
            % rather than only where the average sits.
            absoluteError = abs(yScored - a2);
            maeLow(:, delayIndex, restart, modelIndex) = ...
                prctile(absoluteError, 10, 2);
            maeHigh(:, delayIndex, restart, modelIndex) = ...
                prctile(absoluteError, 90, 2);
            mapeCurve(:, delayIndex, restart, modelIndex) = ...
                100 * mean(absoluteError ./ yScored, 2);

            maeCurve(:, delayIndex, restart, modelIndex) = maeByHorizon;
            validationMae(delayIndex, restart, modelIndex) = ...
                mean(validationMaeByHorizon);
            testMae(delayIndex, restart, modelIndex) = mean(maeByHorizon);
        end

        [~, selectedRestart(delayIndex, modelIndex)] = ...
            min(validationMae(delayIndex, :, modelIndex));
        chosen = selectedRestart(delayIndex, modelIndex);
        fprintf(['  %-6s delay %2d: run %d selected, %.2f MW on validation, ', ...
            '%.2f MW held out (%d of %d runs stable)\n'], ...
            transferFcn, delay, chosen, ...
            validationMae(delayIndex, chosen, modelIndex), ...
            testMae(delayIndex, chosen, modelIndex), ...
            sum(stable_runs(validationMae(delayIndex, :, modelIndex))), ...
            nRestarts);

        if strcmp(transferFcn, 'poslin')
            checkpoint = load(checkpointFile(delayIndex, chosen, modelIndex), 'narx');
            deadUnits(delayIndex, modelIndex) = count_dead_units( ...
                checkpoint.narx, pTest, yTestWindows);
        end
    end
end

% ---------------------------------------------------------------------
% 3. The selected run of every delay, and the best delay of each transfer
%    function, both chosen on validation.
% ---------------------------------------------------------------------
selectedMae = zeros(predictionHorizon, nDelays, nModels);
selectedLow = zeros(predictionHorizon, nDelays, nModels);
selectedHigh = zeros(predictionHorizon, nDelays, nModels);
for modelIndex = 1:nModels
    for delayIndex = 1:nDelays
        chosen = selectedRestart(delayIndex, modelIndex);
        selectedMae(:, delayIndex, modelIndex) = ...
            maeCurve(:, delayIndex, chosen, modelIndex);
        selectedLow(:, delayIndex, modelIndex) = ...
            maeLow(:, delayIndex, chosen, modelIndex);
        selectedHigh(:, delayIndex, modelIndex) = ...
            maeHigh(:, delayIndex, chosen, modelIndex);
    end
end

bestDelayIndex = zeros(1, nModels);
bestMae = zeros(predictionHorizon, nModels);
bestLower = zeros(predictionHorizon, nModels);
bestUpper = zeros(predictionHorizon, nModels);
nStableAtBest = zeros(1, nModels);

for modelIndex = 1:nModels
    selectedValidation = nan(1, nDelays);
    for delayIndex = 1:nDelays
        selectedValidation(delayIndex) = validationMae(delayIndex, ...
            selectedRestart(delayIndex, modelIndex), modelIndex);
    end
    [~, bestDelayIndex(modelIndex)] = min(selectedValidation);

    d = bestDelayIndex(modelIndex);
    bestMae(:, modelIndex) = maeCurve(:, d, ...
        selectedRestart(d, modelIndex), modelIndex);

    % The band spans the restarts whose rollout did not diverge. An unstable
    % one lands an order of magnitude out and would otherwise be the only
    % thing the axis showed.
    isStable = stable_runs(validationMae(d, :, modelIndex));
    nStableAtBest(modelIndex) = sum(isStable);
    stableCurves = reshape(maeCurve(:, d, isStable, modelIndex), ...
        predictionHorizon, []);
    bestLower(:, modelIndex) = min(stableCurves, [], 2);
    bestUpper(:, modelIndex) = max(stableCurves, [], 2);
end

% ---------------------------------------------------------------------
% 4. Report.
% ---------------------------------------------------------------------
fprintf('\n  Mean held-out MAE (MW) of every run\n');
for modelIndex = 1:nModels
    fprintf('\n  %s\n    delay |', transferFcns{modelIndex});
    fprintf('%9s', "run " + string(1:nRestarts));
    fprintf(' | selected\n');
    for delayIndex = 1:nDelays
        fprintf('    %5d |', delays(delayIndex));
        fprintf('%9.2f', testMae(delayIndex, :, modelIndex));
        fprintf(' | %d\n', selectedRestart(delayIndex, modelIndex));
    end
end

fprintf('\n  %-24s %12s %12s\n', '', transferFcns{1}, transferFcns{2});
fprintf('  %-24s %12d %12d\n', 'best delay (validation)', ...
    delays(bestDelayIndex(1)), delays(bestDelayIndex(2)));
fprintf('  %-24s %12.2f %12.2f\n', 'held-out MAE, mean', ...
    mean(bestMae(:, 1)), mean(bestMae(:, 2)));
fprintf('  %-24s %12.2f %12.2f\n', 'held-out MAE, k = 1', ...
    bestMae(1, 1), bestMae(1, 2));
fprintf('  %-24s %12.2f %12.2f\n', ...
    sprintf('held-out MAE, k = %d', predictionHorizon), ...
    bestMae(end, 1), bestMae(end, 2));
fprintf('  %-24s %12d %12d\n', 'stable runs at best', ...
    nStableAtBest(1), nStableAtBest(2));
fprintf('\n  daily-naive reference: %.2f MW\n', mean(naiveMae));
fprintf('  poslin is %+.1f%% against tansig on mean held-out MAE\n', ...
    100 * (mean(bestMae(:, 2)) - mean(bestMae(:, 1))) / mean(bestMae(:, 1)));
fprintf('  %d of %d ReLU units dead at the best ReLU delay\n', ...
    deadUnits(bestDelayIndex(2), 2), neurons);

figureFile = fullfile(scriptFolder, 'load_narx_relu_vs_tansig.png');
plot_comparison(figureFile, selectedMae, selectedLow, selectedHigh, ...
    naiveMae, transferFcns, delays, predictionHorizon, neurons, ...
    nWindows, options.TestSplit);

results = struct;
results.options = options;
results.transferFcns = transferFcns;
results.delays = delays;
results.horizon = (1:predictionHorizon).';
results.maeCurve = maeCurve;
results.maeLow = maeLow;
results.maeHigh = maeHigh;
results.mapeCurve = mapeCurve;
results.selectedMae = selectedMae;
results.selectedLow = selectedLow;
results.selectedHigh = selectedHigh;
results.validationMae = validationMae;
results.testMae = testMae;
results.selectedRestart = selectedRestart;
results.parameterCount = parameterCount;
results.deadUnits = deadUnits;
results.trainingSeconds = trainingSeconds;
results.checkpointFile = checkpointFile;
results.origins = origins;
results.validationOrigins = validationOrigins;
results.naiveMae = naiveMae;
results.naiveValidationMae = naiveValidationMae;
results.isStable = false(nDelays, nRestarts, nModels);
for modelIndex = 1:nModels
    for delayIndex = 1:nDelays
        results.isStable(delayIndex, :, modelIndex) = ...
            stable_runs(validationMae(delayIndex, :, modelIndex));
    end
end
results.bestDelayIndex = bestDelayIndex;
results.bestDelay = delays(bestDelayIndex);
results.bestMae = bestMae;
results.bestLower = bestLower;
results.bestUpper = bestUpper;
results.nStableAtBest = nStableAtBest;
results.figureFile = figureFile;

resultsFile = fullfile(scriptFolder, 'load_narx_relu_comparison_results.mat');
save(resultsFile, 'results');
fprintf('  Saved figure: %s\n  Saved values: %s\n', figureFile, resultsFile);
end

function isStable = stable_runs(validationRow)
%STABLE_RUNS Restarts whose closed-loop rollout did not diverge.
%
%   A diverged rollout lands an order of magnitude out, so a run counts as
%   stable when its validation error is within a factor of three of the best
%   run of that delay. The rule is scale free, so it separates a diverged
%   rollout from a delay that is merely a poor model, and it never rejects
%   the selected run.

isStable = validationRow <= 3 * min(validationRow);
end

function folder = checkpoint_folder(scriptFolder, delay, neurons, trainK, ...
    transferFcn)
%CHECKPOINT_FOLDER Where one delay's checkpoints live.
%
%   The tansig folders are the ones RUN_LOAD_DELAY_EXPERIMENT already
%   writes, so its checkpoints are reused rather than refitted.

name = sprintf('narx_%ddelay_%dneurons_%dsteps', delay, neurons, trainK);
if ~strcmp(transferFcn, 'tansig')
    name = sprintf('%s_%s', name, transferFcn);
end
folder = fullfile(scriptFolder, 'train', 'load', name);
end

function file = checkpoint_file(folder, transferFcn, restart)
%CHECKPOINT_FILE One run's checkpoint, named as the delay sweep names it.

if strcmp(transferFcn, 'tansig') && restart == 1
    file = fullfile(folder, 'narx_model.mat');
else
    file = fullfile(folder, sprintf('narx_model_run%d.mat', restart));
end
end

function mae = daily_naive_mae(split, origins, predictionHorizon)
%DAILY_NAIVE_MAE Error of yHat(T+k) = y(T+k-24) on the same origins.

[~, y] = import_data_load(split);
targetIndices = (1:predictionHorizon).' + origins;
mae = mean(abs(y(targetIndices) - y(targetIndices - 24)), 2);
end

function nDead = count_dead_units(narxModel, pWindows, yWindows)
%COUNT_DEAD_UNITS Layer-1 units that never leave zero on the scored windows.
%
%   A ReLU unit whose net input stays negative contributes nothing and
%   receives no gradient, so this counts how much of layer 1 is doing work.

sequenceLength = size(pWindows, 1);
nWindows = size(pWindows, 2);
pNormalized = (pWindows - narxModel.int_u) / narxModel.slope_u;
yNormalized = (yWindows - narxModel.int_y) / narxModel.slope_y;

pCells = mat2cell(pNormalized, ones(sequenceLength, 1), nWindows).';
yCells = mat2cell(yNormalized, ones(sequenceLength, 1), nWindows).';

% Reading a^1(t) out of a trained net: make layer 1 an output too, so that
% sim returns it alongside a^2(t). The weights are untouched.
probe = narxModel.narx;
probe.outputConnect = true(1, probe.numLayers);
[preparedP, initialInputState, initialLayerState] = ...
    preparets(probe, pCells, {}, yCells);
a1 = cell2mat(sim(probe, preparedP, initialInputState, initialLayerState));

nDead = sum(all(a1(1:probe.layers{1}.size, :) <= 0, 2));
end

function plot_comparison(figureFile, selectedMae, selectedLow, ...
    selectedHigh, naiveMae, transferFcns, delays, predictionHorizon, ...
    neurons, nWindows, testSplit)
%PLOT_COMPARISON Error against horizon for every delay and both f^1.
%
%   One line per delay per transfer function: the run each pair chose on the
%   validation season. The bars are the 10th to 90th percentile of the
%   individual absolute errors behind each MAE(k), so they say how variable
%   one forecast is, not how precisely the mean is known.
%
%   Fourteen series is past the number of hues that stay apart, so the two
%   variables are encoded separately: hue is the transfer function, and
%   within each hue a single light-to-dark ramp is the delay, which is
%   ordinal. Line style repeats the transfer function, so the two are never
%   told apart by colour alone.

figureHandle = figure('Color', 'white', 'Visible', 'off', ...
    'Position', [100, 100, 1400, 760]);
if exist('theme', 'file')
    theme(figureHandle, 'light');
end
axesHandle = axes(figureHandle);
hold(axesHandle, 'on');

nDelays = numel(delays);
nModels = numel(transferFcns);
horizons = (1:predictionHorizon).';

rampEnds = cat(3, [0.45 0.66 0.81; 0.03 0.19 0.42], ...   % tansig, blue
                  [0.99 0.55 0.24; 0.50 0.15 0.02]);      % poslin, orange
lineStyles = {'-', '--'};
markers = {'o', 's'};

% Fourteen series of bars at the same 24 horizons would sit on top of one
% another, so each is nudged along x.
modelOffset = 0.26;
delayOffset = 0.055;
seriesX = @(delayIndex, modelIndex) horizons ...
    + (modelIndex - 1.5) * modelOffset ...
    + (delayIndex - (nDelays + 1) / 2) * delayOffset;

colours = zeros(nDelays, 3, nModels);
for modelIndex = 1:nModels
    for delayIndex = 1:nDelays
        t = (delayIndex - 1) / (nDelays - 1);
        colours(delayIndex, :, modelIndex) = ...
            (1 - t) * rampEnds(1, :, modelIndex) + t * rampEnds(2, :, modelIndex);
    end
end

% Bars first and washed out, so the means stay readable on top of them.
for modelIndex = 1:nModels
    for delayIndex = 1:nDelays
        colour = colours(delayIndex, :, modelIndex);
        errorbar(axesHandle, seriesX(delayIndex, modelIndex), ...
            selectedMae(:, delayIndex, modelIndex), ...
            selectedMae(:, delayIndex, modelIndex) - ...
                selectedLow(:, delayIndex, modelIndex), ...
            selectedHigh(:, delayIndex, modelIndex) - ...
                selectedMae(:, delayIndex, modelIndex), ...
            'LineStyle', 'none', 'Marker', 'none', ...
            'Color', colour + 0.72 * (1 - colour), ...
            'LineWidth', 0.4, 'CapSize', 0, 'HandleVisibility', 'off');
    end
end

plot(axesHandle, horizons, naiveMae, '-', 'Color', [0.45, 0.45, 0.45], ...
    'LineWidth', 2.2, 'DisplayName', 'Daily naive');

for modelIndex = 1:nModels
    for delayIndex = 1:nDelays
        plot(axesHandle, seriesX(delayIndex, modelIndex), ...
            selectedMae(:, delayIndex, modelIndex), ...
            'Color', colours(delayIndex, :, modelIndex), ...
            'LineStyle', lineStyles{modelIndex}, ...
            'Marker', markers{modelIndex}, 'MarkerSize', 3, ...
            'MarkerFaceColor', colours(delayIndex, :, modelIndex), ...
            'LineWidth', 1.3, ...
            'DisplayName', sprintf('%2d taps, %s', delays(delayIndex), ...
            transferFcns{modelIndex}));
    end
end

hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
axesHandle.GridColor = [0.75, 0.75, 0.75];
axesHandle.GridAlpha = 0.5;
axesHandle.LineWidth = 0.5;
xlim(axesHandle, [0.4, predictionHorizon + 0.6]);
xticks(axesHandle, [1, 4:4:predictionHorizon]);
ylim(axesHandle, [0, 1.02 * max(selectedHigh(:))]);
xlabel(axesHandle, 'Prediction horizon, $k$ (hours)', 'Interpreter', 'latex');
ylabel(axesHandle, 'MAE of $a^2(T+k)=\hat{y}(T+k)$ (MW)', ...
    'Interpreter', 'latex');
title(axesHandle, ['Hidden-layer transfer function across the delay sweep: ', ...
    'ReLU against tansig'], 'FontWeight', 'bold');
subtitle(axesHandle, sprintf(['%d neurons; one line per delay per $f^1$, ', ...
    'the run selected on validation. Blue solid: tansig. ', ...
    'Orange dashed: poslin. Light to dark is 2 to 48 taps.\n', ...
    'Bars are the 10th to 90th percentile of $|e|$ over the %s ', ...
    'held-out origins of the %s split.'], neurons, ...
    format_number(nWindows), string(testSplit)), 'Interpreter', 'latex');
legend(axesHandle, 'Location', 'northwest', 'NumColumns', 3, ...
    'FontSize', 7, 'EdgeColor', [0.70, 0.70, 0.70]);

exportgraphics(figureHandle, figureFile, 'Resolution', 300);
close(figureHandle);
end

function text = format_number(value)
%FORMAT_NUMBER Add thousands separators without locale dependence.
text = regexprep(sprintf('%d', value), '(?<!^)(?=(\d{3})+$)', ',');
end
