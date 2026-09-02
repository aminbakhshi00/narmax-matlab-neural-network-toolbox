function results = run_load_relu_comparison(options)
%RUN_LOAD_RELU_COMPARISON Does a ReLU hidden layer forecast load any better?
%
%   RESULTS = RUN_LOAD_RELU_COMPARISON() refits the 24-tap load model of
%   RUN_LOAD_DELAY_EXPERIMENT with one thing changed, layer 1's transfer
%   function, from the default tansig to poslin (ReLU), and reports the
%   held-out error of the two side by side.
%
%   Everything else is held fixed on purpose: the same delay, neurons,
%   epoch budget and horizon curriculum, the same restart seeds
%   RandomSeed + r, the same validation split to choose the restart on,
%   and the same held-out prediction origins to score on, read back from
%   the sweep's results file. The tansig side is not refitted; its
%   checkpoints and error curve are taken from that file, so the two
%   differ in f^1 and nothing else.
%
%   NAME-VALUE OPTIONS
%     Delay              Tapped-delay length. Default 24 hours.
%     HiddenTransferFcn  Layer 1 transfer function. Default 'poslin'.
%     Restarts           Networks fitted, from different initialisations.
%                        Default 3, chosen on the validation split.
%     Retrain            Refit runs that already have a checkpoint.
%                        Default false, so an interrupted run resumes.
%
%   Writes one PNG figure, one results MAT file and one checkpoint per
%   restart under train/load, all beside this file.
%
%   See also RUN_LOAD_DELAY_EXPERIMENT, NARXMODEL, NARXFORECAST.

arguments
    options.Delay (1,1) double {mustBePositive} = 24
    options.HiddenTransferFcn (1,:) char = 'poslin'
    options.Neurons (1,1) double {mustBePositive} = 10
    options.TrainK (1,1) double {mustBePositive} = 24
    options.PredictionHorizon (1,1) double {mustBePositive} = 24
    options.IterPerRun (1,1) double {mustBePositive} = 300
    options.Restarts (1,1) double {mustBePositive} = 3
    options.RandomSeed (1,1) double = 2024
    options.TrainSplit = 'dry-train'
    options.ValidationSplit = 'dry-val'
    options.TestSplit = 'dry-test'
    options.Retrain (1,1) logical = false
end

scriptFolder = fileparts(mfilename('fullpath'));
projectFolder = fileparts(scriptFolder);
addpath(projectFolder);                                   % NARXmodel, secs2hms
addpath(scriptFolder);                                    % import_data_load
addpath(fullfile(projectFolder, 'Explainability'));       % narxForecast

delay = options.Delay;
neurons = options.Neurons;
trainK = options.TrainK;
predictionHorizon = options.PredictionHorizon;
nRestarts = options.Restarts;

% ---------------------------------------------------------------------
% 1. The tansig side, and the origins it was scored on. Reusing them is
%    what makes the two columns of the table comparable.
% ---------------------------------------------------------------------
sweepFile = fullfile(scriptFolder, 'load_narx_delay_experiment_results.mat');
assert(isfile(sweepFile), ['Run RUN_LOAD_DELAY_EXPERIMENT first: %s is ', ...
    'where the tansig baseline and the held-out origins come from.'], sweepFile);
sweep = load(sweepFile, 'results');
sweep = sweep.results;

delayIndex = find(sweep.delays == delay, 1);
assert(~isempty(delayIndex), ...
    'The sweep holds delays %s, not %d.', mat2str(sweep.delays), delay);
assert(sweep.options.PredictionHorizon == predictionHorizon, ...
    'The sweep scored horizon %d, not %d.', ...
    sweep.options.PredictionHorizon, predictionHorizon);

origins = sweep.origins;
validationOrigins = sweep.validationOrigins;
tansigMae = sweep.mae(:, delayIndex);                     % selected run
tansigValidationMae = sweep.validationMaeByRestart(delayIndex, :);
tansigTestMae = sweep.testMaeByRestart(delayIndex, :);
tansigSelected = sweep.selectedRestart(delayIndex);
naiveMae = sweep.naiveMae;

fprintf('\nRELU AGAINST TANSIG ON HOURLY LOAD\n');
fprintf('  %d taps, %d neurons, horizon %d, %d restarts\n', ...
    delay, neurons, predictionHorizon, nRestarts);
fprintf('  fit on %s, choose on %s, score on %s (%d origins)\n\n', ...
    string(options.TrainSplit), string(options.ValidationSplit), ...
    string(options.TestSplit), numel(origins));

% ---------------------------------------------------------------------
% 2. The ReLU restarts, on those same origins.
% ---------------------------------------------------------------------
[pTest, yTestWindows] = sample_load_windows('Delay', delay, ...
    'PredictionHorizon', predictionHorizon, ...
    'Split', options.TestSplit, 'Origins', origins);
[pValidation, yValidationWindows] = sample_load_windows('Delay', delay, ...
    'PredictionHorizon', predictionHorizon, ...
    'Split', options.ValidationSplit, 'Origins', validationOrigins);

checkpointFolder = fullfile(scriptFolder, 'train', 'load', ...
    sprintf('narx_%ddelay_%dneurons_%dsteps_%s', delay, neurons, trainK, ...
    options.HiddenTransferFcn));

reluMaeByRestart = zeros(predictionHorizon, nRestarts);
reluValidationMae = nan(1, nRestarts);
reluTestMae = nan(1, nRestarts);
trainingSeconds = nan(1, nRestarts);
checkpointFile = strings(1, nRestarts);

for restart = 1:nRestarts
    checkpointFile(restart) = fullfile(checkpointFolder, ...
        sprintf('narx_model_run%d.mat', restart));

    if options.Retrain || ~isfile(checkpointFile(restart))
        fprintf('TRAINING %s MODEL: delay = %d, run %d of %d\n', ...
            upper(options.HiddenTransferFcn), delay, restart, nRestarts);
        [p_pre, y_pre] = import_data_load(options.TrainSplit);

        rng(options.RandomSeed + restart, 'twister');
        narx = NARXmodel(delay, neurons);
        narx.hiddenTransferFcn = options.HiddenTransferFcn;
        narx.trainAlg = 'trainlm';
        narx.earlyStoppage = true;
        narx.iterPerRun = options.IterPerRun;
        narx.iterAfterValley = 100;
        narx.iterAfterSeq = 50;
        narx.maxStep = 10;
        narx.initialTraining = 10;
        narx.zero_input_delay = false;

        trainingTimer = tic;
        narx = narx.train(p_pre, y_pre, trainK);
        trainingSeconds(restart) = toc(trainingTimer);

        if ~isfolder(checkpointFolder)
            mkdir(checkpointFolder);
        end
        save(checkpointFile(restart), 'narx');
        fprintf('  trained in %s\n', secs2hms(trainingSeconds(restart)));
    else
        checkpoint = load(checkpointFile(restart), 'narx');
        narx = checkpoint.narx;
        fprintf('REUSING CHECKPOINT: run %d of %d\n', restart, nRestarts);
    end

    assert(strcmp(narx.narx.layers{1}.transferFcn, options.HiddenTransferFcn), ...
        'Checkpoint %s has layer 1 = %s.', checkpointFile(restart), ...
        narx.narx.layers{1}.transferFcn);

    [~, ~, validationMaeByHorizon] = narxForecast(narx, pValidation, ...
        yValidationWindows, predictionHorizon);
    [a2, yScored, maeByHorizon] = narxForecast(narx, pTest, ...
        yTestWindows, predictionHorizon);

    reluMaeByRestart(:, restart) = maeByHorizon;
    reluValidationMae(restart) = mean(validationMaeByHorizon);
    reluTestMae(restart) = mean(maeByHorizon);
    fprintf('  run %d: %.2f MW mean MAE on validation, %.2f MW held out\n', ...
        restart, reluValidationMae(restart), reluTestMae(restart));

    if reluValidationMae(restart) == min(reluValidationMae)
        reluSelected = restart;
        reluMae = maeByHorizon;
        reluMape = 100 * mean(abs(yScored - a2) ./ yScored, 2);
        reluDeadUnits = count_dead_units(narx, pTest, yTestWindows);
    end
end

% ---------------------------------------------------------------------
% 3. The comparison.
% ---------------------------------------------------------------------
tansigMean = mean(tansigMae);
reluMean = mean(reluMae);
changePercent = 100 * (reluMean - tansigMean) / tansigMean;

fprintf('\n  %-22s %10s %10s\n', '', 'tansig', options.HiddenTransferFcn);
fprintf('  %-22s %10d %10d\n', 'selected run', tansigSelected, reluSelected);
fprintf('  %-22s %10.2f %10.2f\n', 'validation MAE (MW)', ...
    tansigValidationMae(tansigSelected), reluValidationMae(reluSelected));
fprintf('  %-22s %10.2f %10.2f\n', 'held-out MAE, k = 1', ...
    tansigMae(1), reluMae(1));
fprintf('  %-22s %10.2f %10.2f\n', ...
    sprintf('held-out MAE, k = %d', predictionHorizon), ...
    tansigMae(end), reluMae(end));
fprintf('  %-22s %10.2f %10.2f\n', 'held-out MAE, mean', ...
    tansigMean, reluMean);
fprintf('  %-22s %10d %10d\n', 'runs that diverged', ...
    sum(tansigTestMae > mean(naiveMae)), sum(reluTestMae > mean(naiveMae)));
fprintf('\n  daily-naive reference: %.2f MW\n', mean(naiveMae));
fprintf('  %s is %+.1f%% against tansig on mean held-out MAE\n', ...
    options.HiddenTransferFcn, changePercent);
fprintf('  %d of %d ReLU units are dead on every held-out window\n', ...
    reluDeadUnits, neurons);

figureFile = fullfile(scriptFolder, 'load_narx_relu_vs_tansig.png');
plot_comparison(figureFile, tansigMae, reluMae, naiveMae, ...
    tansigTestMae, reluTestMae, predictionHorizon, delay, neurons, ...
    options.HiddenTransferFcn, numel(origins), options.TestSplit);

results = struct;
results.options = options;
results.horizon = (1:predictionHorizon).';
results.tansigMae = tansigMae;
results.tansigValidationMaeByRestart = tansigValidationMae;
results.tansigTestMaeByRestart = tansigTestMae;
results.tansigSelectedRestart = tansigSelected;
results.reluMae = reluMae;
results.reluMape = reluMape;
results.reluMaeByRestart = reluMaeByRestart;
results.reluValidationMaeByRestart = reluValidationMae;
results.reluTestMaeByRestart = reluTestMae;
results.reluSelectedRestart = reluSelected;
results.reluDeadUnits = reluDeadUnits;
results.naiveMae = naiveMae;
results.changePercent = changePercent;
results.trainingSeconds = trainingSeconds;
results.checkpointFile = checkpointFile;
results.figureFile = figureFile;

resultsFile = fullfile(scriptFolder, 'load_narx_relu_comparison_results.mat');
save(resultsFile, 'results');
fprintf('  Saved figure: %s\n  Saved values: %s\n', figureFile, resultsFile);
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

function plot_comparison(figureFile, tansigMae, reluMae, naiveMae, ...
    tansigTestMae, reluTestMae, predictionHorizon, delay, neurons, ...
    hiddenTransferFcn, nWindows, testSplit)
%PLOT_COMPARISON The two error curves, and the spread over restarts.

figureHandle = figure('Color', 'white', 'Visible', 'off', ...
    'Position', [100, 100, 1100, 450]);
if exist('theme', 'file')
    theme(figureHandle, 'light');
end
tiles = tiledlayout(figureHandle, 1, 2, 'Padding', 'compact', ...
    'TileSpacing', 'compact');

axesHandle = nexttile(tiles);
hold(axesHandle, 'on');
plot(axesHandle, 1:predictionHorizon, tansigMae, '-o', ...
    'Color', [0.0000, 0.4470, 0.7410], 'MarkerSize', 4, 'LineWidth', 1.8, ...
    'DisplayName', 'tansig');
plot(axesHandle, 1:predictionHorizon, reluMae, '-s', ...
    'Color', [0.8500, 0.3250, 0.0980], 'MarkerSize', 4, 'LineWidth', 1.8, ...
    'DisplayName', hiddenTransferFcn);
plot(axesHandle, 1:predictionHorizon, naiveMae, '--', ...
    'Color', [0.35, 0.35, 0.35], 'LineWidth', 1.4, ...
    'DisplayName', 'Daily naive');
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
xlim(axesHandle, [1, predictionHorizon]);
xlabel(axesHandle, 'Prediction horizon, $k$ (hours)', 'Interpreter', 'latex');
ylabel(axesHandle, 'MAE of $a^2(T+k)$ (MW)', 'Interpreter', 'latex');
title(axesHandle, 'Error along the 24-hour curve');
subtitle(axesHandle, sprintf('mean MAE %.2f MW against %.2f MW', ...
    mean(reluMae), mean(tansigMae)));
legend(axesHandle, 'Location', 'southeast', 'EdgeColor', [0.35, 0.35, 0.35]);

% Restarts matter more than the transfer function here, so show them all.
axesHandle = nexttile(tiles);
hold(axesHandle, 'on');
axisTop = 1.35 * max([tansigMae(:); reluMae(:); naiveMae(:)]);
plot_restarts(axesHandle, 1, tansigTestMae, axisTop, [0.0000, 0.4470, 0.7410]);
plot_restarts(axesHandle, 2, reluTestMae, axisTop, [0.8500, 0.3250, 0.0980]);
yline(axesHandle, mean(naiveMae), '--', 'Color', [0.35, 0.35, 0.35], ...
    'LineWidth', 1.0, 'DisplayName', 'Daily naive');
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
xlim(axesHandle, [0.5, 2.5]);
ylim(axesHandle, [0, axisTop]);
xticks(axesHandle, [1, 2]);
xticklabels(axesHandle, {'tansig', hiddenTransferFcn});
ylabel(axesHandle, 'Mean held-out MAE (MW)');
title(axesHandle, 'Every restart, and the one selected');
subtitle(axesHandle, 'runs above the axis rolled out unstably (arrows)');
legend(axesHandle, 'Location', 'northwest', 'EdgeColor', [0.35, 0.35, 0.35]);

title(tiles, sprintf(['Hidden-layer transfer function on the %d-tap load ', ...
    'model: %s against tansig'], delay, hiddenTransferFcn), ...
    'FontWeight', 'bold');
subtitle(tiles, sprintf(['%d neurons, %d restarts, identical seeds and ', ...
    'origins; %d origins from the %s split'], neurons, ...
    numel(tansigTestMae), nWindows, string(testSplit)));
exportgraphics(figureHandle, figureFile, 'Resolution', 300);
close(figureHandle);
end

function plot_restarts(axesHandle, column, testMae, axisTop, colour)
%PLOT_RESTARTS One column of restart outcomes, unstable ones pinned on top.

onScale = testMae <= axisTop;
jitter = linspace(-0.12, 0.12, numel(testMae));
plot(axesHandle, column + jitter(onScale), testMae(onScale), 'o', ...
    'Color', colour, 'MarkerSize', 7, 'LineWidth', 1.4, ...
    'HandleVisibility', 'off');
plot(axesHandle, column + jitter(~onScale), ...
    0.97 * axisTop * ones(1, sum(~onScale)), '^', 'Color', colour, ...
    'MarkerSize', 6, 'LineWidth', 1.2, 'HandleVisibility', 'off');
plot(axesHandle, column, min(testMae), 'o', 'MarkerSize', 11, ...
    'MarkerEdgeColor', colour, 'LineWidth', 1.8, 'HandleVisibility', 'off');
end
