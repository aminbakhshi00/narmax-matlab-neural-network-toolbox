function results = run_load_delay_experiment(options)
%RUN_LOAD_DELAY_EXPERIMENT Read the NARX tapped-delay length off the data.
%
%   RESULTS = RUN_LOAD_DELAY_EXPERIMENT() answers one question: how many
%   hourly taps does the tapped delay line of a NARX load model need? It
%   answers it in two independent ways, and reports both.
%
%   1. From the series alone. The sample autocorrelation of the demand and
%      its mean daily profile show the 24-hour period directly, before any
%      network is fitted (figure 1).
%
%   2. From held-out forecast error. One NARXmodel is trained per candidate
%      delay on the same season window, and each is scored by its
%      closed-loop 24-hour ahead error on the same held-out prediction
%      origins one year later (figures 2 and 3). The delay that wins is
%      measured, not assumed.
%
%   Every checkpoint is trained by the unmodified NARXmodel class with the
%   settings of TRAIN_NARX_LOAD, so the sweep differs from that script in
%   the delay only. Delays are compared on identical origins, so the error
%   curves differ only in the model. A daily-naive forecast,
%   yHat(T+k) = y(T+k-24), is carried through as the reference every load
%   forecast has to beat, and is itself a statement of the 24-hour period.
%
%   NAME-VALUE OPTIONS
%     Delays             Candidate tapped-delay lengths, in hours.
%                        Default [2 6 12 18 24 30 36 48].
%     Neurons            S^1, neurons in layer 1. Default 10.
%     TrainK             Training prediction horizon. Default 24 hours.
%     PredictionHorizon  H used for scoring. Default 24 hours.
%     NumWindows         Held-out prediction origins. Default 2000.
%     IterPerRun         Epoch budget per horizon step. Default 300, the
%                        same for every delay so the sweep is comparable.
%     Restarts           Networks fitted per delay, from different random
%                        initialisations. Default 3. The one with the
%                        lowest error on the training window is reported.
%     RandomSeed         Seed of restart r is RandomSeed + r. Default 2024.
%     TrainSplit         Split to fit on. Default 'dry-train'.
%     ValidationSplit    Split the restarts are chosen on, never fitted on
%                        and never reported. Default 'dry-val'.
%     TestSplit          Held-out split to score on. Default 'dry-test'.
%     Retrain            Refit delays that already have a checkpoint.
%                        Default false, so an interrupted sweep resumes.
%
%   Writes three PNG figures, one results MAT file and one checkpoint per
%   delay under train/load, all beside this file.
%
%   See also IMPORT_DATA_LOAD, SAMPLE_LOAD_WINDOWS, TRAIN_NARX_LOAD.

arguments
    options.Delays (1,:) double {mustBePositive} = [2 6 12 18 24 30 36 48]
    options.Neurons (1,1) double {mustBePositive} = 10
    options.TrainK (1,1) double {mustBePositive} = 24
    options.PredictionHorizon (1,1) double {mustBePositive} = 24
    options.NumWindows (1,1) double {mustBePositive} = 2000
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

delays = sort(options.Delays);
neurons = options.Neurons;
trainK = options.TrainK;
predictionHorizon = options.PredictionHorizon;
nDelays = numel(delays);

% ---------------------------------------------------------------------
% 1. What the series says, before any network is fitted.
% ---------------------------------------------------------------------
[~, yTrain, tTrain] = import_data_load(options.TrainSplit);
maxLag = 192;                                             % eight days
acfLag = (0:maxLag).';
acf = sample_autocorrelation(yTrain, maxLag);
dailyLags = 24:24:maxLag;

% The lag of the first local maximum is the period the series repeats on.
% It is read off the series, not assumed to be 24.
isLocalPeak = [false; acf(2:end-1) > acf(1:end-2) & acf(2:end-1) > acf(3:end); false];
peakLags = acfLag(isLocalPeak);
firstPeakLag = peakLags(1);

hourOfDay = hour(tTrain);
isWeekend = ismember(day(tTrain, 'dayofweek'), [1 7]);    % Sunday, Saturday
weekdayProfile = accumarray(hourOfDay(~isWeekend) + 1, ...
    yTrain(~isWeekend), [24 1], @mean);
weekendProfile = accumarray(hourOfDay(isWeekend) + 1, ...
    yTrain(isWeekend), [24 1], @mean);

% ---------------------------------------------------------------------
% 2. Held-out prediction origins, drawn once and shared by every delay so
%    that the error curves below differ in the model and nothing else.
% ---------------------------------------------------------------------
originDelay = max([delays, 24]);          % 24 so the naive forecast fits too
[~, ~, originInfo] = sample_load_windows('Delay', originDelay, ...
    'PredictionHorizon', predictionHorizon, ...
    'NumWindows', options.NumWindows, 'Split', options.TestSplit);
origins = originInfo.originIndex;
nWindows = originInfo.nWindows;

fprintf('\nDELAY EXPERIMENT ON HOURLY LOAD\n');
fprintf('  fit on   %s (%d hours), horizon %d\n', ...
    string(options.TrainSplit), numel(yTrain), trainK);
fprintf('  choose on %s\n', string(options.ValidationSplit));
fprintf('  score on %s, %d held-out origins, horizon %d\n', ...
    string(options.TestSplit), nWindows, predictionHorizon);
fprintf('  delays   %s\n\n', mat2str(delays));

% ---------------------------------------------------------------------
% 3. The daily-naive reference: yesterday's value at the same hour.
% ---------------------------------------------------------------------
[~, yTest, tTest] = import_data_load(options.TestSplit);
horizonSteps = (1:predictionHorizon).';
targetIndices = horizonSteps + origins;
yMeasured = yTest(targetIndices);
naiveForecast = yTest(targetIndices - 24);
naiveMae = mean(abs(yMeasured - naiveForecast), 2);
naiveMape = 100 * mean(abs(yMeasured - naiveForecast) ./ yMeasured, 2);

% ---------------------------------------------------------------------
% 4. Restarts per delay, then the closed-loop error curve of each.
%
%    A closed-loop NARX fitted from one random initialisation sometimes
%    lands on a solution whose rollout is unstable, so a single run per
%    delay would compare initialisations as much as delays. Several runs
%    are fitted per delay, as in Kelley's comparison of these models, and
%    the run with the lowest error on the validation season is the one
%    carried forward. That season is a third dry season, neither fitted on
%    nor reported; the held-out test season is never consulted to choose a
%    model.
% ---------------------------------------------------------------------
nRestarts = options.Restarts;
mae = zeros(predictionHorizon, nDelays);
mape = zeros(predictionHorizon, nDelays);
maeByRestart = zeros(predictionHorizon, nDelays, nRestarts);
validationMaeByRestart = nan(nDelays, nRestarts);
testMaeByRestart = nan(nDelays, nRestarts);
selectedRestart = zeros(1, nDelays);
checkpointFile = strings(nDelays, nRestarts);
trainingSeconds = nan(nDelays, nRestarts);
parameterCount = zeros(1, nDelays);

[~, ~, validationOriginInfo] = sample_load_windows('Delay', originDelay, ...
    'PredictionHorizon', predictionHorizon, ...
    'NumWindows', options.NumWindows, 'Split', options.ValidationSplit);
validationOrigins = validationOriginInfo.originIndex;

for delayIndex = 1:nDelays
    delay = delays(delayIndex);
    checkpointFolder = fullfile(scriptFolder, 'train', 'load', ...
        sprintf('narx_%ddelay_%dneurons_%dsteps', delay, neurons, trainK));

    [pTest, yTestWindows] = sample_load_windows('Delay', delay, ...
        'PredictionHorizon', predictionHorizon, ...
        'Split', options.TestSplit, 'Origins', origins);
    [pValidation, yValidationWindows] = sample_load_windows('Delay', delay, ...
        'PredictionHorizon', predictionHorizon, ...
        'Split', options.ValidationSplit, 'Origins', validationOrigins);

    for restart = 1:nRestarts
        if restart == 1
            % Restart 1 is the checkpoint TRAIN_NARX_LOAD itself writes.
            checkpointFile(delayIndex, restart) = ...
                fullfile(checkpointFolder, 'narx_model.mat');
        else
            checkpointFile(delayIndex, restart) = fullfile(checkpointFolder, ...
                sprintf('narx_model_run%d.mat', restart));
        end

        if options.Retrain || ~isfile(checkpointFile(delayIndex, restart))
            fprintf('TRAINING NARX MODEL: delay = %d, run %d of %d\n', ...
                delay, restart, nRestarts);
            [p_pre, y_pre] = import_data_load(options.TrainSplit);

            rng(options.RandomSeed + restart, 'twister');
            narx = NARXmodel(delay, neurons);
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
            trainingSeconds(delayIndex, restart) = toc(trainingTimer);

            if ~isfolder(checkpointFolder)
                mkdir(checkpointFolder);
            end
            save(checkpointFile(delayIndex, restart), 'narx');
            fprintf('  trained in %s\n', ...
                secs2hms(trainingSeconds(delayIndex, restart)));
        else
            checkpoint = load(checkpointFile(delayIndex, restart), 'narx');
            narx = checkpoint.narx;
            fprintf('REUSING CHECKPOINT: delay = %d, run %d of %d\n', ...
                delay, restart, nRestarts);
        end

        parameterCount(delayIndex) = numel(getwb(narx.narx));

        [~, ~, validationMaeByHorizon] = narxForecast(narx, pValidation, ...
            yValidationWindows, predictionHorizon);
        [a2, yScored, maeByHorizon] = narxForecast(narx, pTest, ...
            yTestWindows, predictionHorizon);
        assert(max(abs(yScored - yMeasured), [], 'all') < 1e-9, ...
            'The scored targets differ from the shared held-out origins.');

        maeByRestart(:, delayIndex, restart) = maeByHorizon;
        validationMaeByRestart(delayIndex, restart) = mean(validationMaeByHorizon);
        testMaeByRestart(delayIndex, restart) = mean(maeByHorizon);
        fprintf('  delay %2d run %d: %d weights, %.2f MW mean MAE on validation, %.2f MW held out\n', ...
            delay, restart, parameterCount(delayIndex), ...
            validationMaeByRestart(delayIndex, restart), ...
            testMaeByRestart(delayIndex, restart));

        if validationMaeByRestart(delayIndex, restart) == ...
                min(validationMaeByRestart(delayIndex, :))
            selectedRestart(delayIndex) = restart;
            mae(:, delayIndex) = maeByHorizon;
            mape(:, delayIndex) = 100 * mean(abs(yScored - a2) ./ yScored, 2);
        end
    end

    fprintf('  delay %2d: run %d selected, MAE %.2f MW at k = 1, %.2f MW at k = %d, %.2f MW mean\n\n', ...
        delay, selectedRestart(delayIndex), mae(1, delayIndex), ...
        mae(end, delayIndex), predictionHorizon, mean(mae(:, delayIndex)));
end

% ---------------------------------------------------------------------
% 5. The delay that wins, on the mean error over the horizon the models
%    were trained to minimise.
% ---------------------------------------------------------------------
meanMae = mean(mae, 1);
[bestMeanMae, bestIndex] = min(meanMae);
bestDelay = delays(bestIndex);

shortestIndex = 1;
[pBest, yBest] = sample_load_windows('Delay', bestDelay, ...
    'PredictionHorizon', predictionHorizon, ...
    'Split', options.TestSplit, 'Origins', origins);
bestCheckpoint = load(checkpointFile(bestIndex, selectedRestart(bestIndex)), 'narx');
a2Best = narxForecast(bestCheckpoint.narx, pBest, yBest, predictionHorizon);

% Error by the hour of day being predicted, for the daily structure that
% survives in the residuals.
targetHour = hour(tTest(targetIndices));
absoluteErrorBest = abs(yMeasured - a2Best);
absoluteErrorNaive = abs(yMeasured - naiveForecast);
maeByHourBest = accumarray(targetHour(:) + 1, absoluteErrorBest(:), [24 1], @mean);
maeByHourNaive = accumarray(targetHour(:) + 1, absoluteErrorNaive(:), [24 1], @mean);

% ---------------------------------------------------------------------
% 6. Figures.
% ---------------------------------------------------------------------
figureFiles = strings(1, 3);
figureFiles(1) = fullfile(scriptFolder, 'load_daily_periodicity.png');
figureFiles(2) = fullfile(scriptFolder, 'load_narx_mae_vs_delay.png');
figureFiles(3) = fullfile(scriptFolder, 'load_narx_error_by_hour.png');

plot_periodicity(figureFiles(1), acfLag, acf, dailyLags, firstPeakLag, ...
    weekdayProfile, weekendProfile, options.TrainSplit, numel(yTrain));
plot_delay_sweep(figureFiles(2), delays, mae, meanMae, testMaeByRestart, ...
    naiveMae, bestDelay, bestIndex, predictionHorizon, neurons, trainK, ...
    nWindows, options.TestSplit);
plot_error_by_hour(figureFiles(3), maeByHourBest, maeByHourNaive, ...
    mae(:, bestIndex), mae(:, shortestIndex), naiveMae, bestDelay, ...
    delays(shortestIndex), predictionHorizon);

% ---------------------------------------------------------------------
% 7. Results, saved whole so the explainability step can start from here.
% ---------------------------------------------------------------------
results = struct;
results.options = options;
results.delays = delays;
results.checkpointFile = checkpointFile;
results.selectedRestart = selectedRestart;
results.parameterCount = parameterCount;
results.trainingSeconds = trainingSeconds;
results.maeByRestart = maeByRestart;
results.validationMaeByRestart = validationMaeByRestart;
results.testMaeByRestart = testMaeByRestart;
results.validationOrigins = validationOrigins;
results.horizon = horizonSteps;
results.mae = mae;
results.mape = mape;
results.meanMae = meanMae;
results.naiveMae = naiveMae;
results.naiveMape = naiveMape;
results.bestDelay = bestDelay;
results.bestIndex = bestIndex;
results.bestMeanMae = bestMeanMae;
results.acfLag = acfLag;
results.acf = acf;
results.acfAtDailyLags = acf(dailyLags + 1);
results.firstPeakLag = firstPeakLag;
results.peakLags = peakLags;
results.weekdayProfile = weekdayProfile;
results.weekendProfile = weekendProfile;
results.origins = origins;
results.originDatetime = originInfo.originDatetime;
results.maeByHourBest = maeByHourBest;
results.maeByHourNaive = maeByHourNaive;
results.figureFiles = figureFiles;

resultsFile = fullfile(scriptFolder, 'load_narx_delay_experiment_results.mat');
save(resultsFile, 'results');

fprintf('\nFirst local autocorrelation peak: lag %d h (r = %.4f)\n', ...
    firstPeakLag, acf(firstPeakLag + 1));
fprintf('Autocorrelation at 24 h: %.4f, at 168 h: %.4f\n', ...
    acf(25), acf(169));
fprintf('Daily-naive reference: %.2f MW mean MAE (%.2f%% MAPE)\n', ...
    mean(naiveMae), mean(naiveMape));
fprintf('Best delay: %d hours, %.2f MW mean MAE (%.2f%% MAPE)\n', ...
    bestDelay, bestMeanMae, mean(mape(:, bestIndex)));
fprintf('Saved figures: %s\n', strjoin(cellstr(figureFiles), ', '));
fprintf('Saved values: %s\n', resultsFile);
end

function acf = sample_autocorrelation(y, maxLag)
%SAMPLE_AUTOCORRELATION Sample autocorrelation of y(t) for lags 0..maxLag.

centred = y - mean(y);
denominator = sum(centred .^ 2);
acf = zeros(maxLag + 1, 1);
for lag = 0:maxLag
    acf(lag + 1) = sum(centred(1:end - lag) .* centred(1 + lag:end)) / denominator;
end
end

function plot_periodicity(figureFile, acfLag, acf, dailyLags, firstPeakLag, ...
    weekdayProfile, weekendProfile, trainSplit, nHours)
%PLOT_PERIODICITY The 24-hour period as it appears in the series itself.

figureHandle = light_figure('Daily periodicity of the hourly demand');
tiles = tiledlayout(figureHandle, 1, 2, 'Padding', 'compact', ...
    'TileSpacing', 'compact');

axesHandle = nexttile(tiles);
hold(axesHandle, 'on');
for lag = dailyLags
    xline(axesHandle, lag, ':', 'Color', [0.65, 0.65, 0.65], ...
        'HandleVisibility', 'off');
end
yline(axesHandle, 0, '-', 'Color', [0.75, 0.75, 0.75], ...
    'HandleVisibility', 'off');
plot(axesHandle, acfLag, acf, 'Color', [0.0000, 0.4470, 0.7410], ...
    'LineWidth', 1.6, 'DisplayName', 'Sample autocorrelation');
plot(axesHandle, dailyLags, acf(dailyLags + 1), 'o', 'MarkerSize', 6, ...
    'MarkerFaceColor', [0.8500, 0.3250, 0.0980], ...
    'MarkerEdgeColor', 'white', 'LineWidth', 1.0, ...
    'DisplayName', 'Multiples of 24 h');
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
xlim(axesHandle, [0, acfLag(end)]);
xticks(axesHandle, 0:24:acfLag(end));
xlabel(axesHandle, 'Lag (hours)');
ylabel(axesHandle, 'Autocorrelation of $y(t)$', 'Interpreter', 'latex');
title(axesHandle, 'Demand repeats every 24 hours');
subtitle(axesHandle, sprintf( ...
    'first local peak at lag %d h, r = %.3f;  r at 168 h = %.3f', ...
    firstPeakLag, acf(firstPeakLag + 1), acf(169)));
legend(axesHandle, 'Location', 'northeast', 'EdgeColor', [0.35, 0.35, 0.35]);

axesHandle = nexttile(tiles);
hold(axesHandle, 'on');
plot(axesHandle, 0:23, weekdayProfile, 'Color', [0.0000, 0.4470, 0.7410], ...
    'LineWidth', 1.8, 'DisplayName', 'Monday to Friday');
plot(axesHandle, 0:23, weekendProfile, 'Color', [0.8500, 0.3250, 0.0980], ...
    'LineWidth', 1.8, 'DisplayName', 'Saturday and Sunday');
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
xlim(axesHandle, [0, 23]);
xticks(axesHandle, 0:3:23);
xlabel(axesHandle, 'Hour of day');
ylabel(axesHandle, 'Mean $y(t)$ (MW)', 'Interpreter', 'latex');
title(axesHandle, 'The shape of the day, and of the weekend');
legend(axesHandle, 'Location', 'southeast', 'EdgeColor', [0.35, 0.35, 0.35]);

title(tiles, 'Hourly national demand: what the series says before any model', ...
    'FontWeight', 'bold');
subtitle(tiles, sprintf('%s split, %s hours', string(trainSplit), ...
    format_number(nHours)));
exportgraphics(figureHandle, figureFile, 'Resolution', 300);
close(figureHandle);
end

function plot_delay_sweep(figureFile, delays, mae, meanMae, testMaeByRestart, ...
    naiveMae, bestDelay, bestIndex, predictionHorizon, neurons, trainK, ...
    nWindows, testSplit)
%PLOT_DELAY_SWEEP Held-out error against tapped-delay length, and horizon.

figureHandle = light_figure('Held-out error by tapped-delay length');
tiles = tiledlayout(figureHandle, 1, 2, 'Padding', 'compact', ...
    'TileSpacing', 'compact');

axesHandle = nexttile(tiles);
hold(axesHandle, 'on');
xline(axesHandle, 24, ':', 'Color', [0.65, 0.65, 0.65], ...
    'HandleVisibility', 'off');
% Some restarts roll out unstably and land hundreds of MW away. They are
% the point of running restarts at all, but left on the axis they would
% flatten everything else, so they are pinned to the top edge and counted.
axisTop = 1.15 * max([meanMae, mae(end, :), mean(naiveMae)]);
delayGrid = repmat(delays(:), 1, size(testMaeByRestart, 2));
onScale = testMaeByRestart <= axisTop;
nOffScale = sum(~onScale, 'all');
plot(axesHandle, delayGrid(onScale), testMaeByRestart(onScale), '.', ...
    'Color', [0.70, 0.70, 0.70], 'MarkerSize', 11, 'HandleVisibility', 'off');
plot(axesHandle, delayGrid(~onScale), axisTop * ones(nOffScale, 1), '^', ...
    'Color', [0.70, 0.70, 0.70], 'MarkerSize', 5, 'HandleVisibility', 'off');
plot(axesHandle, NaN, NaN, '.', 'Color', [0.70, 0.70, 0.70], ...
    'MarkerSize', 11, 'DisplayName', 'Individual runs');
plot(axesHandle, delays, meanMae, '-o', 'Color', [0.0000, 0.4470, 0.7410], ...
    'MarkerFaceColor', [0.0000, 0.4470, 0.7410], 'MarkerSize', 5, ...
    'LineWidth', 1.8, 'DisplayName', 'Selected run, mean over $k$');
plot(axesHandle, delays, mae(end, :), '-s', ...
    'Color', [0.4660, 0.6740, 0.1880], 'MarkerSize', 5, 'LineWidth', 1.4, ...
    'DisplayName', sprintf('At $k = %d$', predictionHorizon));
yline(axesHandle, mean(naiveMae), '--', 'Color', [0.35, 0.35, 0.35], ...
    'LineWidth', 1.0, 'DisplayName', 'Daily-naive reference');
plot(axesHandle, bestDelay, meanMae(bestIndex), 'o', 'MarkerSize', 10, ...
    'MarkerEdgeColor', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.6, ...
    'DisplayName', sprintf('Best: %d taps', bestDelay));
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
ylim(axesHandle, [0, axisTop]);
xticks(axesHandle, delays);
xlabel(axesHandle, 'Tapped-delay length (hours)');
ylabel(axesHandle, 'MAE of $a^2(T+k)=\hat{y}(T+k)$ (MW)', ...
    'Interpreter', 'latex');
title(axesHandle, 'How much memory the forecast needs');
subtitle(axesHandle, sprintf( ...
    '%d of %d runs rolled out unstably, above the axis (arrows)', ...
    nOffScale, numel(testMaeByRestart)));
legend(axesHandle, 'Location', 'best', 'Interpreter', 'latex', ...
    'EdgeColor', [0.35, 0.35, 0.35]);

axesHandle = nexttile(tiles);
hold(axesHandle, 'on');
% Ordered dark blue to dark red: the delay is ordinal, and every colour
% stays legible on white.
shade = linspace(0, 1, numel(delays)).';
colours = [0.00 + 0.80 * shade, 0.30 - 0.20 * shade, 0.60 - 0.50 * shade];
for delayIndex = 1:numel(delays)
    lineWidth = 1.4;
    if delayIndex == bestIndex
        lineWidth = 2.4;
    end
    plot(axesHandle, 1:predictionHorizon, mae(:, delayIndex), ...
        'Color', colours(delayIndex, :), 'LineWidth', lineWidth, ...
        'DisplayName', sprintf('%d taps', delays(delayIndex)));
end
plot(axesHandle, 1:predictionHorizon, naiveMae, '--', ...
    'Color', [0.35, 0.35, 0.35], 'LineWidth', 1.4, ...
    'DisplayName', 'Daily naive');
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
xlim(axesHandle, [1, predictionHorizon]);
xlabel(axesHandle, 'Prediction horizon, $k$ (hours)', 'Interpreter', 'latex');
ylabel(axesHandle, 'MAE (MW)');
title(axesHandle, 'Error along the 24-hour curve');
legend(axesHandle, 'Location', 'southeast', 'NumColumns', 2, ...
    'EdgeColor', [0.35, 0.35, 0.35]);

title(tiles, ['Closed-loop NARX load forecast on held-out origins: ', ...
    'the delay is measured, not assumed'], 'FontWeight', 'bold');
subtitle(tiles, sprintf(['%d neurons, trained to %d steps; %s origins ', ...
    'from the %s split'], neurons, trainK, format_number(nWindows), ...
    string(testSplit)));
exportgraphics(figureHandle, figureFile, 'Resolution', 300);
close(figureHandle);
end

function plot_error_by_hour(figureFile, maeByHourBest, maeByHourNaive, ...
    maeBest, maeShortest, naiveMae, bestDelay, shortestDelay, ...
    predictionHorizon)
%PLOT_ERROR_BY_HOUR Where in the day the remaining error sits.

figureHandle = light_figure('Held-out error by hour of day');
tiles = tiledlayout(figureHandle, 1, 2, 'Padding', 'compact', ...
    'TileSpacing', 'compact');

axesHandle = nexttile(tiles);
hold(axesHandle, 'on');
plot(axesHandle, 0:23, maeByHourBest, '-o', ...
    'Color', [0.0000, 0.4470, 0.7410], 'MarkerSize', 4, 'LineWidth', 1.8, ...
    'DisplayName', sprintf('NARX, %d taps', bestDelay));
plot(axesHandle, 0:23, maeByHourNaive, '--', ...
    'Color', [0.35, 0.35, 0.35], 'LineWidth', 1.4, ...
    'DisplayName', 'Daily naive');
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
xlim(axesHandle, [0, 23]);
xticks(axesHandle, 0:3:23);
xlabel(axesHandle, 'Hour of day being predicted');
ylabel(axesHandle, 'MAE (MW)');
title(axesHandle, 'The error keeps the shape of the day');
legend(axesHandle, 'Location', 'best', 'EdgeColor', [0.35, 0.35, 0.35]);

axesHandle = nexttile(tiles);
hold(axesHandle, 'on');
plot(axesHandle, 1:predictionHorizon, maeShortest, ...
    'Color', [0.4940, 0.1840, 0.5560], 'LineWidth', 1.6, ...
    'DisplayName', sprintf('NARX, %d taps', shortestDelay));
plot(axesHandle, 1:predictionHorizon, maeBest, ...
    'Color', [0.0000, 0.4470, 0.7410], 'LineWidth', 1.8, ...
    'DisplayName', sprintf('NARX, %d taps', bestDelay));
plot(axesHandle, 1:predictionHorizon, naiveMae, '--', ...
    'Color', [0.35, 0.35, 0.35], 'LineWidth', 1.4, ...
    'DisplayName', 'Daily naive');
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
xlim(axesHandle, [1, predictionHorizon]);
xlabel(axesHandle, 'Prediction horizon, $k$ (hours)', 'Interpreter', 'latex');
ylabel(axesHandle, 'MAE (MW)');
title(axesHandle, 'Shortest against best tapped-delay line');
legend(axesHandle, 'Location', 'northwest', 'EdgeColor', [0.35, 0.35, 0.35]);

title(tiles, 'What the chosen delay buys, hour by hour', ...
    'FontWeight', 'bold');
exportgraphics(figureHandle, figureFile, 'Resolution', 300);
close(figureHandle);
end

function text = format_number(value)
%FORMAT_NUMBER Add thousands separators without locale dependence.
text = regexprep(sprintf('%d', value), '(?<!^)(?=(\d{3})+$)', ',');
end

function figureHandle = light_figure(name)
%LIGHT_FIGURE A hidden, light-themed figure, so the PNG matches the repository.

figureHandle = figure('Color', 'white', 'Visible', 'off', 'Name', name, ...
    'Position', [100, 100, 1100, 450]);
if exist('theme', 'file')
    theme(figureHandle, 'light');
end
end
