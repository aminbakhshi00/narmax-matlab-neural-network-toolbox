function results = run_weather_availability(options)
%RUN_WEATHER_AVAILABILITY What the forecast costs when the weather is not known.
%
%   RESULTS = RUN_WEATHER_AVAILABILITY() re-scores the trained closed-loop
%   NARX load model on the same held-out origins four times over, changing
%   only where the *future* external input p(T+j), j >= 1, comes from.
%
%   The parallel NARX architecture closes the loop on the output and not on
%   the external input. At horizon k the layer-1 computation reads
%
%       n^1(T+k) = IW^{1,1} [p(T+k-1) ... p(T+k-n_p)]
%                + LW^{1,2} [a2Tilde(T+k-1) ... a2Tilde(T+k-n_a)] + b^1
%
%   so the delayed outputs are supplied by the model itself, but every
%   p(T+j) with j >= 1 has to be supplied from outside. With the delay set
%   1:delay the largest future offset a horizon needs is k - 1, so a
%   one-step forecast needs no future temperature at all, a three-hour
%   forecast needs two hours of it, and the full 24-hour curve needs 23.
%
%   The four sources, applied to the future taps only -- the loading window
%   stays measured in every case:
%
%     'measured'     the recorded temperature. Perfect weather foresight;
%                    the model's own error with weather error removed.
%     'persistence'  p(T+j) = p(T+j-24), yesterday at the same clock hour.
%                    Always available to an operator, and a weak forecast.
%     'climatology'  the training-window mean temperature for that clock
%                    hour. Available arbitrarily far ahead, carries no
%                    information about the particular day.
%     'frozen'       p(T) held constant, the crudest substitute.
%
%   A real numerical weather forecast is better than persistence and worse
%   than the recording, so the operational error of this model sits between
%   the 'measured' and 'persistence' curves.
%
%   Writes load_weather_availability.png and a results MAT file.
%
%   See also RUN_LOAD_EXPLAINABILITY, NARXFORECAST, IMPORT_DATA_LOAD.

arguments
    options.Delay (1,1) double {mustBePositive} = 24
    options.PredictionHorizon (1,1) double {mustBePositive} = 24
    options.NumWindows (1,1) double {mustBePositive} = 2000
    options.TrainSplit = 'dry-train'
    options.TestSplit = 'dry-test'
    options.RandomSeed (1,1) double = 2024
end

scriptFolder = fileparts(mfilename('fullpath'));
projectFolder = fileparts(scriptFolder);
addpath(projectFolder);
addpath(scriptFolder);
addpath(fullfile(projectFolder, 'Explainability'));

delay = options.Delay;
H = options.PredictionHorizon;
dailyPeriod = 24;
narxModel = loadSelectedCheckpoint(scriptFolder, delay);

% Origins carry a full extra day of history so that yesterday's temperature
% is available as a substitute for every one of them.
[p, y, t] = import_data_load(options.TestSplit);
firstOrigin = delay + dailyPeriod;
lastOrigin = numel(y) - H;
rng(options.RandomSeed, 'twister');
candidates = firstOrigin:lastOrigin;
nWindows = min(options.NumWindows, numel(candidates));
origins = sort(candidates(randperm(numel(candidates), nWindows)));

[pWindows, yWindows] = sample_load_windows('Delay', delay, ...
    'PredictionHorizon', H, 'Split', options.TestSplit, 'Origins', origins);

futureRows = delay + (1:H);
futureIndex = (1:H).' + origins;
measuredFuture = p(futureIndex);

% Clock-hour climatology of the temperature, from the training window only.
[pTrain, ~, tTrain] = import_data_load(options.TrainSplit);
climatology = accumarray(hour(tTrain) + 1, pTrain, [24 1], @mean);

source = struct('name', {}, 'future', {});
source(1).name = 'measured';
source(1).future = measuredFuture;
source(2).name = 'persistence';
source(2).future = p(futureIndex - dailyPeriod);
source(3).name = 'climatology';
source(3).future = climatology(hour(t(futureIndex)) + 1);
source(4).name = 'frozen';
source(4).future = repmat(reshape(p(origins), 1, []), H, 1);

horizon = (1:H).';
mae = zeros(H, numel(source));
mape = zeros(H, numel(source));
temperatureError = zeros(H, numel(source));

fprintf('\nWEATHER AVAILABILITY, %d-tap checkpoint, %d held-out origins\n', ...
    delay, nWindows);
for sourceIndex = 1:numel(source)
    substituted = pWindows;
    substituted(futureRows, :) = source(sourceIndex).future;
    [a2, yScored, maeByHorizon] = narxForecast(narxModel, substituted, ...
        yWindows, H);
    mae(:, sourceIndex) = maeByHorizon;
    mape(:, sourceIndex) = 100 * mean(abs(yScored - a2) ./ yScored, 2);
    temperatureError(:, sourceIndex) = ...
        mean(abs(source(sourceIndex).future - measuredFuture), 2);
    fprintf(['  %-12s MAE %.2f MW at k = 1, %.2f at k = 3, %.2f at k = 24; ', ...
        'mean %.2f MW (%.2f%%)\n'], source(sourceIndex).name, ...
        mae(1, sourceIndex), mae(3, sourceIndex), mae(end, sourceIndex), ...
        mean(mae(:, sourceIndex)), mean(mape(:, sourceIndex)));
end

naiveMae = mean(abs(y(futureIndex) - y(futureIndex - dailyPeriod)), 2);
penalty = mae - mae(:, 1);
fprintf('  daily-naive load reference: %.2f MW mean\n', mean(naiveMae));
fprintf(['  cost of losing the weather: %+.2f MW at k = 3, %+.2f MW at ', ...
    'k = 24 (persistence temperature)\n'], penalty(3, 2), penalty(end, 2));

makeFigure(scriptFolder, horizon, mae, temperatureError, naiveMae, ...
    {source.name}, delay, nWindows, options.TestSplit);

results = struct;
results.options = options;
results.origins = origins;
results.horizon = horizon;
results.sourceNames = {source.name};
results.mae = mae;
results.mape = mape;
results.penalty = penalty;
results.temperatureError = temperatureError;
results.naiveMae = naiveMae;
resultsFile = fullfile(scriptFolder, 'load_weather_availability_results.mat');
save(resultsFile, 'results');
fprintf('Saved values: %s\n', resultsFile);
end

function narxModel = loadSelectedCheckpoint(scriptFolder, delay)
%LOADSELECTEDCHECKPOINT The run the delay experiment kept for this delay.
restart = 1;
neurons = 10;
trainK = 24;
sweepFile = fullfile(scriptFolder, 'load_narx_delay_experiment_results.mat');
if isfile(sweepFile)
    loaded = load(sweepFile, 'results');
    delayIndex = find(loaded.results.delays == delay, 1);
    if ~isempty(delayIndex)
        restart = loaded.results.selectedRestart(delayIndex);
        neurons = loaded.results.options.Neurons;
        trainK = loaded.results.options.TrainK;
    end
end
folder = fullfile(scriptFolder, 'train', 'load', ...
    sprintf('narx_%ddelay_%dneurons_%dsteps', delay, neurons, trainK));
if restart == 1
    modelFile = fullfile(folder, 'narx_model.mat');
else
    modelFile = fullfile(folder, sprintf('narx_model_run%d.mat', restart));
end
assert(isfile(modelFile), 'Checkpoint not found: %s', modelFile);
loaded = load(modelFile, 'narx');
narxModel = loaded.narx;
end

function makeFigure(outputFolder, horizon, mae, temperatureError, ...
    naiveMae, sourceNames, delay, nWindows, testSplit)
%MAKEFIGURE Load error and temperature error, side by side.

colors = [0.0000, 0.4470, 0.7410; 0.4940, 0.1840, 0.5560; ...
    0.8500, 0.3250, 0.0980; 0.4660, 0.6740, 0.1880];
figureHandle = figure('Name', 'Weather availability', 'Color', 'white', ...
    'Visible', 'off', 'Position', [80, 60, 1250, 470]);
if exist('theme', 'file')
    theme(figureHandle, 'light');
end
layout = tiledlayout(figureHandle, 1, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

axesLoad = nexttile(layout);
hold(axesLoad, 'on');
for sourceIndex = 1:numel(sourceNames)
    plot(axesLoad, horizon, mae(:, sourceIndex), '-', ...
        'Color', colors(sourceIndex, :), 'LineWidth', 1.8, ...
        'DisplayName', sprintf('%s temperature', sourceNames{sourceIndex}));
end
plot(axesLoad, horizon, naiveMae, '--', 'Color', [0.35, 0.35, 0.35], ...
    'LineWidth', 1.4, 'DisplayName', 'Daily-naive load forecast');
hold(axesLoad, 'off');
grid(axesLoad, 'on');
box(axesLoad, 'on');
xlim(axesLoad, [1, max(horizon)]);
xlabel(axesLoad, 'Prediction horizon $k$ (hours)', 'Interpreter', 'latex');
ylabel(axesLoad, 'MAE of $a^2(T+k)$ (MW)', 'Interpreter', 'latex');
title(axesLoad, 'The forecast, by where its future weather came from');
legend(axesLoad, 'Location', 'southeast', 'EdgeColor', [0.35, 0.35, 0.35]);

axesTemperature = nexttile(layout);
hold(axesTemperature, 'on');
for sourceIndex = 2:numel(sourceNames)
    plot(axesTemperature, horizon, temperatureError(:, sourceIndex), '-', ...
        'Color', colors(sourceIndex, :), 'LineWidth', 1.8, ...
        'DisplayName', sourceNames{sourceIndex});
end
hold(axesTemperature, 'off');
grid(axesTemperature, 'on');
box(axesTemperature, 'on');
xlim(axesTemperature, [1, max(horizon)]);
xlabel(axesTemperature, 'Lead time of the temperature needed (hours)');
ylabel(axesTemperature, 'Temperature error (deg C)');
title(axesTemperature, 'How wrong each substitute weather is');
legend(axesTemperature, 'Location', 'southeast', ...
    'EdgeColor', [0.35, 0.35, 0.35]);

title(layout, ['What the day-ahead load forecast costs when tomorrow''s ' ...
    'temperature is not known'], 'FontWeight', 'bold', 'Color', 'black');
subtitle(layout, sprintf(['%d-tap checkpoint, %d held-out origins from ' ...
    'the %s split; only the future taps p(T+j), j >= 1, are substituted'], ...
    delay, nWindows, string(testSplit)), 'FontSize', 9, ...
    'Color', [0.25, 0.25, 0.25]);
set(findall(figureHandle, 'Type', 'axes'), 'Color', 'white', ...
    'XColor', 'black', 'YColor', 'black', 'GridColor', [0.78, 0.78, 0.78], ...
    'FontSize', 9);
exportgraphics(figureHandle, ...
    fullfile(outputFolder, 'load_weather_availability.png'), ...
    'Resolution', 300);
close(figureHandle);
end
