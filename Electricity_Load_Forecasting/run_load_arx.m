function results = run_load_arx(options)
%RUN_LOAD_ARX Linear ARX on the hourly load.
%
%   RESULTS = RUN_LOAD_ARX() fits the linear ARX model of ARXMODEL at every
%   tapped-delay length of the NARX sweep. Same splits, same [0, 1]
%   normalisation, same 300-epoch budget, same horizon schedule, same
%   closed-loop rollout at scoring time as the NARX runs on this page, so the
%   difference in held-out error is the value of the nonlinearity.
%
%   Training sequences are cut by NARXMODEL.PREPARE_DATA, which draws their
%   origins at random -- the "randomly overlapping" scheme of Kelley (2024).
%   This used to be a two-way comparison: TILESEQUENCES chose between a
%   one-sample stride and a tiling stride, and the tiling stride lost badly
%   because at the final stage it lands every origin 24 hours apart, the
%   period of the series, so all of them fall at the same clock hour. That
%   property is gone and there is nothing left to sweep, so this script now
%   fits one model per delay.
%
%   The optimiser is 'trainbr', Bayesian regularisation. It beat 'trainlm' at
%   both delays it was tested on (77.96 against 80.92 MW at 6 taps, 61.47
%   against 63.83 at 48) for the same runtime, so the choice is settled and
%   no longer an option. The NARX runs go the other way and keep 'trainlm':
%   'trainbr' has to estimate the effective number of parameters through the
%   Hessian, which is cheap for these 97 linear weights and punishing for a
%   nonconvex net, where it scored 225.29 MW against 73.30 and took 37 times
%   as long. The ARX-to-NARX comparison therefore differs in the optimiser as
%   well as the model. Those figures predate the change of sampling scheme.
%
%   Early stopping is off: 'trainbr' ignores a validation set by design, so
%   its only effect was to withhold an inner test block that nothing reads.
%   Every sequence now trains, which measured better at every delay tried.
%   The 300-epoch budget never binds -- stages end after 7 to 19 epochs and
%   100, 300 and 900 give identical fits.
%
%   TWO SEEDS, WITH DIFFERENT CONSEQUENCES
%     RANDOMSEED    seeds the weight initialisation. Repeating a fit from a
%                   different initialisation is still pointless for a linear
%                   model: the first stage of the horizon schedule predicts
%                   one step ahead, which is a convex least-squares problem,
%                   so the initialisation is erased before the second stage
%                   begins and every seed reaches the same weights.
%     SEQUENCESEED  selects which subsequences exist. PREPARE_DATA draws its
%                   origins from a private stream seeded by this and the
%                   window geometry, so it is independent of RANDOMSEED and
%                   the convexity argument above does NOT cover it: a
%                   different value is a genuinely different training set and
%                   gives a genuinely different fit. One fit per delay is
%                   therefore a sample of size one. Sweep SEQUENCESEED if you
%                   need to know how much of a delay-to-delay difference is
%                   just the draw.
%
%   Because the model is linear, its closed-loop rollout is a linear
%   recursion and its long-horizon behaviour is fixed by the roots of
%
%     A(z) = z^d - a_1 z^(d-1) - ... - a_d ,
%
%   the poles of the fitted model. A pole outside the unit circle means an
%   error is amplified geometrically as the rollout runs, so the poles are
%   extracted for every fit and reported beside the error.
%
%   NAME-VALUE OPTIONS
%     Delays        Tapped-delay lengths, in hours.
%                   Default [2 6 12 18 24 30 48].
%     SequenceSeed  Which draw of subsequence origins to fit on. Default 0.
%     Retrain       Refit models that already have a checkpoint. Default false.
%
%   Writes two PNG figures and one results MAT file beside this file, all
%   named load_arx_random_*, and one checkpoint per fit under train/load. The
%   older load_arx_* figures are the retired two-stride comparison and are
%   left alone: the README still shows them as the evidence for the change.
%
%   See also ARXMODEL, NARXMODEL/PREPARE_DATA, RUN_LOAD_RELU_COMPARISON.

arguments
    options.Delays (1,:) double {mustBePositive} = [2 6 12 18 24 30 48]
    options.TrainK (1,1) double {mustBePositive} = 24
    options.PredictionHorizon (1,1) double {mustBePositive} = 24
    options.NumWindows (1,1) double {mustBePositive} = 2000
    options.IterPerRun (1,1) double {mustBePositive} = 300
    options.RandomSeed (1,1) double = 2024
    options.SequenceSeed (1,1) double = 0
    options.Tag (1,1) string = ""
    options.TrainSplit = 'dry-train'
    options.ValidationSplit = 'dry-val'
    options.TestSplit = 'dry-test'
    options.Retrain (1,1) logical = false
end

scriptFolder = fileparts(mfilename('fullpath'));
projectFolder = fileparts(scriptFolder);
addpath(projectFolder);                                   % ARXmodel, secs2hms
addpath(scriptFolder);                                    % import_data_load
addpath(fullfile(projectFolder, 'Explainability'));       % narxForecast

delays = sort(options.Delays);
nDelays = numel(delays);
if strlength(options.Tag) > 0
    suffix = "_" + options.Tag;
else
    suffix = "";
end
trainK = options.TrainK;
predictionHorizon = options.PredictionHorizon;

% The origins are drawn against a 48-hour delay, the longest in the sweep,
% so that every experiment on this page scores on one and the same set.
originDelay = 48;
[~, ~, testOriginInfo] = sample_load_windows('Delay', originDelay, ...
    'PredictionHorizon', predictionHorizon, ...
    'NumWindows', options.NumWindows, 'Split', options.TestSplit);
origins = testOriginInfo.originIndex;
[~, ~, validationOriginInfo] = sample_load_windows('Delay', originDelay, ...
    'PredictionHorizon', predictionHorizon, ...
    'NumWindows', options.NumWindows, 'Split', options.ValidationSplit);
validationOrigins = validationOriginInfo.originIndex;

[~, ~, trainTimes] = import_data_load(options.TrainSplit);
[~, ~, validationTimes] = import_data_load(options.ValidationSplit);
validationIsInSample = validationTimes(1) <= trainTimes(end) && ...
    validationTimes(end) >= trainTimes(1);

[~, yTest] = import_data_load(options.TestSplit);
targetIndices = (1:predictionHorizon).' + origins;
naiveMae = mean(abs(yTest(targetIndices) - yTest(targetIndices - 24)), 2);

fprintf('\nLINEAR ARX ON HOURLY LOAD\n');
fprintf('  delays %s, horizon %d, one fit per delay\n', ...
    mat2str(delays), predictionHorizon);
fprintf('  random-overlap sequences, draw %d\n', options.SequenceSeed);
fprintf('  trainbr, early stopping on, %d epochs per horizon stage\n', ...
    options.IterPerRun);
fprintf('  fit on %s (%d h), checked on %s, scored on %s (%d origins)\n', ...
    string(options.TrainSplit), numel(trainTimes), ...
    string(options.ValidationSplit), string(options.TestSplit), ...
    numel(origins));
if validationIsInSample
    fprintf(['  NOTE %s lies inside the %s span, so its error is ', ...
        'in-sample and is not reported. Nothing is selected on it here:\n', ...
        '       there is one fit per delay.\n'], ...
        string(options.ValidationSplit), string(options.TrainSplit));
end
fprintf('\n');

maeCurve = nan(predictionHorizon, nDelays);
validationMae = nan(1, nDelays);
testMae = nan(1, nDelays);
maxPole = nan(1, nDelays);
poles = cell(1, nDelays);
trainingSeconds = nan(1, nDelays);
checkpointFile = strings(1, nDelays);
parameterCount = zeros(1, nDelays);

for delayIndex = 1:nDelays
    delay = delays(delayIndex);
    checkpointFolder = fullfile(scriptFolder, 'train', 'load', ...
        sprintf('arx_%ddelay_%dsteps_random_seed%d%s', delay, trainK, ...
        options.SequenceSeed, suffix));
    checkpointFile(delayIndex) = fullfile(checkpointFolder, 'arx_model.mat');
    thisFile = checkpointFile(delayIndex);

    if options.Retrain || ~isfile(thisFile)
        fprintf('TRAINING ARX delay %2d\n', delay);
        [p_pre, y_pre, ~, segmentStart] = import_data_load(options.TrainSplit);

        rng(options.RandomSeed, 'twister');
        arx = ARXmodel(delay);
        arx.segmentStart = segmentStart;
        arx.sequenceSeed = options.SequenceSeed;
        arx.trainAlg = 'trainbr';
        arx.iterPerRun = options.IterPerRun;
        arx.iterAfterValley = 100;
        arx.iterAfterSeq = 50;
        arx.maxStep = 10;
        arx.initialTraining = 10;
        arx.zero_input_delay = false;
        arx.writeToConsole = false;

        trainingTimer = tic;
        arx = arx.train(p_pre, y_pre, trainK);
        trainingSeconds(delayIndex) = toc(trainingTimer);

        if ~isfolder(checkpointFolder)
            mkdir(checkpointFolder);
        end
        save(thisFile, 'arx');
    else
        checkpoint = load(thisFile, 'arx');
        arx = checkpoint.arx;
    end

    assert(arx.narx.numLayers == 1, ...
        'Checkpoint %s has %d layers; ARX must have one.', ...
        thisFile, arx.narx.numLayers);
    parameterCount(delayIndex) = numel(getwb(arx.narx));

    % The poles of the fitted difference equation.
    feedback = arx.narx.LW{1,1};
    poles{delayIndex} = roots([1, -feedback(:).']);
    maxPole(delayIndex) = max(abs(poles{delayIndex}));

    [pTest, yTestWindows] = sample_load_windows('Delay', delay, ...
        'PredictionHorizon', predictionHorizon, ...
        'Split', options.TestSplit, 'Origins', origins);
    [pValidation, yValidationWindows] = sample_load_windows( ...
        'Delay', delay, 'PredictionHorizon', predictionHorizon, ...
        'Split', options.ValidationSplit, 'Origins', validationOrigins);

    [~, ~, validationMaeByHorizon] = narxForecast(arx, pValidation, ...
        yValidationWindows, predictionHorizon);
    [~, ~, maeByHorizon] = narxForecast(arx, pTest, yTestWindows, ...
        predictionHorizon);

    maeCurve(:, delayIndex) = maeByHorizon;
    validationMae(delayIndex) = mean(validationMaeByHorizon);
    testMae(delayIndex) = mean(maeByHorizon);

    if validationIsInSample
        fprintf(['  delay %2d: %3d weights | max|pole| %.4f%s | ', ...
            'test %8.2f\n'], delay, parameterCount(delayIndex), ...
            maxPole(delayIndex), stability_flag(maxPole(delayIndex)), ...
            testMae(delayIndex));
    else
        fprintf(['  delay %2d: %3d weights | max|pole| %.4f%s | ', ...
            'val %8.2f | test %8.2f\n'], delay, parameterCount(delayIndex), ...
            maxPole(delayIndex), stability_flag(maxPole(delayIndex)), ...
            validationMae(delayIndex), testMae(delayIndex));
    end
end
fprintf('\n');

% ---------------------------------------------------------------------
% Report.
% ---------------------------------------------------------------------
fprintf('  Held-out mean MAE (MW), and the largest pole of each fit\n');
fprintf('    %-6s %8s | %10s | %10s\n', 'delay', 'weights', 'MAE', 'max|pole|');
for delayIndex = 1:nDelays
    fprintf('    %-6d %8d | %10.2f | %10.4f\n', delays(delayIndex), ...
        parameterCount(delayIndex), testMae(delayIndex), maxPole(delayIndex));
end

unstable = maxPole >= 1;
fprintf('\n  fits with a pole on or outside the unit circle: %d of %d\n', ...
    sum(unstable), numel(unstable));
if any(unstable)
    fprintf('    delays %s\n', mat2str(delays(unstable)));
else
    fprintf('    none\n');
end
fprintf('\n  daily-naive reference: %.2f MW\n', mean(naiveMae));

maeFigure = fullfile(scriptFolder, ...
    sprintf('load_arx_random_mae_vs_horizon%s.png', suffix));
plot_arx_mae(maeFigure, maeCurve, naiveMae, delays, predictionHorizon, ...
    numel(origins), options.TestSplit, options.TrainSplit, ...
    numel(trainTimes), options.SequenceSeed);
poleFigure = fullfile(scriptFolder, ...
    sprintf('load_arx_random_poles%s.png', suffix));
plot_arx_poles(poleFigure, poles, maxPole, delays, options.TrainSplit);
angleFigure = fullfile(scriptFolder, ...
    sprintf('load_arx_random_pole_angles%s.png', suffix));
plot_arx_pole_angles(angleFigure, poles, delays, options.TrainSplit);

results = struct;
results.options = options;
results.trainAlg = "trainbr";
results.delays = delays;
results.sequenceSeed = options.SequenceSeed;
results.horizon = (1:predictionHorizon).';
results.maeCurve = maeCurve;
results.validationMae = validationMae;
results.testMae = testMae;
results.poles = poles;
results.maxPole = maxPole;
results.unstable = unstable;
results.parameterCount = parameterCount;
results.trainingSeconds = trainingSeconds;
results.checkpointFile = checkpointFile;
results.origins = origins;
results.validationOrigins = validationOrigins;
results.naiveMae = naiveMae;
results.validationIsInSample = validationIsInSample;
results.trainHours = numel(trainTimes);
results.figureFiles = [string(maeFigure), string(poleFigure), string(angleFigure)];

resultsFile = fullfile(scriptFolder, ...
    sprintf('load_arx_random_results%s.mat', suffix));
save(resultsFile, 'results');
fprintf('  Saved figures: %s, %s\n  Saved values: %s\n', ...
    maeFigure, poleFigure, resultsFile);
end

function flag = stability_flag(maxPoleValue)
%STABILITY_FLAG Mark a fit whose rollout can amplify without bound.
if maxPoleValue >= 1
    flag = ' UNSTABLE';
else
    flag = '';
end
end

function plot_arx_mae(figureFile, maeCurve, naiveMae, delays, ...
    predictionHorizon, nWindows, testSplit, trainSplit, trainHours, ...
    sequenceSeed)
%PLOT_ARX_MAE Held-out error against horizon, every delay.

figureHandle = figure('Color', 'white', 'Visible', 'off', ...
    'Position', [100, 100, 1180, 660]);
if exist('theme', 'file')
    theme(figureHandle, 'light');
end
axesHandle = axes(figureHandle);
hold(axesHandle, 'on');

nDelays = numel(delays);
horizons = (1:predictionHorizon).';
% One light-to-dark ramp, because delay is the only ordinal axis left.
rampEnds = [0.45 0.66 0.81; 0.03 0.19 0.42];

plot(axesHandle, horizons, naiveMae, '-', 'Color', [0.45, 0.45, 0.45], ...
    'LineWidth', 2.2, 'DisplayName', 'Daily naive');
for delayIndex = 1:nDelays
    t = (delayIndex - 1) / max(nDelays - 1, 1);
    colour = (1 - t) * rampEnds(1, :) + t * rampEnds(2, :);
    plot(axesHandle, horizons, maeCurve(:, delayIndex), ...
        'Color', colour, 'LineStyle', '-', 'Marker', 'o', ...
        'MarkerSize', 3, 'MarkerFaceColor', colour, 'LineWidth', 1.3, ...
        'DisplayName', sprintf('%2d taps', delays(delayIndex)));
end

hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
axesHandle.GridColor = [0.75, 0.75, 0.75];
axesHandle.GridAlpha = 0.5;
xlim(axesHandle, [1, predictionHorizon]);
xticks(axesHandle, [1, 4:4:predictionHorizon]);
set(axesHandle, 'YScale', 'log');
xlabel(axesHandle, 'Prediction horizon, $k$ (hours)', 'Interpreter', 'latex');
ylabel(axesHandle, 'MAE of $\hat{y}(T+k)$ (MW, log scale)', ...
    'Interpreter', 'latex');
title(axesHandle, ...
    'Linear ARX: held-out error along the 24-hour load curve', ...
    'FontWeight', 'bold');
subtitle(axesHandle, sprintf(['one fit per delay, random-overlap ', ...
    'sequences (draw %d). Light to dark is %d to %d taps.\n', ...
    'fitted on %s (%s hours); %s origins from the %s split'], ...
    sequenceSeed, delays(1), delays(end), string(trainSplit), ...
    format_number(trainHours), format_number(nWindows), string(testSplit)));
legend(axesHandle, 'Location', 'eastoutside', 'NumColumns', 1, ...
    'FontSize', 8, 'EdgeColor', [0.70, 0.70, 0.70]);

exportgraphics(figureHandle, figureFile, 'Resolution', 300);
close(figureHandle);
end

function plot_arx_poles(figureFile, poles, maxPole, delays, trainSplit)
%PLOT_ARX_POLES Where the fitted difference equations put their poles.
%
%   Each delay gets its own colour and marker, so the complex plane can be read
%   per model rather than as one undifferentiated cloud: which fit owns the
%   poles nearest the unit circle is the whole question, and that is invisible
%   when every fit is drawn in one colour. The palette is Okabe-Ito, chosen
%   because its hues stay distinct under the common forms of colour blindness;
%   the marker shape repeats the delay so the two are never told apart by
%   colour alone. The right panel reuses the same colours, so a point there and
%   its cloud on the left are the same model.

figureHandle = figure('Color', 'white', 'Visible', 'off', ...
    'Position', [100, 100, 1180, 540]);
if exist('theme', 'file')
    theme(figureHandle, 'light');
end
tiles = tiledlayout(figureHandle, 1, 2, 'Padding', 'compact', ...
    'TileSpacing', 'compact');

palette = [0.00 0.45 0.70;      % blue
           0.00 0.62 0.45;      % bluish green
           0.34 0.71 0.91;      % sky blue
           0.90 0.62 0.00;      % orange
           0.84 0.37 0.00;      % vermillion
           0.80 0.47 0.65;      % reddish purple
           0.00 0.00 0.00];     % black
markers = {'o', 's', '^', 'd', 'v', 'p', 'h'};
nDelays = numel(delays);
colourOf = @(i) palette(mod(i - 1, size(palette, 1)) + 1, :);
markerOf = @(i) markers{mod(i - 1, numel(markers)) + 1};

% ---- the complex plane ------------------------------------------------
axesHandle = nexttile(tiles);
hold(axesHandle, 'on');
angles = linspace(0, 2*pi, 400);
plot(axesHandle, cos(angles), sin(angles), '-', 'Color', [0.35 0.35 0.35], ...
    'LineWidth', 1.4, 'DisplayName', 'Unit circle');
yline(axesHandle, 0, '-', 'Color', [0.88 0.88 0.88], 'HandleVisibility', 'off');
xline(axesHandle, 0, '-', 'Color', [0.88 0.88 0.88], 'HandleVisibility', 'off');
for delayIndex = 1:nDelays
    z = poles{delayIndex};
    plot(axesHandle, real(z), imag(z), markerOf(delayIndex), ...
        'Color', colourOf(delayIndex), 'MarkerSize', 5, 'LineWidth', 1.1, ...
        'DisplayName', sprintf('%d taps', delays(delayIndex)));
end
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
axis(axesHandle, 'equal');
xlim(axesHandle, [-1.25, 1.25]);
ylim(axesHandle, [-1.25, 1.25]);
xlabel(axesHandle, 'Real part');
ylabel(axesHandle, 'Imaginary part');
title(axesHandle, 'Poles of every fitted ARX');
subtitle(axesHandle, 'inside the circle the rollout decays, outside it grows');
legend(axesHandle, 'Location', 'southoutside', 'NumColumns', 4, ...
    'FontSize', 8, 'EdgeColor', [0.70, 0.70, 0.70]);

% ---- the largest pole, delay by delay ---------------------------------
axesHandle = nexttile(tiles);
hold(axesHandle, 'on');
yline(axesHandle, 1, '-', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.4, ...
    'DisplayName', 'Unit circle, $|z| = 1$');
plot(axesHandle, delays, maxPole, '-', 'Color', [0.65 0.65 0.65], ...
    'LineWidth', 1.4, 'HandleVisibility', 'off');
for delayIndex = 1:nDelays
    plot(axesHandle, delays(delayIndex), maxPole(delayIndex), ...
        markerOf(delayIndex), 'Color', colourOf(delayIndex), ...
        'MarkerFaceColor', colourOf(delayIndex), 'MarkerSize', 7, ...
        'LineWidth', 1.1, 'HandleVisibility', 'off');
end
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
xticks(axesHandle, delays);
xlim(axesHandle, [delays(1) - 2, delays(end) + 2]);
ylim(axesHandle, [0.85, max(1.02, 1.02 * max(maxPole))]);
xlabel(axesHandle, 'Tapped-delay length (hours)');
ylabel(axesHandle, 'Largest $|$pole$|$', 'Interpreter', 'latex');
title(axesHandle, 'How close the rollout sits to instability');
subtitle(axesHandle, sprintf('%d of %d fits are outside', ...
    nnz(maxPole >= 1), numel(maxPole)));
legend(axesHandle, 'Location', 'southoutside', 'Interpreter', 'latex', ...
    'FontSize', 8, 'EdgeColor', [0.70, 0.70, 0.70]);

title(tiles, 'The linear rollout is governed by where the poles sit', ...
    'FontWeight', 'bold');
subtitle(tiles, sprintf('fitted on the %s split', string(trainSplit)));
exportgraphics(figureHandle, figureFile, 'Resolution', 300);
close(figureHandle);
end

function plot_arx_pole_angles(figureFile, poles, delays, trainSplit)
%PLOT_ARX_POLE_ANGLES What rhythms each fitted ARX encodes, and how persistent.
%
%   A pole z = rho*exp(i*theta) is one mode of the model's free behaviour: rho
%   is how much of it survives each hour, and theta is its rhythm, repeating
%   every 360/theta_deg hours. This figure reads the poles as periods rather
%   than as positions, which is what makes the daily cycle visible: a network
%   that has learned the 24-hour rhythm must put a pole near 15 degrees, and
%   near the unit circle, or the mode dies before the day is out.
%
%   Conjugates carry the same rhythm, so only the upper half plane is drawn.
%   Purely real poles have no rhythm at all (theta = 0) and are counted in the
%   subtitle rather than placed on a period axis.
%
%   Colours and markers are the Okabe-Ito set used by PLOT_ARX_POLES, so a
%   delay keeps its identity across both figures.

figureHandle = figure('Color', 'white', 'Visible', 'off', ...
    'Position', [100, 100, 1180, 540]);
if exist('theme', 'file')
    theme(figureHandle, 'light');
end
tiles = tiledlayout(figureHandle, 1, 2, 'Padding', 'compact', ...
    'TileSpacing', 'compact');

palette = [0.00 0.45 0.70; 0.00 0.62 0.45; 0.34 0.71 0.91; 0.90 0.62 0.00;
           0.84 0.37 0.00; 0.80 0.47 0.65; 0.00 0.00 0.00];
markers = {'o', 's', '^', 'd', 'v', 'p', 'h'};
nDelays = numel(delays);
colourOf = @(i) palette(mod(i - 1, size(palette, 1)) + 1, :);
markerOf = @(i) markers{mod(i - 1, numel(markers)) + 1};
dailyAngle  = 360 / 24;                     % 15 degrees is a 24-hour rhythm
weeklyAngle = 360 / 168;                    % 2.14 degrees is a 168-hour rhythm

% ---- persistence against rhythm ---------------------------------------
axesHandle = nexttile(tiles);
hold(axesHandle, 'on');
yline(axesHandle, 1, '-', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.4, ...
    'DisplayName', 'Unit circle');
xline(axesHandle, dailyAngle, '--', 'Color', [0.45 0.45 0.45], ...
    'LineWidth', 1.2, 'DisplayName', '24 h rhythm');
xline(axesHandle, weeklyAngle, '-.', 'Color', [0.60 0.35 0.60], ...
    'LineWidth', 1.2, 'DisplayName', '168 h (one week)');
nReal = 0;
for delayIndex = 1:nDelays
    z = poles{delayIndex};
    z = z(imag(z) >= 0);                    % conjugates repeat the rhythm
    nReal = nReal + sum(abs(imag(z)) < 1e-9);
    z = z(imag(z) > 1e-9);                  % a real pole has no rhythm to place
    plot(axesHandle, abs(angle(z)) * 180/pi, abs(z), markerOf(delayIndex), ...
        'Color', colourOf(delayIndex), 'MarkerSize', 5, 'LineWidth', 1.1, ...
        'DisplayName', sprintf('%d taps', delays(delayIndex)));
end
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
% Log angle, because the rhythms worth naming are a factor of seven apart:
% a week is 2.1 degrees and a day is 15, and on a linear 0-180 axis the
% weekly modes would sit on top of the y-axis.
set(axesHandle, 'XScale', 'log');
xlim(axesHandle, [1.4, 200]);
xticks(axesHandle, [weeklyAngle 5 dailyAngle 30 60 90 180]);
xticklabels(axesHandle, {'2.14','5','15','30','60','90','180'});
ylim(axesHandle, [0, 1.08]);
xlabel(axesHandle, 'Angle of the pole (degrees, log scale)');
ylabel(axesHandle, 'Persistence $|z|$', 'Interpreter', 'latex');
title(axesHandle, 'Every mode: how fast it repeats, how long it lasts');
subtitle(axesHandle, sprintf(['left is slow, high is persistent; a week is ', ...
    '2.1$^\\circ$ and a day 15$^\\circ$. %d real poles ($\\theta = 0$) ', ...
    'are not shown'], nReal), 'Interpreter', 'latex');
legend(axesHandle, 'Location', 'southoutside', 'NumColumns', 5, ...
    'FontSize', 7.5, 'EdgeColor', [0.70, 0.70, 0.70]);

% ---- the periods themselves, delay by delay ---------------------------
axesHandle = nexttile(tiles);
hold(axesHandle, 'on');
yline(axesHandle, 24, '--', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.4, ...
    'DisplayName', 'One day');
yline(axesHandle, 12, ':', 'Color', [0.65 0.65 0.65], 'LineWidth', 1.2, ...
    'DisplayName', 'Half day');
yline(axesHandle, 168, '-.', 'Color', [0.60 0.35 0.60], 'LineWidth', 1.4, ...
    'DisplayName', 'One week');
for delayIndex = 1:nDelays
    z = poles{delayIndex};
    z = z(imag(z) > 1e-9);                  % a real pole has no period
    period = 360 ./ (abs(angle(z)) * 180/pi);
    % marker area grows with persistence, so a mode that survives the day
    % reads louder than one that has decayed by the next hour.
    scatter(axesHandle, delays(delayIndex) + zeros(size(period)), period, ...
        8 + 90 * max(abs(z) - 0.5, 0).^2, colourOf(delayIndex), ...
        markerOf(delayIndex), 'LineWidth', 1.0, 'HandleVisibility', 'off');
end
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
set(axesHandle, 'YScale', 'log');
xticks(axesHandle, delays);
xlim(axesHandle, [delays(1) - 3, delays(end) + 3]);
ylim(axesHandle, [1.8, 300]);
yticks(axesHandle, [2 4 6 12 24 48 96 168 240]);
xlabel(axesHandle, 'Tapped-delay length (hours)');
ylabel(axesHandle, 'Period of the mode (hours, log)');
title(axesHandle, 'Which rhythms each network actually carries');
subtitle(axesHandle, sprintf(['marker size grows with $|z|$; %d real poles ', ...
    'have no period and are not shown'], nReal), 'Interpreter', 'latex');
legend(axesHandle, 'Location', 'southoutside', 'NumColumns', 3, ...
    'FontSize', 8, 'EdgeColor', [0.70, 0.70, 0.70]);

title(tiles, 'The rhythms a linear ARX learns, delay by delay', ...
    'FontWeight', 'bold');
subtitle(tiles, sprintf('fitted on the %s split', string(trainSplit)));
exportgraphics(figureHandle, figureFile, 'Resolution', 300);
close(figureHandle);
end

function text = format_number(value)
%FORMAT_NUMBER Add thousands separators without locale dependence.
text = regexprep(sprintf('%d', value), '(?<!^)(?=(\d{3})+$)', ',');
end
