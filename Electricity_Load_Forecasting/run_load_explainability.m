function results = run_load_explainability(options)
%RUN_LOAD_EXPLAINABILITY Explain the closed-loop NARX load forecast.
%
%   RESULTS = RUN_LOAD_EXPLAINABILITY() attributes the parallel
%   (closed-loop) 24-hour load forecast a^2(T+k) = yHat(T+k) of the trained
%   checkpoints in train/load to the measured quantities that produced it,
%   using the Integrated Gradients engine in Explainability/narxExplain.m.
%   Nothing in the NARX model, its training code, or the attribution engine
%   is modified: checkpoints are loaded, frozen and differentiated.
%
%   The delay experiment left two questions that an error curve cannot
%   answer, and this driver answers both by measurement.
%
%   1. Tapped-delay lengths of 12, 18, 24 and 30 hours gave held-out errors
%      within 2.5 MW of each other. Are those four networks doing the same
%      thing? How far back does each actually reach?
%   2. The hourly demand repeats every 24 hours, and the 24-tap checkpoint
%      has the lowest error exactly at k = 24. Does the network actually
%      read the demand one day before the hour it is predicting, or does it
%      only extrapolate the last few hours?
%
%   Question 2 is answered with a permutation test. For a target at T+k,
%   the tap holding the same clock hour one day earlier is at time offset
%   j = k - 24. Relevance is aligned on that moving offset and pooled over
%   horizons; the null distribution comes from permuting k, which destroys
%   the pairing between horizon and offset while leaving both marginals
%   untouched.
%
%   NOTATION (Hagan, per AGENTS.md)
%     p(t)  external input: Tocumen air temperature, degrees C
%     y(t)  measured target: national hourly demand, MW
%     a^2(T+k) = yHat(T+k)   closed-loop prediction, k hours past origin T
%     j     time offset from the origin T; lag is -j for j <= 0
%
%   REFERENCES THE FORECAST IS EXPLAINED AGAINST
%   A regression model has no decision boundary, so the point of comparison
%   must be supplied and it encodes the question being asked. Two are used.
%
%     'yesterday'  (primary) Every measured tap is replaced by its value 24
%                  hours earlier: the baseline window is the same window
%                  shifted back one day. The conserved total is then
%                  yHat(T+k) minus the forecast the same frozen network
%                  would have made from yesterday's data, so the
%                  attribution answers "why does the model expect today to
%                  differ from yesterday at this hour?" -- the question the
%                  daily-naive reference of the delay experiment poses. It
%                  is a recorded window, so the straight-line path stays
%                  close to the data manifold, and no coordinate equals its
%                  own baseline, so there are no structural zeros.
%     'trainmean'  Every tap held at the training-window mean: a flat day
%                  at the average level. The daily shape is absent from
%                  this reference, so the model has to build it. Used to
%                  confirm that the daily-alignment result is a property of
%                  the network and not of the reference.
%
%   FIGURES
%     1  Receptive field over the day: relevance across (k, j) for the
%        demand and temperature channels, with the daily offset j = k - 24
%        drawn on it, beside the error curve.
%     2  Does the network read yesterday at this hour? Daily-alignment
%        profile against a permutation null, under both references and
%        across checkpoints.
%     3  Effective memory: relevance by lag and its cumulative share, per
%        checkpoint. The measured version of the delay sweep.
%     4  What the forecast leans on, by horizon and by time of day, split
%        into past demand, past temperature and future temperature.
%     5  Why today is not yesterday: one 24-hour forecast decomposed, with
%        a conservation waterfall in MW.
%     6  Validation: sim parity, completeness, analytic versus
%        central-difference gradients, and an occlusion cross-check.
%
%   NAME-VALUE OPTIONS
%     Delay               Primary checkpoint. Default 24.
%     CompareDelays       Checkpoints for figures 2 and 3.
%                         Default [12 18 30 48].
%     PredictionHorizon   H, hours. Default 24.
%     NumWindows          Attribution windows, primary checkpoint. Default
%                         240, drawn balanced across the 24 clock hours.
%     NumCompareWindows   Windows per comparison checkpoint. Default 192.
%     NumTrainMeanWindows Windows for the trainmean cross-check. Default 144.
%     NumErrorWindows     Windows for the error curve only. Default 2000.
%     NumValidationWindows Windows carrying occlusion and the gradient
%                         check. Default 24.
%     IntegrationSteps    Starting M. Default 128.
%     MaxIntegrationSteps Cap on the adaptive refinement of M for the
%                         yesterday reference. Default 2048: that path is
%                         short and on-manifold, and converges there.
%     MaxIntegrationStepsTrainMean Cap for the training-mean reference,
%                         whose path runs from a flat day to a real one and
%                         needs far more samples. Default 8192.
%     NumPermutations     Permutations in the daily-alignment null. 2000.
%     TrainSplit/TestSplit  Defaults 'dry-train' and 'dry-test'.
%     Recompute           Ignore cached attribution runs. Default false.
%
%   Attribution is expensive and the refinement of M makes it more so, so
%   every run is cached under cache/ and reused. An interrupted session
%   therefore resumes: call the function again and only the missing stages
%   are computed.
%
%   See also NARXEXPLAIN, NARXFORECAST, SAMPLE_LOAD_WINDOWS,
%   RUN_LOAD_DELAY_EXPERIMENT.

arguments
    options.Delay (1,1) double {mustBePositive} = 24
    options.CompareDelays (1,:) double = [12 18 30 48]
    options.PredictionHorizon (1,1) double {mustBePositive} = 24
    options.NumWindows (1,1) double {mustBePositive} = 240
    options.NumCompareWindows (1,1) double {mustBePositive} = 192
    options.NumTrainMeanWindows (1,1) double {mustBePositive} = 144
    options.NumErrorWindows (1,1) double {mustBePositive} = 2000
    options.NumValidationWindows (1,1) double {mustBePositive} = 24
    options.IntegrationSteps (1,1) double {mustBePositive} = 128
    options.MaxIntegrationSteps (1,1) double {mustBePositive} = 2048
    options.MaxIntegrationStepsTrainMean (1,1) double {mustBePositive} = 8192
    options.NumPermutations (1,1) double {mustBePositive} = 2000
    options.TrainSplit = 'dry-train'
    options.TestSplit = 'dry-test'
    options.RandomSeed (1,1) double = 2024
    options.Recompute (1,1) logical = false
end

scriptFolder = fileparts(mfilename('fullpath'));
projectFolder = fileparts(scriptFolder);
addpath(projectFolder);
addpath(scriptFolder);
addpath(fullfile(projectFolder, 'Explainability'));

cacheFolder = fullfile(scriptFolder, 'cache');
if ~isfolder(cacheFolder)
    mkdir(cacheFolder);
end

style = figureStyle();
H = options.PredictionHorizon;
dailyPeriod = 24;
allDelays = [options.Delay, options.CompareDelays];

fprintf('\n=== NARX explainability: hourly load, %d-tap primary ===\n', ...
    options.Delay);

% =====================================================================
% 1. Checkpoints, taken from the delay experiment's own selection
% =====================================================================
sweepFile = fullfile(scriptFolder, 'load_narx_delay_experiment_results.mat');
sweep = [];
if isfile(sweepFile)
    loaded = load(sweepFile, 'results');
    sweep = loaded.results;
end

checkpoint = struct('delay', {}, 'restart', {}, 'model', {}, 'label', {});
for delayIndex = 1:numel(allDelays)
    delay = allDelays(delayIndex);
    [model, restart] = loadLoadCheckpoint(scriptFolder, delay, sweep);
    checkpoint(delayIndex).delay = delay;
    checkpoint(delayIndex).restart = restart;
    checkpoint(delayIndex).model = model;
    checkpoint(delayIndex).label = sprintf('%d taps', delay);
    fprintf('  checkpoint %2d taps (run %d), %d weights\n', ...
        delay, restart, numel(getwb(model.narx)));
end
primary = checkpoint(1).model;

% =====================================================================
% 2. Windows: prediction origins balanced across the 24 clock hours
%
%    Balance matters here. The daily-alignment test compares relevance at
%    a moving offset j = k - 24 against a permutation null, and it would be
%    confounded if the size of the difference from the reference at a given
%    offset were itself tied to the horizon. Spreading the origins evenly
%    over the clock removes that coupling by construction.
%
%    Every origin also carries a full extra day of history, so the same
%    origin can be explained against its own window shifted back 24 hours.
% =====================================================================
[~, ~, timeStamps] = import_data_load(options.TestSplit);
originFloor = max(allDelays) + dailyPeriod;
origins = stratifiedOrigins(timeStamps, originFloor, H, ...
    options.NumWindows, options.RandomSeed);
compareOrigins = stratifiedOrigins(timeStamps, originFloor, H, ...
    options.NumCompareWindows, options.RandomSeed);
trainMeanOrigins = stratifiedOrigins(timeStamps, originFloor, H, ...
    options.NumTrainMeanWindows, options.RandomSeed);
originHour = hour(timeStamps(origins));
fprintf('  %d origins on %s, %d per clock hour, %s to %s\n', ...
    numel(origins), string(options.TestSplit), numel(origins) / 24, ...
    string(min(timeStamps(origins))), string(max(timeStamps(origins))));

[pTrain, yTrain] = import_data_load(options.TrainSplit);
trainMeanP = mean(pTrain);
trainMeanY = mean(yTrain);

% =====================================================================
% 3. Attribution runs, cached
% =====================================================================
% Occlusion is computed alongside every attribution. It costs one rollout
% per coordinate, next to nothing beside the integral, and it is the
% independent probe the headline result is checked against: setting one tap
% back to its yesterday value and re-rolling asks the same question as the
% attribution without sharing any of its machinery.
explainOptions = {'IntegrationSteps', options.IntegrationSteps, ...
    'MaxIntegrationSteps', options.MaxIntegrationSteps, ...
    'PredictionHorizon', H, 'ComputeOcclusion', true};

main = cachedExplain(cacheFolder, options.Recompute, ...
    sprintf('yesterday_%ddelay_%dwin_occ', options.Delay, numel(origins)), ...
    @() explainAgainstYesterday(primary, origins, options.TestSplit, ...
        H, dailyPeriod, explainOptions));

trainMeanOptions = [explainOptions(1:2), ...
    {'MaxIntegrationSteps', options.MaxIntegrationStepsTrainMean}, ...
    explainOptions(5:end)];
trainMeanRun = cachedExplain(cacheFolder, options.Recompute, ...
    sprintf('trainmean_%ddelay_%dwin_occ', options.Delay, numel(trainMeanOrigins)), ...
    @() explainAgainstTrainMean(primary, trainMeanOrigins, ...
        options.TestSplit, H, trainMeanP, trainMeanY, trainMeanOptions));

compareRuns = cell(1, numel(checkpoint));
for delayIndex = 1:numel(checkpoint)
    delay = checkpoint(delayIndex).delay;
    if delayIndex == 1 && numel(origins) == numel(compareOrigins)
        compareRuns{delayIndex} = main;
        continue;
    end
    compareRuns{delayIndex} = cachedExplain(cacheFolder, options.Recompute, ...
        sprintf('yesterday_%ddelay_%dwin_occ', delay, numel(compareOrigins)), ...
        @() explainAgainstYesterday(checkpoint(delayIndex).model, ...
            compareOrigins, options.TestSplit, H, dailyPeriod, explainOptions));
end

validation = cachedExplain(cacheFolder, options.Recompute, ...
    sprintf('validation_%ddelay_%dwin_occ', options.Delay, options.NumValidationWindows), ...
    @() explainAgainstYesterday(primary, ...
        origins(1:options.NumValidationWindows), options.TestSplit, H, ...
        dailyPeriod, [explainOptions, {'ValidateGradient', true}]));

% =====================================================================
% 4. The error curve the attribution is read against
% =====================================================================
errorOrigins = stratifiedOrigins(timeStamps, originFloor, H, ...
    options.NumErrorWindows, options.RandomSeed + 1);
[pError, yError] = sample_load_windows('Delay', options.Delay, ...
    'PredictionHorizon', H, 'Split', options.TestSplit, ...
    'Origins', errorOrigins);
[~, ~, mae] = narxForecast(primary, pError, yError, H);
naiveMae = dailyNaiveMae(options.TestSplit, errorOrigins, H, dailyPeriod);

% =====================================================================
% 5. Analyses
% =====================================================================
alignment = struct();
alignment.yesterday = dailyAlignment(main, dailyPeriod, ...
    options.NumPermutations, options.RandomSeed);
alignment.trainmean = dailyAlignment(trainMeanRun, dailyPeriod, ...
    options.NumPermutations, options.RandomSeed);
alignment.occlusion = dailyAlignment(main, dailyPeriod, ...
    options.NumPermutations, options.RandomSeed, 'occlusion');
% Horizons every checkpoint in the comparison can actually reach.
commonFirstHorizon = max(1, dailyPeriod + 1 - min(allDelays));
commonRange = [commonFirstHorizon, H];
alignment.commonRange = commonRange;
alignment.byCheckpoint = cell(1, numel(checkpoint));
alignment.byCheckpointOcclusion = cell(1, numel(checkpoint));
for delayIndex = 1:numel(checkpoint)
    alignment.byCheckpoint{delayIndex} = dailyAlignment( ...
        compareRuns{delayIndex}, dailyPeriod, options.NumPermutations, ...
        options.RandomSeed, 'relevance', commonRange);
    alignment.byCheckpointOcclusion{delayIndex} = dailyAlignment( ...
        compareRuns{delayIndex}, dailyPeriod, options.NumPermutations, ...
        options.RandomSeed, 'occlusion', commonRange);
end

fprintf('\nDaily-alignment test (relevance at j = k - 24, against a k-permutation null)\n');
fprintf('  %-22s enrichment %.3f, null 95%% [%.3f, %.3f], p = %.4f\n', ...
    'yesterday reference', alignment.yesterday.enrichment, ...
    alignment.yesterday.nullLow, alignment.yesterday.nullHigh, ...
    alignment.yesterday.pValue);
fprintf('  %-22s enrichment %.3f, null 95%% [%.3f, %.3f], p = %.4f\n', ...
    'trainmean reference', alignment.trainmean.enrichment, ...
    alignment.trainmean.nullLow, alignment.trainmean.nullHigh, ...
    alignment.trainmean.pValue);
fprintf('  %-22s enrichment %.3f, null 95%% [%.3f, %.3f], p = %.4f\n', ...
    'occlusion cross-check', alignment.occlusion.enrichment, ...
    alignment.occlusion.nullLow, alignment.occlusion.nullHigh, ...
    alignment.occlusion.pValue);
fprintf('  on horizons k = %d..%d, which every checkpoint can reach:\n', ...
    commonRange(1), commonRange(2));
for delayIndex = 1:numel(checkpoint)
    stat = alignment.byCheckpoint{delayIndex};
    occlusionStat = alignment.byCheckpointOcclusion{delayIndex};
    fprintf(['    %-9s IG %.3f (p = %.4f), occlusion %.3f (p = %.4f), ', ...
        'null 95%% [%.3f, %.3f]\n'], checkpoint(delayIndex).label, ...
        stat.enrichment, stat.pValue, occlusionStat.enrichment, ...
        occlusionStat.pValue, stat.nullLow, stat.nullHigh);
end

memory = cell(1, numel(checkpoint));
for delayIndex = 1:numel(checkpoint)
    memory{delayIndex} = memoryProfile(compareRuns{delayIndex});
    memory{delayIndex}.label = checkpoint(delayIndex).label;
    memory{delayIndex}.delay = checkpoint(delayIndex).delay;
end
fprintf('\nEffective memory (share of loading relevance within a lag)\n');
for delayIndex = 1:numel(checkpoint)
    fprintf('  %-10s half of the relevance within %g h, 90%% within %g h\n', ...
        memory{delayIndex}.label, memory{delayIndex}.lagAtHalf, ...
        memory{delayIndex}.lagAt90);
end

diagnostic = horizonDiagnostics(main);
fprintf(['\nIntegrated Gradients against occlusion: r = %.2f at k = 1, ', ...
    '%.2f at k = %d; last horizon with r >= 0.5 is k = %d\n'], ...
    diagnostic.agreement(1), diagnostic.agreement(end), ...
    numel(diagnostic.agreement), diagnostic.lastAgreeingHorizon);
fprintf(['Cancellation, median sum|R| / |sum R|: %.0f at k = 1, ', ...
    '%.0f at k = %d\n'], diagnostic.cancellation(1), ...
    diagnostic.cancellation(end), numel(diagnostic.cancellation));

channel = channelShares(main, originHour);
fprintf('\nChannel shares over the whole horizon: demand taps %.1f%%, past temperature %.1f%%, future temperature %.1f%%\n', ...
    100 * channel.overall(1), 100 * channel.overall(2), 100 * channel.overall(3));

% =====================================================================
% 6. Figures
% =====================================================================
figureFiles = strings(1, 6);
figureFiles(1) = makeReceptiveFieldFigure(scriptFolder, main, mae, ...
    naiveMae, dailyPeriod, style);
figureFiles(2) = makeAlignmentFigure(scriptFolder, alignment, checkpoint, ...
    dailyPeriod, style);
figureFiles(3) = makeMemoryFigure(scriptFolder, memory, style);
figureFiles(4) = makeChannelFigure(scriptFolder, channel, mae, style);
figureFiles(5) = makeYesterdayFigure(scriptFolder, main, timeStamps, ...
    origins, dailyPeriod, style);
figureFiles(6) = makeValidationFigure(scriptFolder, validation, ...
    diagnostic, style);

% =====================================================================
% 7. Results
% =====================================================================
results = struct;
results.options = options;
results.delays = allDelays;
results.checkpointRestart = [checkpoint.restart];
results.horizon = main.horizon;
results.mae = mae;
results.naiveMae = naiveMae;
results.origins = origins;
results.originDatetime = timeStamps(origins);
results.alignment = alignment;
results.memory = memory;
results.channel = channel;
results.diagnostic = diagnostic;
results.completeness = struct( ...
    'yesterdayAbsolute', main.completenessAbsoluteError, ...
    'yesterdayRelative', main.completenessRelativeError, ...
    'integrationSteps', main.integrationSteps, ...
    'trustworthy', main.nTrustworthy, 'windows', main.nWindows, ...
    'simParity', main.simParityError, ...
    'gradientCheck', validation.gradientCheck.maxAbsoluteError);
results.meanAbsRelevance = meanAbsoluteRelevance(main);
results.coordinate = main.coordinate;
results.figures = figureFiles;

resultsFile = fullfile(scriptFolder, ...
    sprintf('load_%ddelay_explainability.mat', options.Delay));
save(resultsFile, 'results');

fprintf('\nSaved results : %s\n', resultsFile);
for figureIndex = 1:numel(figureFiles)
    fprintf('Saved figure  : %s\n', figureFiles(figureIndex));
end
end

% =====================================================================
% Checkpoints, windows and cached attribution runs
% =====================================================================
function [narxModel, restart] = loadLoadCheckpoint(scriptFolder, delay, sweep)
%LOADLOADCHECKPOINT Load the run the delay experiment selected for a delay.
%   The sweep fitted several networks per delay and kept the one with the
%   lowest error on the validation season. Explaining a different run would
%   explain a network the experiment did not report.

restart = 1;
neurons = 10;
trainK = 24;
if ~isempty(sweep)
    delayIndex = find(sweep.delays == delay, 1);
    if ~isempty(delayIndex)
        restart = sweep.selectedRestart(delayIndex);
        neurons = sweep.options.Neurons;
        trainK = sweep.options.TrainK;
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
assert(narxModel.delay == delay, ...
    'Checkpoint in %s has delay %d, not %d.', folder, narxModel.delay, delay);
end

function origins = stratifiedOrigins(timeStamps, originFloor, ...
    predictionHorizon, nWindows, seed)
%STRATIFIEDORIGINS Prediction origins spread evenly over the 24 clock hours.

candidates = originFloor:(numel(timeStamps) - predictionHorizon);
candidateHour = hour(timeStamps(candidates));
perHour = max(1, floor(nWindows / 24));

rng(seed, 'twister');
origins = zeros(1, 24 * perHour);
for clockHour = 0:23
    pool = candidates(candidateHour == clockHour);
    assert(numel(pool) >= perHour, ...
        'Only %d origins available at hour %d; %d requested.', ...
        numel(pool), clockHour, perHour);
    selected = pool(randperm(numel(pool), perHour));
    origins(clockHour * perHour + (1:perHour)) = selected;
end
origins = sort(origins);
end

function result = cachedExplain(cacheFolder, recompute, key, computeFcn)
%CACHEDEXPLAIN Run an attribution once and keep it.
%   Integrated Gradients on a 24-tap checkpoint refines M into the
%   thousands and costs minutes per hundred windows, so a run is saved and
%   reused. An interrupted session resumes from whatever finished.

cacheFile = fullfile(cacheFolder, sprintf('%s.mat', key));
if ~recompute && isfile(cacheFile)
    try
        loaded = load(cacheFile, 'result');
        result = loaded.result;
        fprintf('  cached   %s (M = %d, %d/%d windows retained)\n', key, ...
            result.integrationSteps, result.nTrustworthy, result.nWindows);
        return;
    catch cacheError
        warning('run_load_explainability:cache', ...
            'Ignoring unreadable cache file %s (%s); recomputing.', ...
            cacheFile, cacheError.message);
    end
end

fprintf('  computing %s ...\n', key);
runTimer = tic;
result = computeFcn();
fprintf('    done in %s: M = %d, %d/%d windows retained, sim parity %.2e\n', ...
    secs2hms(toc(runTimer)), result.integrationSteps, ...
    result.nTrustworthy, result.nWindows, result.simParityError);
% Default format, not -v7.3: the HDF5 writer is unreliable over the
% network path this project is often opened through.
save(cacheFile, 'result');
end

function result = explainAgainstYesterday(narxModel, origins, split, ...
    predictionHorizon, dailyPeriod, explainOptions)
%EXPLAINAGAINSTYESTERDAY Reference is the same window one day earlier.
%   The baseline is a recorded window, so the straight line the integral
%   runs along stays close to trajectories the grid actually produced, and
%   no coordinate equals its own baseline value, so no coordinate is a
%   structural zero. The conserved total becomes yHat(T+k) minus the
%   forecast this same frozen network would have made 24 hours earlier.

delay = narxModel.delay;
[pWindows, yWindows] = sample_load_windows('Delay', delay, ...
    'PredictionHorizon', predictionHorizon, 'Split', split, ...
    'Origins', origins);
[pBaseline, yBaseline] = sample_load_windows('Delay', delay, ...
    'PredictionHorizon', predictionHorizon, 'Split', split, ...
    'Origins', origins - dailyPeriod);

result = narxExplain(narxModel, pWindows, yWindows, ...
    'Baseline', 'custom', ...
    'CustomBaselineP', pBaseline, 'CustomBaselineY', yBaseline, ...
    explainOptions{:});
result.referenceName = 'yesterday';
result.origins = origins;
end

function result = explainAgainstTrainMean(narxModel, origins, split, ...
    predictionHorizon, trainMeanP, trainMeanY, explainOptions)
%EXPLAINAGAINSTTRAINMEAN Reference is a flat day at the training mean.

delay = narxModel.delay;
[pWindows, yWindows] = sample_load_windows('Delay', delay, ...
    'PredictionHorizon', predictionHorizon, 'Split', split, ...
    'Origins', origins);

result = narxExplain(narxModel, pWindows, yWindows, ...
    'Baseline', 'trainmean', ...
    'TrainMeanP', trainMeanP, 'TrainMeanY', trainMeanY, ...
    explainOptions{:});
result.referenceName = 'trainmean';
result.origins = origins;
end

function mae = dailyNaiveMae(split, origins, predictionHorizon, dailyPeriod)
%DAILYNAIVEMAE Error of yHat(T+k) = y(T+k-24), the reference to beat.

[~, y] = import_data_load(split);
targetIndex = (1:predictionHorizon).' + origins;
measured = y(targetIndex);
mae = mean(abs(measured - y(targetIndex - dailyPeriod)), 2);
end

% =====================================================================
% Analyses
% =====================================================================
function meanAbs = meanAbsoluteRelevance(result)
%MEANABSOLUTERELEVANCE Mean |relevance| over the windows worth averaging.
%   A window that fails the completeness tolerance has relevances that do
%   not sum to the difference they claim to explain, so it is excluded
%   rather than quietly averaged in.
assert(result.nTrustworthy > 0, ...
    ['No window passed the completeness tolerance. Raise ', ...
     'MaxIntegrationSteps, or use a reference whose straight-line path ', ...
     'stays closer to the data.']);
meanAbs = mean(abs(result.relevance(:, :, result.trustworthy)), 3);
end

function stat = dailyAlignment(result, dailyPeriod, nPermutations, seed, ...
    source, horizonRange)
%DAILYALIGNMENT Is relevance enriched one day before the predicted hour?
%
%   For a target at T+k, the demand tap holding the same clock hour one day
%   earlier sits at time offset j = k - dailyPeriod. Relevance is pooled
%   over horizons along that moving offset, at displacements delta from it,
%   after normalising each horizon by its own mean so that no horizon
%   dominates by scale alone.
%
%   The null keeps every marginal and destroys only the pairing: horizons
%   are permuted before the same alignment is taken. An enrichment above
%   the null band therefore cannot be produced by relevance simply being
%   larger near the origin, which is what makes the test worth running.

if nargin < 5 || isempty(source)
    source = 'relevance';
end
if nargin < 6 || isempty(horizonRange)
    horizonRange = [1, result.predictionHorizon];
end
coordinate = result.coordinate;
switch source
    case 'relevance'
        meanAbs = meanAbsoluteRelevance(result);
    case 'occlusion'
        % The same question asked without the integral: replace one tap by
        % its yesterday value, re-roll every later step, and record how far
        % the forecast moves.
        meanAbs = mean(result.occlusion.magnitude(:, :, result.trustworthy), 3);
    otherwise
        error('Unknown alignment source ''%s''.', source);
end
profile = meanAbs(:, coordinate.yIndex);
offsets = coordinate.yOffsets;
profile = profile ./ mean(profile, 2);

% A tapped delay line of na hours holds the daily tap j = k - dailyPeriod
% only for k >= dailyPeriod + 1 - na. Comparing checkpoints of different
% length therefore has to be done on horizons all of them can reach, or the
% shorter lines are scored on fewer horizons than the longer ones.
horizons = max(horizonRange(1), 1):min(horizonRange(2), size(profile, 1));
profile = profile(horizons, :);

displacement = -8:8;
observed = alignedProfile(profile, offsets, horizons, ...
    dailyPeriod, displacement);

rng(seed, 'twister');
nullProfiles = zeros(nPermutations, numel(displacement));
for permutationIndex = 1:nPermutations
    nullProfiles(permutationIndex, :) = alignedProfile(profile, offsets, ...
        horizons(randperm(numel(horizons))), dailyPeriod, displacement);
end

centreIndex = find(displacement == 0, 1);
nullCentre = nullProfiles(:, centreIndex);

stat.displacement = displacement;
stat.observed = observed;
stat.nullLowBand = quantileNoToolbox(nullProfiles, 0.025);
stat.nullMedianBand = quantileNoToolbox(nullProfiles, 0.5);
stat.nullHighBand = quantileNoToolbox(nullProfiles, 0.975);
stat.enrichment = observed(centreIndex);
stat.nullLow = stat.nullLowBand(centreIndex);
stat.nullHigh = stat.nullHighBand(centreIndex);
stat.nullMedian = stat.nullMedianBand(centreIndex);
% One-sided: the hypothesis is that the daily tap carries more relevance
% than chance, not merely a different amount.
stat.pValue = (1 + sum(nullCentre >= stat.enrichment)) / (1 + nPermutations);
stat.nPermutations = nPermutations;
stat.nWindows = result.nTrustworthy;
stat.delay = result.delay;
stat.source = source;
stat.horizons = horizons;
[~, peakIndex] = max(observed);
stat.peakDisplacement = displacement(peakIndex);
% Horizons at which the daily tap exists at all: a tapped delay line of
% na hours reaches j = k - 24 only when k - 24 >= 1 - na.
stat.firstReachableHorizon = max(1, dailyPeriod + 1 - result.delay);
end

function aligned = alignedProfile(profile, offsets, horizonOrder, ...
    dailyPeriod, displacement)
%ALIGNEDPROFILE Average relevance at offset (k' - dailyPeriod) + delta.
%   horizonOrder supplies the k' paired with each row, so passing the rows'
%   own horizons gives the observed alignment and passing a shuffle of them
%   gives a draw from the null.

nHorizons = size(profile, 1);
aligned = nan(1, numel(displacement));
for displacementIndex = 1:numel(displacement)
    accumulated = 0;
    counted = 0;
    for horizon = 1:nHorizons
        offset = horizonOrder(horizon) - dailyPeriod + ...
            displacement(displacementIndex);
        column = find(offsets == offset, 1);
        if ~isempty(column)
            accumulated = accumulated + profile(horizon, column);
            counted = counted + 1;
        end
    end
    if counted > 0
        aligned(displacementIndex) = accumulated / counted;
    end
end
end

function profile = memoryProfile(result)
%MEMORYPROFILE Relevance by lag, and how much of it sits within each lag.
%   Tapped-delay length is otherwise a trial-and-error hyperparameter. The
%   cumulative share turns it into a measurement: the lag containing half
%   the loading relevance is the memory the trained network actually uses.

coordinate = result.coordinate;
meanAbs = meanAbsoluteRelevance(result);
loadingColumns = [coordinate.yIndex, coordinate.pPastIndex];
loadingLags = 1 - coordinate.offset(loadingColumns);
lags = unique(loadingLags);

byLag = zeros(size(lags));
for lagIndex = 1:numel(lags)
    selected = loadingColumns(loadingLags == lags(lagIndex));
    byLag(lagIndex) = mean(sum(meanAbs(:, selected), 2));
end

profile.lags = lags;
profile.relevance = byLag;
profile.normalized = byLag / max(byLag);
profile.cumulativeShare = cumsum(byLag) / sum(byLag);
profile.lagAtHalf = lags(find(profile.cumulativeShare >= 0.5, 1));
profile.lagAt90 = lags(find(profile.cumulativeShare >= 0.9, 1));
profile.shareBeyond12h = 1 - profile.cumulativeShare( ...
    find(lags >= 12, 1, 'first') - 1);
end

function channel = channelShares(result, originHour)
%CHANNELSHARES How the forecast divides between its three input channels.
%   Past demand y(T+j), past temperature p(T+j) and the future temperature
%   p(T+j), j >= 1, that the model is handed. The last one matters
%   operationally: a real forecast has to predict that temperature, so any
%   share resting on it is a share resting on perfect weather foresight.

coordinate = result.coordinate;
groups = {coordinate.yIndex, coordinate.pPastIndex, coordinate.pFutureIndex};
channel.names = {'Past demand y(T+j)', 'Past temperature p(T+j)', ...
    'Future temperature p(T+j)'};

meanAbs = meanAbsoluteRelevance(result);
byHorizon = zeros(size(meanAbs, 1), 3);
for groupIndex = 1:3
    byHorizon(:, groupIndex) = sum(meanAbs(:, groups{groupIndex}), 2);
end
channel.byHorizon = byHorizon ./ sum(byHorizon, 2);
channel.overall = sum(byHorizon, 1) / sum(byHorizon, 'all');

% Same split, conditioned on the clock hour being predicted rather than on
% the origin. Every (window, horizon) pair carries a target hour, so this
% pools 24 times as many samples as conditioning on the origin would, and
% it asks the question that matters operationally: what is the forecast
% leaning on when it predicts the afternoon peak, or the overnight trough?
keep = result.trustworthy;
originHour = originHour(keep);
relevance = abs(result.relevance(:, :, keep));
predictionHorizon = size(relevance, 1);
nKept = numel(originHour);

byChannel = zeros(predictionHorizon, nKept, 3);
for groupIndex = 1:3
    byChannel(:, :, groupIndex) = ...
        reshape(sum(relevance(:, groups{groupIndex}, :), 2), ...
        predictionHorizon, nKept);
end
targetHour = mod(originHour(:).' + (1:predictionHorizon).', 24);

hours = 0:23;
byHour = zeros(24, 3);
count = zeros(24, 1);
for hourIndex = 1:24
    selected = targetHour == hours(hourIndex);
    count(hourIndex) = sum(selected(:));
    for groupIndex = 1:3
        page = byChannel(:, :, groupIndex);
        byHour(hourIndex, groupIndex) = mean(page(selected));
    end
end
channel.hours = hours;
channel.byTargetHour = byHour ./ sum(byHour, 2);
channel.nPerTargetHour = count;
end

function diagnostic = horizonDiagnostics(result)
%HORIZONDIAGNOSTICS How far the attribution can be pushed, horizon by horizon.
%
%   Two numbers per horizon k.
%
%   AGREEMENT is the correlation between the Integrated Gradients relevance
%   and the occlusion effect over every coordinate and window. They answer
%   the same question by different means, so they should agree; where they
%   stop agreeing, the per-coordinate ranking has stopped being safe to
%   read even though completeness still holds exactly.
%
%   CANCELLATION is sum_i |R_i(k)| divided by |sum_i R_i(k)|. Completeness
%   fixes the sum, not the pieces: a value of 50 means the individual
%   contributions are fifty times larger than the net difference they
%   explain and therefore almost entirely cancel. A decomposition like that
%   is arithmetically correct and still a poor thing to read one coordinate
%   at a time.

keep = result.trustworthy;
relevance = result.relevance(:, :, keep);
occlusion = result.occlusion.signed(:, :, keep);
predictionHorizon = size(relevance, 1);

agreement = zeros(predictionHorizon, 1);
cancellation = zeros(predictionHorizon, 1);
for horizon = 1:predictionHorizon
    igRow = reshape(relevance(horizon, :, :), [], 1);
    occlusionRow = reshape(occlusion(horizon, :, :), [], 1);
    agreement(horizon) = correlationNoToolbox(igRow, occlusionRow);
    ratio = sum(abs(relevance(horizon, :, :)), 2) ./ ...
        max(abs(sum(relevance(horizon, :, :), 2)), eps);
    cancellation(horizon) = median(ratio(:));
end

diagnostic.horizon = (1:predictionHorizon).';
diagnostic.agreement = agreement;
diagnostic.cancellation = cancellation;
diagnostic.lastAgreeingHorizon = find(agreement >= 0.5, 1, 'last');
if isempty(diagnostic.lastAgreeingHorizon)
    diagnostic.lastAgreeingHorizon = 0;
end
end

function value = quantileNoToolbox(samples, probability)
%QUANTILENOTOOLBOX Column-wise linear-interpolation quantile.
sorted = sort(samples, 1);
nSamples = size(sorted, 1);
if nSamples == 1
    value = sorted;
    return;
end
positions = ((0.5:(nSamples - 0.5)) / nSamples).';
value = zeros(1, size(sorted, 2));
for column = 1:size(sorted, 2)
    value(column) = interp1(positions, sorted(:, column), probability, ...
        'linear', 'extrap');
end
end

% =====================================================================
% Figure 1: receptive field over the day
% =====================================================================
function figureFile = makeReceptiveFieldFigure(outputFolder, main, mae, ...
    naiveMae, dailyPeriod, style)
%MAKERECEPTIVEFIELDFIGURE Where the relevance sits, and where the error is.
%   The white line is the daily offset j = k - 24: the tap holding the same
%   clock hour one day before the hour being predicted.

coordinate = main.coordinate;
meanAbs = meanAbsoluteRelevance(main);
horizon = 1:main.predictionHorizon;

yMap = meanAbs(:, coordinate.yIndex);
pMap = meanAbs(:, [coordinate.pPastIndex, coordinate.pFutureIndex]);
pOffsets = [coordinate.pPastOffsets, coordinate.pFutureOffsets];
colorLimit = [0, max([yMap(:); pMap(:)])];

figureHandle = newFigure('Load NARX receptive field', [80, 60, 1320, 470]);
layout = tiledlayout(figureHandle, 1, 3, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

axesY = nexttile(layout);
drawField(axesY, coordinate.yOffsets, horizon, yMap, colorLimit, style);
hold(axesY, 'on');
plot(axesY, horizon - dailyPeriod, horizon, '--', 'Color', [1, 1, 1, 0.9], ...
    'LineWidth', 1.8);
hold(axesY, 'off');
xlabel(axesY, 'Time offset $j$ from origin $T$ (hours)', 'Interpreter', 'latex');
ylabel(axesY, 'Prediction horizon $k$ (hours)', 'Interpreter', 'latex');
title(axesY, 'Demand taps $y(T+j)$', 'Interpreter', 'latex');
subtitle(axesY, sprintf('dashed: the daily tap j = k - %d', dailyPeriod));

axesP = nexttile(layout);
drawField(axesP, pOffsets, horizon, pMap, colorLimit, style);
hold(axesP, 'on');
xline(axesP, 0.5, '-', 'Color', [1, 1, 1, 0.7], 'LineWidth', 1.2);
hold(axesP, 'off');
xlabel(axesP, 'Time offset $j$ from origin $T$ (hours)', 'Interpreter', 'latex');
title(axesP, 'Temperature taps $p(T+j)$', 'Interpreter', 'latex');
subtitle(axesP, 'left of the line: measured; right: handed to the model');
colorBar = colorbar(axesP);
colorBar.Label.String = sprintf('mean |relevance| (MW)');
colorBar.Label.FontSize = style.smallFont;

axesError = nexttile(layout);
hold(axesError, 'on');
plot(axesError, mae, horizon, '-', 'Color', style.primary, ...
    'LineWidth', 1.8, 'DisplayName', 'NARX');
plot(axesError, naiveMae, horizon, '--', 'Color', style.reference, ...
    'LineWidth', 1.4, 'DisplayName', 'Daily naive');
hold(axesError, 'off');
grid(axesError, 'on');
box(axesError, 'on');
ylim(axesError, [1, main.predictionHorizon]);
xlabel(axesError, 'MAE (MW)');
title(axesError, 'Error at the same horizons');
legend(axesError, 'Location', 'southeast', 'EdgeColor', style.reference);

finishFigure(figureHandle, layout, ...
    'What the 24-hour load forecast is built from', ...
    sprintf(['%d-tap checkpoint, %d held-out origins balanced over the ', ...
    'clock, reference: the same window 24 h earlier'], ...
    main.delay, main.nTrustworthy), style);
figureFile = fullfile(outputFolder, 'load_xai_receptive_field.png');
exportFigure(figureHandle, figureFile);
end

function drawField(axesHandle, offsets, horizon, field, colorLimit, style)
imagesc(axesHandle, offsets, horizon, field);
set(axesHandle, 'YDir', 'normal');
colormap(axesHandle, style.sequentialMap);
clim(axesHandle, colorLimit);
box(axesHandle, 'on');
xlim(axesHandle, [min(offsets) - 0.5, max(offsets) + 0.5]);
ylim(axesHandle, [0.5, max(horizon) + 0.5]);
end

% =====================================================================
% Figure 2: does the network read yesterday at this hour?
% =====================================================================
function figureFile = makeAlignmentFigure(outputFolder, alignment, ...
    checkpoint, dailyPeriod, style)
%MAKEALIGNMENTFIGURE The daily-alignment test, by reference and by checkpoint.

figureHandle = newFigure('Daily alignment of relevance', [80, 60, 1250, 480]);
layout = tiledlayout(figureHandle, 1, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

axesProfile = nexttile(layout);
hold(axesProfile, 'on');
stat = alignment.yesterday;
fill(axesProfile, [stat.displacement, fliplr(stat.displacement)], ...
    [stat.nullLowBand, fliplr(stat.nullHighBand)], [0.85, 0.85, 0.85], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.75, ...
    'DisplayName', 'Permutation null, 95%');
plot(axesProfile, stat.displacement, stat.nullMedianBand, ':', ...
    'Color', style.reference, 'LineWidth', 1.2, 'DisplayName', 'Null median');
plot(axesProfile, stat.displacement, stat.observed, '-o', ...
    'Color', style.primary, 'MarkerFaceColor', style.primary, ...
    'MarkerSize', 5, 'LineWidth', 1.9, ...
    'DisplayName', sprintf('Yesterday reference (p = %.4f)', stat.pValue));
plot(axesProfile, alignment.trainmean.displacement, ...
    alignment.trainmean.observed, '-s', 'Color', style.accent, ...
    'MarkerSize', 5, 'LineWidth', 1.6, ...
    'DisplayName', sprintf('Training-mean reference (p = %.4f)', ...
    alignment.trainmean.pValue));
xline(axesProfile, 0, '-', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1.0, ...
    'HandleVisibility', 'off');
hold(axesProfile, 'off');
grid(axesProfile, 'on');
box(axesProfile, 'on');
xlabel(axesProfile, ...
    'Displacement from the daily tap, $j - (k - 24)$ (hours)', ...
    'Interpreter', 'latex');
ylabel(axesProfile, 'Relevance relative to the horizon mean');
title(axesProfile, 'Relevance peaks one day before the hour predicted');
legend(axesProfile, 'Location', 'northwest', 'EdgeColor', style.reference);

axesByDelay = nexttile(layout);
nCheckpoints = numel(checkpoint);
enrichment = zeros(1, nCheckpoints);
occlusionEnrichment = zeros(1, nCheckpoints);
nullLow = zeros(1, nCheckpoints);
nullHigh = zeros(1, nCheckpoints);
for index = 1:nCheckpoints
    stat = alignment.byCheckpoint{index};
    enrichment(index) = stat.enrichment;
    nullLow(index) = stat.nullLow;
    nullHigh(index) = stat.nullHigh;
    occlusionEnrichment(index) = alignment.byCheckpointOcclusion{index}.enrichment;
end
[delays, order] = sort([checkpoint.delay]);
enrichment = enrichment(order);
occlusionEnrichment = occlusionEnrichment(order);
nullLow = nullLow(order);
nullHigh = nullHigh(order);

hold(axesByDelay, 'on');
halfWidth = 0.02 * (max(delays) - min(delays)) + 1.2;
for index = 1:nCheckpoints
    patch(axesByDelay, delays(index) + halfWidth * [-1, 1, 1, -1], ...
        [nullLow(index), nullLow(index), nullHigh(index), nullHigh(index)], ...
        [0.85, 0.85, 0.85], 'EdgeColor', [0.6, 0.6, 0.6], ...
        'FaceAlpha', 0.7, 'HandleVisibility', 'off');
end
patch(axesByDelay, NaN(1, 4), NaN(1, 4), [0.85, 0.85, 0.85], ...
    'EdgeColor', [0.6, 0.6, 0.6], 'FaceAlpha', 0.7, ...
    'DisplayName', 'Permutation null, 95%');
plot(axesByDelay, delays, enrichment, '-o', 'Color', style.primary, ...
    'MarkerFaceColor', style.primary, 'MarkerSize', 7, 'LineWidth', 1.8, ...
    'DisplayName', 'Integrated Gradients');
plot(axesByDelay, delays, occlusionEnrichment, '--s', ...
    'Color', style.channelPFuture, 'MarkerSize', 7, 'LineWidth', 1.6, ...
    'DisplayName', 'Occlusion, an independent probe');
yline(axesByDelay, 1, ':', 'Color', style.reference, 'LineWidth', 1.0, ...
    'DisplayName', 'No enrichment');
hold(axesByDelay, 'off');
grid(axesByDelay, 'on');
box(axesByDelay, 'on');
xticks(axesByDelay, delays);
xlabel(axesByDelay, 'Tapped-delay length (hours)');
ylabel(axesByDelay, 'Enrichment at the daily tap');
title(axesByDelay, 'Two independent probes, five checkpoints');
subtitle(axesByDelay, sprintf( ...
    'horizons k = %d..%d only, the ones every tapped line reaches', ...
    alignment.commonRange(1), alignment.commonRange(2)));
legend(axesByDelay, 'Location', 'southeast', 'EdgeColor', style.reference);

finishFigure(figureHandle, layout, ...
    'Does the network read the demand one day before the hour it predicts?', ...
    sprintf(['Relevance aligned on j = k - %d and pooled over horizons; ', ...
    'null from %d permutations of the horizon'], dailyPeriod, ...
    alignment.yesterday.nPermutations), style);
figureFile = fullfile(outputFolder, 'load_xai_daily_alignment.png');
exportFigure(figureHandle, figureFile);
end

% =====================================================================
% Figure 3: effective memory
% =====================================================================
function figureFile = makeMemoryFigure(outputFolder, memory, style)
%MAKEMEMORYFIGURE How far back each checkpoint actually reaches.

figureHandle = newFigure('Effective memory of the load checkpoints', ...
    [80, 60, 1250, 480]);
layout = tiledlayout(figureHandle, 1, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

nRuns = numel(memory);
colors = style.referenceRamp(max(nRuns, 2));
axesProfile = nexttile(layout);
axesCumulative = nexttile(layout);
hold(axesProfile, 'on');
hold(axesCumulative, 'on');

for runIndex = 1:nRuns
    profile = memory{runIndex};
    color = colors(runIndex, :);
    plot(axesProfile, profile.lags, profile.normalized, '-o', ...
        'Color', color, 'MarkerFaceColor', color, 'MarkerSize', 4, ...
        'LineWidth', 1.6, 'DisplayName', profile.label);
    plot(axesCumulative, profile.lags, profile.cumulativeShare, '-', ...
        'Color', color, 'LineWidth', 1.8, 'DisplayName', profile.label);
    plot(axesCumulative, profile.lagAtHalf, 0.5, 'o', 'MarkerSize', 7, ...
        'MarkerFaceColor', color, 'MarkerEdgeColor', 'white', ...
        'HandleVisibility', 'off');
end

xline(axesProfile, 24, ':', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1.0, ...
    'HandleVisibility', 'off');
hold(axesProfile, 'off');
grid(axesProfile, 'on');
box(axesProfile, 'on');
xlabel(axesProfile, 'Lag into the past (hours)');
ylabel(axesProfile, 'Relevance, relative to its own peak');
title(axesProfile, 'Relevance by lag');
legend(axesProfile, 'Location', 'northeast', 'EdgeColor', style.reference);

yline(axesCumulative, 0.5, ':', 'Color', style.reference, ...
    'LineWidth', 1.0, 'DisplayName', 'Half the relevance');
hold(axesCumulative, 'off');
grid(axesCumulative, 'on');
box(axesCumulative, 'on');
ylim(axesCumulative, [0, 1]);
xlabel(axesCumulative, 'Lag into the past (hours)');
ylabel(axesCumulative, 'Share of loading relevance within this lag');
title(axesCumulative, 'How much of the memory is actually used');
legend(axesCumulative, 'Location', 'southeast', 'EdgeColor', style.reference);

finishFigure(figureHandle, layout, ...
    'Tapped-delay length, measured rather than searched', ...
    ['Loading channels pooled (demand and temperature taps), ' ...
     'reference: the same window 24 h earlier'], style);
figureFile = fullfile(outputFolder, 'load_xai_memory.png');
exportFigure(figureHandle, figureFile);
end

% =====================================================================
% Figure 4: channels, by horizon and by time of day
% =====================================================================
function figureFile = makeChannelFigure(outputFolder, channel, mae, style)
%MAKECHANNELFIGURE What the forecast leans on, and when.

channelColors = [style.channelY; style.channelPPast; style.channelPFuture];
figureHandle = newFigure('Channel shares of the load forecast', ...
    [80, 60, 1250, 480]);
layout = tiledlayout(figureHandle, 1, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

axesHorizon = nexttile(layout);
areaHandle = area(axesHorizon, 1:size(channel.byHorizon, 1), ...
    channel.byHorizon, 'LineStyle', 'none');
for groupIndex = 1:3
    areaHandle(groupIndex).FaceColor = channelColors(groupIndex, :);
    areaHandle(groupIndex).FaceAlpha = 0.85;
    areaHandle(groupIndex).DisplayName = channel.names{groupIndex};
end
hold(axesHorizon, 'on');
maeAxis = mae / max(mae);
plot(axesHorizon, 1:numel(mae), maeAxis, '-', 'Color', 'white', ...
    'LineWidth', 2.6, 'HandleVisibility', 'off');
plot(axesHorizon, 1:numel(mae), maeAxis, '--', 'Color', 'black', ...
    'LineWidth', 1.4, 'DisplayName', 'MAE, scaled to its maximum');
hold(axesHorizon, 'off');
box(axesHorizon, 'on');
xlim(axesHorizon, [1, size(channel.byHorizon, 1)]);
ylim(axesHorizon, [0, 1]);
xlabel(axesHorizon, 'Prediction horizon $k$ (hours)', 'Interpreter', 'latex');
ylabel(axesHorizon, 'Share of relevance');
title(axesHorizon, 'Handover from measured history to future weather');
legend(axesHorizon, 'Location', 'southoutside', 'NumColumns', 2, ...
    'EdgeColor', style.reference);

axesHour = nexttile(layout);
hold(axesHour, 'on');
for groupIndex = 1:3
    plot(axesHour, channel.hours, channel.byTargetHour(:, groupIndex), ...
        '-o', 'Color', channelColors(groupIndex, :), 'MarkerSize', 4, ...
        'MarkerFaceColor', channelColors(groupIndex, :), 'LineWidth', 1.7, ...
        'DisplayName', channel.names{groupIndex});
end
hold(axesHour, 'off');
grid(axesHour, 'on');
box(axesHour, 'on');
xlim(axesHour, [0, 23]);
xticks(axesHour, 0:3:23);
xlabel(axesHour, 'Clock hour being predicted', 'Interpreter', 'latex');
ylabel(axesHour, 'Share of relevance');
title(axesHour, 'The split depends on which hour is being predicted');
legend(axesHour, 'Location', 'southoutside', 'NumColumns', 2, ...
    'EdgeColor', style.reference);

finishFigure(figureHandle, layout, ...
    'What the forecast leans on: past demand, past weather, future weather', ...
    sprintf(['Over the whole horizon: %.0f%% past demand, %.0f%% past ', ...
    'temperature, %.0f%% future temperature the model is handed'], ...
    100 * channel.overall(1), 100 * channel.overall(2), ...
    100 * channel.overall(3)), style);
figureFile = fullfile(outputFolder, 'load_xai_channels.png');
exportFigure(figureHandle, figureFile);
end

% =====================================================================
% Figure 5: why today is not yesterday
% =====================================================================
function figureFile = makeYesterdayFigure(outputFolder, main, timeStamps, ...
    origins, dailyPeriod, style)
%MAKEYESTERDAYFIGURE One day-ahead forecast, decomposed against yesterday.
%   The reference is the forecast this same frozen network would have made
%   24 hours earlier, so conservation reads directly as "today minus
%   yesterday, in MW", split across the measured quantities that moved it.

% A representative window, not the most extreme one. The largest day-over-
% day difference makes a dramatic picture and a misleading one: it is the
% window where the contributions cancel hardest.
trustworthy = find(main.trustworthy);
deviation = abs(main.outputDelta(end, trustworthy));
[~, ranked] = sort(deviation);
windowIndex = trustworthy(ranked(round(0.5 * numel(ranked))));
horizon = main.predictionHorizon;
originTime = timeStamps(origins(windowIndex));

figureHandle = newFigure('Today against yesterday', [80, 60, 1250, 480]);
layout = tiledlayout(figureHandle, 1, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

axesForecast = nexttile(layout);
horizonHours = (1:horizon).';
hold(axesForecast, 'on');
plot(axesForecast, horizonHours, main.yMeasured(:, windowIndex), '-', ...
    'Color', 'black', 'LineWidth', 1.8, 'DisplayName', 'Measured $y(T+k)$');
plot(axesForecast, horizonHours, main.a2(:, windowIndex), '-o', ...
    'Color', style.primary, 'MarkerSize', 4, ...
    'MarkerFaceColor', style.primary, 'LineWidth', 1.8, ...
    'DisplayName', 'Forecast $a^2(T+k)$');
plot(axesForecast, horizonHours, main.a2Reference(:, windowIndex), '--', ...
    'Color', style.accent, 'LineWidth', 1.6, ...
    'DisplayName', sprintf('Same network, %d h earlier', dailyPeriod));
hold(axesForecast, 'off');
grid(axesForecast, 'on');
box(axesForecast, 'on');
xlim(axesForecast, [1, horizon]);
xlabel(axesForecast, 'Prediction horizon $k$ (hours)', 'Interpreter', 'latex');
ylabel(axesForecast, 'Demand (MW)');
title(axesForecast, sprintf('Origin %s', string(originTime)));
legend(axesForecast, 'Location', 'best', 'Interpreter', 'latex', ...
    'EdgeColor', style.reference);

axesWaterfall = nexttile(layout);
drawWaterfall(axesWaterfall, main, windowIndex, horizon, style);

finishFigure(figureHandle, layout, ...
    'Why the model expects today to differ from yesterday', ...
    sprintf(['%d-tap checkpoint; conservation residual on this window ', ...
    '%.2e MW against a %.1f MW explained difference'], main.delay, ...
    abs(main.completenessResidual(horizon, windowIndex)), ...
    abs(main.outputDelta(horizon, windowIndex))), style);
figureFile = fullfile(outputFolder, 'load_xai_today_vs_yesterday.png');
exportFigure(figureHandle, figureFile);
end

function drawWaterfall(axesHandle, main, windowIndex, horizon, style)
%DRAWWATERFALL From yesterday's forecast to today's, coordinate by
%   coordinate. Completeness is what makes the bars land exactly on the
%   forecast: they are a decomposition, not an illustration.

coordinate = main.coordinate;
relevance = main.relevance(horizon, :, windowIndex);
groups = {coordinate.yIndex, coordinate.pPastIndex, coordinate.pFutureIndex};
groupNames = {'Past demand', 'Past temperature', 'Future temperature'};
groupColors = [style.channelY; style.channelPPast; style.channelPFuture];

% One bar per channel, plus the largest individual taps inside the demand
% channel, so the figure stays readable at 24 taps per channel.
[~, order] = sort(abs(relevance(coordinate.yIndex)), 'descend');
topTaps = coordinate.yIndex(sort(order(1:min(3, numel(order)))));

barValues = [];
barLabels = {};
barColors = [];
for groupIndex = 1:3
    columns = setdiff(groups{groupIndex}, topTaps);
    if groupIndex == 1
        for tapColumn = topTaps
            barValues(end + 1) = relevance(tapColumn); %#ok<AGROW>
            barLabels{end + 1} = coordinate.label{tapColumn}; %#ok<AGROW>
            barColors(end + 1, :) = groupColors(groupIndex, :); %#ok<AGROW>
        end
    end
    barValues(end + 1) = sum(relevance(columns)); %#ok<AGROW>
    barLabels{end + 1} = sprintf('%s, rest', groupNames{groupIndex}); %#ok<AGROW>
    barColors(end + 1, :) = groupColors(groupIndex, :) * 0.55 + 0.45; %#ok<AGROW>
end

start = main.a2Reference(horizon, windowIndex);
running = start + [0, cumsum(barValues)];
nBars = numel(barValues);

hold(axesHandle, 'on');
yline(axesHandle, start, ':', 'Color', style.accent, 'LineWidth', 1.2);
for barIndex = 1:nBars
    left = running(barIndex);
    right = running(barIndex + 1);
    patch(axesHandle, barIndex + [-0.4, 0.4, 0.4, -0.4], ...
        [left, left, right, right], barColors(barIndex, :), ...
        'EdgeColor', [0.25, 0.25, 0.25], 'LineWidth', 0.6);
    if barIndex < nBars
        plot(axesHandle, barIndex + [0.4, 0.6], [right, right], '-', ...
            'Color', [0.45, 0.45, 0.45], 'LineWidth', 0.8);
    end
end
plot(axesHandle, [0.4, nBars + 0.6], ...
    [main.a2(horizon, windowIndex), main.a2(horizon, windowIndex)], '-', ...
    'Color', style.primary, 'LineWidth', 1.4);
text(axesHandle, nBars + 0.55, main.a2(horizon, windowIndex), ...
    sprintf(' a^2(T+%d)', horizon), 'Color', style.primary, ...
    'FontSize', style.smallFont, 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom');
text(axesHandle, 0.45, start, ' yesterday', 'Color', style.accent, ...
    'FontSize', style.smallFont, 'VerticalAlignment', 'top');
hold(axesHandle, 'off');
grid(axesHandle, 'on');
box(axesHandle, 'on');
xlim(axesHandle, [0.3, nBars + 0.7]);
xticks(axesHandle, 1:nBars);
xticklabels(axesHandle, barLabels);
xtickangle(axesHandle, 35);
ylabel(axesHandle, 'Demand (MW)');
title(axesHandle, sprintf('Contributions at k = %d', horizon));
end

% =====================================================================
% Figure 6: validation
% =====================================================================
function figureFile = makeValidationFigure(outputFolder, validation, ...
    diagnostic, style)
%MAKEVALIDATIONFIGURE The checks that make the attribution worth reading.

figureHandle = newFigure('Attribution validation', [80, 60, 1300, 900]);
layout = tiledlayout(figureHandle, 2, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

axesCompleteness = nexttile(layout);
explained = validation.relevanceSum(:);
target = validation.outputDelta(:);
hold(axesCompleteness, 'on');
limits = [min([explained; target]), max([explained; target])];
plot(axesCompleteness, limits, limits, '-', 'Color', style.reference, ...
    'LineWidth', 1.2, 'DisplayName', 'y = x');
scatter(axesCompleteness, target, explained, 14, style.primary, 'filled', ...
    'MarkerFaceAlpha', 0.5, 'DisplayName', 'Windows x horizons');
hold(axesCompleteness, 'off');
grid(axesCompleteness, 'on');
box(axesCompleteness, 'on');
axis(axesCompleteness, 'square');
xlabel(axesCompleteness, 'a^2(T+k) - reference (MW)');
ylabel(axesCompleteness, '\Sigma_i R_i(k) (MW)');
title(axesCompleteness, sprintf('Completeness: %.2e MW worst', ...
    validation.completenessAbsoluteError));
legend(axesCompleteness, 'Location', 'northwest', 'EdgeColor', style.reference);

axesGradient = nexttile(layout);
check = validation.gradientCheck;
hold(axesGradient, 'on');
gradientLimits = [min([check.analytic(:); check.numeric(:)]), ...
    max([check.analytic(:); check.numeric(:)])];
plot(axesGradient, gradientLimits, gradientLimits, '-', ...
    'Color', style.reference, 'LineWidth', 1.2, 'DisplayName', 'y = x');
scatter(axesGradient, check.numeric(:), check.analytic(:), 14, ...
    style.accent, 'filled', 'MarkerFaceAlpha', 0.5, ...
    'DisplayName', 'Jacobian entries');
hold(axesGradient, 'off');
grid(axesGradient, 'on');
box(axesGradient, 'on');
axis(axesGradient, 'square');
xlabel(axesGradient, 'Central difference');
ylabel(axesGradient, 'Analytic forward mode');
title(axesGradient, sprintf('Gradients agree to %.1e', ...
    check.maxAbsoluteError));
legend(axesGradient, 'Location', 'northwest', 'EdgeColor', style.reference);

axesOcclusion = nexttile(layout);
% occlusion.signed is already reference-minus-occluded in MW, the same
% sign convention and unit as the relevance, so the two are directly
% comparable without any rescaling.
occlusionSigned = validation.occlusion.signed(:, :, validation.trustworthy);
relevanceSigned = validation.relevance(:, :, validation.trustworthy);
occlusionEffect = occlusionSigned(:);
relevanceAtEnd = relevanceSigned(:);
hold(axesOcclusion, 'on');
horizonIndex = repmat((1:size(occlusionSigned, 1)).', ...
    size(occlusionSigned, 2) * size(occlusionSigned, 3), 1);
scatter(axesOcclusion, occlusionEffect, relevanceAtEnd, 9, horizonIndex, ...
    'filled', 'MarkerFaceAlpha', 0.35, ...
    'DisplayName', 'Coordinates x horizons x windows');
colormap(axesOcclusion, style.sequentialMap);
horizonBar = colorbar(axesOcclusion);
horizonBar.Label.String = 'prediction horizon k';
% A handful of windows carry contributions in the thousands of MW. Letting
% them set the frame hides the bulk, so the axes are clipped to the 99th
% percentile and the clipping is stated.
occlusionSpan = quantileNoToolbox( ...
    abs([occlusionEffect; relevanceAtEnd]), 0.99);
occlusionLimits = [-occlusionSpan, occlusionSpan];
plot(axesOcclusion, occlusionLimits, occlusionLimits, '-', ...
    'Color', style.reference, 'LineWidth', 1.2, 'DisplayName', 'y = x');
hold(axesOcclusion, 'off');
grid(axesOcclusion, 'on');
box(axesOcclusion, 'on');
axis(axesOcclusion, 'square');
xlim(axesOcclusion, occlusionLimits);
ylim(axesOcclusion, occlusionLimits);
xlabel(axesOcclusion, 'Occlusion effect (MW)');
ylabel(axesOcclusion, 'Integrated Gradients (MW)');
title(axesOcclusion, sprintf('Occlusion cross-check, r = %.3f overall', ...
    correlationNoToolbox(occlusionEffect, relevanceAtEnd)));
subtitle(axesOcclusion, 'axes clipped to the 99th percentile of both');

axesDiagnostic = nexttile(layout);
yyaxis(axesDiagnostic, 'left');
plot(axesDiagnostic, diagnostic.horizon, diagnostic.agreement, '-o', ...
    'MarkerSize', 4, 'LineWidth', 1.8);
ylabel(axesDiagnostic, 'r, Integrated Gradients vs occlusion');
ylim(axesDiagnostic, [0, 1]);
yyaxis(axesDiagnostic, 'right');
plot(axesDiagnostic, diagnostic.horizon, diagnostic.cancellation, '-s', ...
    'MarkerSize', 4, 'LineWidth', 1.6);
set(axesDiagnostic, 'YScale', 'log');
ylabel(axesDiagnostic, 'median \Sigma|R_i| / |\Sigma R_i|');
yyaxis(axesDiagnostic, 'left');
grid(axesDiagnostic, 'on');
box(axesDiagnostic, 'on');
xlim(axesDiagnostic, [1, max(diagnostic.horizon)]);
xlabel(axesDiagnostic, 'Prediction horizon k (hours)');
title(axesDiagnostic, 'How far the per-coordinate reading can be pushed');

finishFigure(figureHandle, layout, ...
    'The attribution decomposes the shipped network, not an approximation of it', ...
    sprintf(['%d windows, M = %d; the explicit unroll reproduces ', ...
    'sim(net, ...) to %.1e MW. Completeness holds at every horizon; the ', ...
    'per-coordinate split does not survive to k = %d.'], ...
    validation.nWindows, validation.integrationSteps, ...
    validation.simParityError, max(diagnostic.horizon)), style);
figureFile = fullfile(outputFolder, 'load_xai_validation.png');
exportFigure(figureHandle, figureFile);
end

function value = correlationNoToolbox(firstVector, secondVector)
firstVector = firstVector - mean(firstVector);
secondVector = secondVector - mean(secondVector);
denominator = sqrt(sum(firstVector .^ 2) * sum(secondVector .^ 2));
if denominator == 0
    value = NaN;
else
    value = sum(firstVector .* secondVector) / denominator;
end
end

% =====================================================================
% Figure style, matching Explainability/run_narx_explainability.m
% =====================================================================
function style = figureStyle()
style.primary = [0.0000, 0.4470, 0.7410];
style.accent = [0.8500, 0.3250, 0.0980];
style.reference = [0.35, 0.35, 0.35];
style.channelY = [0.0000, 0.5255, 0.5451];
style.channelPPast = [0.8500, 0.3250, 0.0980];
style.channelPFuture = [0.4940, 0.1840, 0.5560];
style.baseFont = 11;
style.smallFont = 9;
style.sequentialMap = sequentialColormap();
style.referenceRamp = @referenceRamp;
end

function map = sequentialColormap()
%SEQUENTIALCOLORMAP White to deep blue, monotone in lightness so the figure
%   survives grayscale printing.
anchors = [ ...
    1.0000, 1.0000, 1.0000; ...
    0.8706, 0.9216, 0.9686; ...
    0.6196, 0.7922, 0.8824; ...
    0.4196, 0.6824, 0.8392; ...
    0.2588, 0.5725, 0.7765; ...
    0.1294, 0.4431, 0.7098; ...
    0.0314, 0.3176, 0.6118; ...
    0.0314, 0.1882, 0.4196];
positions = linspace(0, 1, size(anchors, 1));
query = linspace(0, 1, 256);
map = [interp1(positions, anchors(:, 1), query).', ...
       interp1(positions, anchors(:, 2), query).', ...
       interp1(positions, anchors(:, 3), query).'];
end

function colors = referenceRamp(nColors)
anchors = [ ...
    0.2706, 0.4588, 0.7059; ...
    0.4549, 0.6784, 0.8196; ...
    0.9569, 0.6471, 0.5098; ...
    0.8392, 0.3765, 0.3020; ...
    0.6980, 0.0941, 0.1686];
if nColors == 1
    colors = anchors(1, :);
    return;
end
positions = linspace(0, 1, size(anchors, 1));
query = linspace(0, 1, nColors);
colors = [interp1(positions, anchors(:, 1), query).', ...
          interp1(positions, anchors(:, 2), query).', ...
          interp1(positions, anchors(:, 3), query).'];
end

function figureHandle = newFigure(name, position)
figureHandle = figure('Name', name, 'Color', 'white', 'Visible', 'off', ...
    'Position', position);
if exist('theme', 'file')
    theme(figureHandle, 'light');
end
end

function finishFigure(figureHandle, layout, titleText, subtitleText, style)
titleHandle = title(layout, titleText, 'FontWeight', 'bold', ...
    'FontSize', style.baseFont + 2, 'Color', 'black');
subtitleHandle = subtitle(layout, subtitleText, ...
    'FontSize', style.smallFont, 'Color', [0.25, 0.25, 0.25]);
set(findall(figureHandle, 'Type', 'axes'), 'Color', 'white', ...
    'XColor', 'black', 'YColor', 'black', ...
    'GridColor', [0.78, 0.78, 0.78], 'FontSize', style.smallFont);
titleHandle.Color = 'black';
subtitleHandle.Color = [0.25, 0.25, 0.25];
end

function exportFigure(figureHandle, figureFile)
exportgraphics(figureHandle, figureFile, 'Resolution', 300);
close(figureHandle);
end
