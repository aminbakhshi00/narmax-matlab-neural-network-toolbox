function results = plot_maglev_narx_mae_vs_horizon()
%PLOT_MAGLEV_NARX_MAE_VS_HORIZON Plot recursive maglev error by horizon.
%   RESULTS = PLOT_MAGLEV_NARX_MAE_VS_HORIZON loads the saved closed-loop
%   NARX checkpoint, draws 10,000 deterministic subsequences from held-out
%   maglev files 16 through 20, and computes the mean absolute error (MAE)
%   of a^2(t+k) = yHat(t+k) for prediction horizons k = 1,...,100.
%
%   Each recursive prediction is initialized with the checkpoint's eight
%   required samples of measured target y(t) and external input p(t). Future
%   measured targets are used only to score the predictions, not as feedback.
%
%   The function creates two new files beside this MATLAB file:
%     maglev_narx_2delay_mae_vs_horizon.png
%     maglev_narx_2delay_mae_vs_horizon_results.mat

scriptFolder = fileparts(mfilename('fullpath'));
modelFile = fullfile(scriptFolder, 'train', 'maglev', ...
    'narx_2delay_10neurons_100steps', 'narx_model.mat');
outputFigure = fullfile(scriptFolder, ...
    'maglev_narx_2delay_mae_vs_horizon.png');
outputData = fullfile(scriptFolder, ...
    'maglev_narx_2delay_mae_vs_horizon_results.mat');

assert(isfile(modelFile), 'Saved NARX checkpoint not found: %s', modelFile);
modelData = load(modelFile, 'narx');
assert(isfield(modelData, 'narx') && isa(modelData.narx, 'NARXmodel'), ...
    'The checkpoint must contain a NARXmodel object named narx.');

narxModel = modelData.narx;
net = narxModel.narx;
predictionHorizon = 100;
nSubsequences = 10000;
evaluationFileIds = 16:20;
randomSeed = 2024;
delay = narxModel.delay;
sequenceLength = delay + predictionHorizon;

assert(delay == 2, ...
    'Expected the saved 2-delay checkpoint, but found delay = %d.', delay);
assert(net.numFeedbackDelays == 0 && net.numLayerDelays == delay, ...
    'Expected the saved network to be in closed-loop (parallel) form.');

% Load complete held-out trajectories separately so no window crosses from
% one experiment into another.
pByFile = cell(size(evaluationFileIds));
yByFile = cell(size(evaluationFileIds));
nStartsByFile = zeros(size(evaluationFileIds));

for fileIndex = 1:numel(evaluationFileIds)
    fileId = evaluationFileIds(fileIndex);
    dataFile = fullfile(scriptFolder, 'data', 'maglev', ...
        sprintf('Data_Train_Ex%d.mat', fileId));
    assert(isfile(dataFile), 'Maglev data file not found: %s', dataFile);

    data = load(dataFile, 'Voltage', 'Mag_Pos_F');
    assert(isfield(data, 'Voltage') && isfield(data, 'Mag_Pos_F'), ...
        'Expected Voltage and Mag_Pos_F in %s.', dataFile);

    pByFile{fileIndex} = double(data.Voltage(:));
    yByFile{fileIndex} = double(data.Mag_Pos_F(:));
    assert(numel(pByFile{fileIndex}) == numel(yByFile{fileIndex}), ...
        'Input and target lengths differ in %s.', dataFile);

    nStartsByFile(fileIndex) = ...
        numel(pByFile{fileIndex}) - sequenceLength + 1;
    assert(nStartsByFile(fileIndex) > 0, ...
        'Trajectory in %s is too short for this evaluation.', dataFile);
end

% Select windows without replacement, deterministically.
totalCandidateWindows = sum(nStartsByFile);
assert(totalCandidateWindows >= nSubsequences, ...
    'Only %d valid windows are available; %d were requested.', ...
    totalCandidateWindows, nSubsequences);
rng(randomSeed, 'twister');
selectedGlobalStarts = randperm(totalCandidateWindows, nSubsequences);

pSequences = zeros(sequenceLength, nSubsequences);
ySequences = zeros(sequenceLength, nSubsequences);
fileEdges = cumsum([0, nStartsByFile]);
timeOffsets = (0:sequenceLength-1).';

for fileIndex = 1:numel(evaluationFileIds)
    selectedMask = selectedGlobalStarts > fileEdges(fileIndex) & ...
        selectedGlobalStarts <= fileEdges(fileIndex + 1);
    destinationColumns = find(selectedMask);
    localStarts = selectedGlobalStarts(selectedMask) - fileEdges(fileIndex);
    sampleIndices = timeOffsets + localStarts;

    pSequences(:, destinationColumns) = pByFile{fileIndex}(sampleIndices);
    ySequences(:, destinationColumns) = yByFile{fileIndex}(sampleIndices);
end

% Apply only the normalization stored with the trained checkpoint.
pNormalized = (pSequences - narxModel.int_u) / narxModel.slope_u;
yNormalized = (ySequences - narxModel.int_y) / narxModel.slope_y;

% At each time, each cell contains all independently initialized sequences.
pCells = mat2cell(pNormalized, ones(sequenceLength, 1), nSubsequences).';
yCells = mat2cell(yNormalized, ones(sequenceLength, 1), nSubsequences).';
[preparedP, initialInputState, initialLayerState, preparedY] = ...
    preparets(net, pCells, {}, yCells);
a2Cells = sim(net, preparedP, initialInputState, initialLayerState);

assert(numel(a2Cells) == predictionHorizon, ...
    'Expected %d prediction horizons, but received %d.', ...
    predictionHorizon, numel(a2Cells));

a2Normalized = vertcat(a2Cells{:});
preparedYNormalized = vertcat(preparedY{:});
a2 = narxModel.slope_y * a2Normalized + narxModel.int_y;
y = narxModel.slope_y * preparedYNormalized + narxModel.int_y;

targetAlignmentError = max(abs( ...
    y - ySequences(delay + 1:end, :)), [], 'all');
assert(targetAlignmentError < 1e-10, ...
    'Prepared targets are not aligned with horizons 1 through %d.', ...
    predictionHorizon);

horizon = (1:predictionHorizon).';
absoluteError = abs(y - a2);
mae = mean(absoluteError, 2);
[peakMae, peakHorizon] = max(mae);
maeAtFinalHorizon = mae(end);
shortHorizonLimit = 20;
[earlyPeakMae, earlyPeakHorizon] = max(mae(1:shortHorizonLimit));
hasEarlyPeakAboveFinalHorizon = earlyPeakMae > maeAtFinalHorizon;
isMonotonicNondecreasing = all(diff(mae) >= 0);

figureHandle = figure('Color', 'white', 'Visible', 'off', ...
    'Name', 'Maglev NARX MAE by prediction horizon', ...
    'Position', [100, 100, 1000, 650]);
axesHandle = axes(figureHandle, 'Color', 'white', ...
    'XColor', 'black', 'YColor', 'black', ...
    'GridColor', [0.75, 0.75, 0.75]);
hold(axesHandle, 'on');
plot(axesHandle, horizon, mae, 'Color', [0.0000, 0.4470, 0.7410], ...
    'LineWidth', 1.8, 'DisplayName', 'Mean absolute error');
plot(peakHorizon, peakMae, 'o', 'MarkerSize', 7, ...
    'MarkerFaceColor', [0.8500, 0.3250, 0.0980], ...
    'MarkerEdgeColor', 'white', 'LineWidth', 1.0, ...
    'DisplayName', sprintf('Peak: k = %d', peakHorizon));
yline(maeAtFinalHorizon, '--', 'Color', [0.35, 0.35, 0.35], ...
    'LineWidth', 1.0, ...
    'DisplayName', sprintf('MAE at k = %d', predictionHorizon));
hold(axesHandle, 'off');
grid on;
box on;
xlim([1, predictionHorizon]);
xlabel('Prediction horizon, k (steps)', 'Color', 'black');
ylabel('$\mathrm{MAE}\left(a^2(t+k)=\hat{y}(t+k)\right)$', ...
    'Interpreter', 'latex', 'Color', 'black');
title('Closed-loop NARX prediction error on held-out maglev runs', ...
    'Color', 'black');
subtitleHandle = subtitle(sprintf( ...
    '%d-delay, %d-neuron checkpoint; %s subsequences from files %d--%d', ...
    delay, narxModel.neurons, formatNumber(nSubsequences), ...
    evaluationFileIds(1), evaluationFileIds(end)));
subtitleHandle.Color = [0.20, 0.20, 0.20];
legendHandle = legend('Location', 'best');
legendHandle.Color = 'white';
legendHandle.TextColor = 'black';
legendHandle.EdgeColor = [0.35, 0.35, 0.35];
exportgraphics(figureHandle, outputFigure, 'Resolution', 300);
close(figureHandle);

results = struct;
results.modelFile = modelFile;
results.evaluationFileIds = evaluationFileIds;
results.randomSeed = randomSeed;
results.nSubsequences = nSubsequences;
results.delay = delay;
results.neurons = narxModel.neurons;
results.horizon = horizon;
results.mae = mae;
results.peakHorizon = peakHorizon;
results.peakMae = peakMae;
results.maeAtFinalHorizon = maeAtFinalHorizon;
results.shortHorizonLimit = shortHorizonLimit;
results.earlyPeakHorizon = earlyPeakHorizon;
results.earlyPeakMae = earlyPeakMae;
results.hasEarlyPeakAboveFinalHorizon = ...
    hasEarlyPeakAboveFinalHorizon;
results.isMonotonicNondecreasing = isMonotonicNondecreasing;
results.targetAlignmentError = targetAlignmentError;
results.outputFigure = outputFigure;
save(outputData, 'results');

fprintf('Saved plot: %s\n', outputFigure);
fprintf('Saved values: %s\n', outputData);
fprintf('Peak MAE: %.9g at horizon %d\n', peakMae, peakHorizon);
fprintf('MAE at horizon %d: %.9g\n', ...
    predictionHorizon, maeAtFinalHorizon);
fprintf('Largest MAE in horizons 1-%d: %.9g at horizon %d\n', ...
    shortHorizonLimit, earlyPeakMae, earlyPeakHorizon);
fprintf('Early peak above final-horizon MAE: %s\n', ...
    string(hasEarlyPeakAboveFinalHorizon));
fprintf('MAE monotonically nondecreasing: %s\n', ...
    string(isMonotonicNondecreasing));
end

function text = formatNumber(value)
%FORMATNUMBER Add thousands separators without locale dependence.
text = regexprep(sprintf('%d', value), ...
    '(?<!^)(?=(\d{3})+$)', ',');
end
