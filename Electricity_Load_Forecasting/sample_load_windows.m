function [pWindows, yWindows, info] = sample_load_windows(options)
%SAMPLE_LOAD_WINDOWS Draw multi-step windows from the hourly load series.
%
%   [PWINDOWS, YWINDOWS, INFO] = SAMPLE_LOAD_WINDOWS('Delay', D) returns
%   matrices of size (D + PREDICTIONHORIZON) x NWINDOWS holding the
%   external input p(t) and the measured target y(t) in their recorded
%   units. Column j is one independently initialised window: its first D
%   samples load the tapped delay lines, and the remaining
%   PREDICTIONHORIZON samples are the measured targets used to score
%   a^2(T+k) = yHat(T+k).
%
%   This is the load-series counterpart of NARXSAMPLEWINDOWS, and the
%   matrices it returns are accepted directly by NARXFORECAST and
%   NARXEXPLAIN.
%
%   NAME-VALUE OPTIONS
%     Delay              Tapped-delay length of the checkpoint. Required.
%     PredictionHorizon  H, in hours. Default 24, one day-ahead curve.
%     NumWindows         Number of windows to draw. Default 2000.
%     RandomSeed         Seed for the deterministic draw. Default 2024.
%     Split              Any split accepted by IMPORT_DATA_LOAD.
%                        Default 'dry-test', the held-out season.
%     Origins            Prediction-origin indices into the split, as
%                        returned in INFO.ORIGININDEX. Supplying them makes
%                        two checkpoints of different DELAY forecast from
%                        exactly the same origins, so their error curves
%                        are comparable. Default [], meaning draw them.
%
%   INFO reports the draw and, for each window, the prediction origin T:
%   its index into the split (ORIGININDEX) and its timestamp
%   (ORIGINDATETIME), so that results can afterwards be grouped by hour of
%   day, weekday or month without re-reading the file.
%
%   See also IMPORT_DATA_LOAD, NARXFORECAST, NARXSAMPLEWINDOWS.

arguments
    options.Delay (1,1) double {mustBePositive}
    options.PredictionHorizon (1,1) double {mustBePositive} = 24
    options.NumWindows (1,1) double {mustBePositive} = 2000
    options.RandomSeed (1,1) double = 2024
    options.Split = 'dry-test'
    options.Origins (1,:) double = []
end

delay = options.Delay;
predictionHorizon = options.PredictionHorizon;
sequenceLength = delay + predictionHorizon;

[p, y, t] = import_data_load(options.Split);
nSamples = numel(y);

% The prediction origin T is the last sample carrying measured data. A
% window covers indices (T - delay + 1) : (T + predictionHorizon), so the
% origins that fit the series are these:
firstOrigin = delay;
lastOrigin = nSamples - predictionHorizon;
assert(lastOrigin >= firstOrigin, ...
    'The split holds %d hours, too few for a %d-hour window.', ...
    nSamples, sequenceLength);

if isempty(options.Origins)
    candidateOrigins = firstOrigin:lastOrigin;
    nWindows = min(options.NumWindows, numel(candidateOrigins));
    rng(options.RandomSeed, 'twister');
    origins = candidateOrigins(randperm(numel(candidateOrigins), nWindows));
else
    origins = options.Origins;
    nWindows = numel(origins);
    assert(all(origins >= firstOrigin & origins <= lastOrigin), ...
        ['Supplied origins do not all fit a %d-delay, %d-hour window. ', ...
         'Draw them with the largest delay of the sweep.'], ...
        delay, predictionHorizon);
end

sampleIndices = (-delay + 1:predictionHorizon).' + origins;
pWindows = p(sampleIndices);
yWindows = y(sampleIndices);

info.split = options.Split;
info.delay = delay;
info.predictionHorizon = predictionHorizon;
info.sequenceLength = sequenceLength;
info.nWindows = nWindows;
info.randomSeed = options.RandomSeed;
info.totalCandidateOrigins = lastOrigin - firstOrigin + 1;
info.originIndex = origins;
info.originDatetime = t(origins);
info.inputName = 'Tocumen air temperature p(t)';
info.targetName = 'national demand y(t)';
end
