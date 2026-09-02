% Clear Data
clc;
clear variables;
clear;
warning off MATLAB:subscripting:noSubscriptsSpecified

% add the NARX model and this folder to the path
addpath(fileparts(fileparts(mfilename('fullpath'))));
addpath(fileparts(mfilename('fullpath')));

% initiate network architecture and prediction horizon
delay = 24;             % 24 taps of one hour each: one full day of memory
neurons = 10;
train_k = 24;           % 24-hour ahead load curve, the operational horizon
train_split = 'dry-train';

% import training data
% p_pre is the external input p(t), Tocumen air temperature in degrees C
% y_pre is the measured target y(t), national hourly demand in MW
[p_pre, y_pre] = import_data_load(train_split);

% train narx
fprintf('\nTRAINING NARX MODEL:\n');

% same initialisation as run 1 of run_load_delay_experiment, whose restart r
% is seeded with 2024 + r, so this script reproduces that checkpoint
rng(2024 + 1, 'twister');

narx = NARXmodel(delay, neurons);
narx.trainAlg = 'trainlm';
narx.earlyStoppage = true;
narx.iterPerRun = 300;    % the budget the delay sweep gives every delay
narx.iterAfterValley = 100;
narx.iterAfterSeq = 50;
narx.maxStep = 10;
narx.initialTraining = 10;
narx.zero_input_delay = false;

narx = narx.train(p_pre, y_pre, train_k);
dir = fullfile(fileparts(mfilename('fullpath')), 'train', 'load', ...
    sprintf('narx_%sdelay_%sneurons_%ssteps', num2str(delay), num2str(neurons), num2str(train_k)));
mkdir(dir);
save(sprintf('%s/narx_model', dir), 'narx');
