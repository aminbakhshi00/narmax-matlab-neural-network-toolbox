function [dataSeed, weightSeed] = adding_seeds(finalTime, run)
%ADDING_SEEDS The dataset seed and the weight seed for one run.
%
%   [DATASEED, WEIGHTSEED] = ADDING_SEEDS(FINALTIME, RUN) maps a run index
%   1..15 onto the two independent sources of randomness in this study.
%
%   The 15 runs vary BOTH of them: five draws of the initial weights, each
%   trained against three independent datasets.
%
%       run    1  2  3  4  5   6  7  8  9 10  11 12 13 14 15
%       weight 1  2  3  4  5   1  2  3  4  5   1  2  3  4  5
%       data   A  A  A  A  A   B  B  B  B  B   C  C  C  C  C
%
%   Varying only the weights would tie the dataset to the sequence length,
%   and then "this length is hard" and "these particular sequences are hard"
%   could not be told apart: a lucky draw at one length would look like a
%   property of that length, in every configuration tested, forever.
%
%   Reliability is therefore reported over 15 (weight, dataset) pairs, and
%   both activation functions are given exactly the same 15 pairs, so the
%   comparison between them is paired run by run.
%
%   See also RUN_ADDING, GENERATE_ADDING_DATASET, MAKE_ADDING_NARX.

arguments
    finalTime (1,1) double {mustBeInteger, mustBePositive}
    run (1,1) double {mustBeInteger, mustBePositive}
end

weightDraw = mod(run - 1, 5) + 1;      % 1..5, cycling
dataDraw = floor((run - 1) / 5);       % 0, 1, 2

weightSeed = 100 + weightDraw;
dataSeed = 1000 * finalTime + weightDraw + 500000 * dataDraw;
end
