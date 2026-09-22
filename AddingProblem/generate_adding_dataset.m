function dataset = generate_adding_dataset(options)
%GENERATE_ADDING_DATASET The adding problem, with Hochreiter marker placement.
%
%   DATASET = GENERATE_ADDING_DATASET('SequenceLength', T) builds independent
%   training and test sets for the adding problem, the first of the four
%   long-range-dependency benchmarks in Le2015.
%
%   The external input p(t) has R^1 = 2 components at every time t:
%
%     p_1(t)   a value drawn uniformly from [0,1]
%     p_2(t)   a marker, 0 everywhere except at two times, where it is 1
%
%   The measured target is supplied only at the final time T,
%
%     y(T) = p_1(t_a) + p_1(t_b),    p_2(t_a) = p_2(t_b) = 1,
%
%   and is undefined at every earlier time. Both marked values are uniform on
%   [0,1], so y(T) has mean 1 and variance 1/6; always predicting 1 therefore
%   gives a mean squared error of 1/6 = 0.1667. Le2015 states this baseline as
%   0.1767. DATASET.BASELINEMSE reports the value measured on the test set
%   actually generated here, which is the reference these runs are scored
%   against.
%
%   MARKER PLACEMENT follows Hochreiter and Schmidhuber (1997), section 5.4,
%   which is where the task was first defined and is reproduced in this folder
%   as hochreiter1997_section_5_4.md. Le2015 states only that the mask is 1 at
%   "two steps" and never says where, so the rule is taken from the original:
%
%     one mark is drawn uniformly from the first 10 pairs; the other is drawn
%     uniformly from the first T/2 - 1 pairs that are still unmarked.
%
%   Both marks therefore fall in the first half of the sequence, so **both**
%   marked values must be carried at least T/2 steps to the readout at T. That
%   is the "minimal time lag T/2" the original's Table 7 reports, and it is the
%   property that makes this a long-term dependency benchmark rather than a
%   regression benchmark: no model can score well by remembering only one of
%   the two numbers.
%
%   The first window is min(10, floor(T/2)) rather than a literal 10, so that
%   the minimal lag of T/2 is preserved at the short sequence lengths used
%   here; for every T >= 20 it is the original's literal 10.
%
%   The value distribution, the marker alphabet {0,1}, the unscaled sum target
%   and the always-predict-1 baseline are Le2015's; only the placement comes
%   from the original. See README.md for the full comparison of the two.
%
%   NAME-VALUE OPTIONS
%     SequenceLength   T, the number of time steps. Default 30.
%     NumTrain         Training sequences. Default 500.
%     NumTest          Test sequences. Default 500.
%     Seed             Seed of the private RandStream. Default 1.
%
%   DATASET has fields TRAIN and TEST, each a struct with
%
%     pValues    (T x Q) the p_1(t) channel
%     pMarkers   (T x Q) the p_2(t) channel
%     yFinal     (1 x Q) the target y(T)
%     markIndex  (2 x Q) the two marked times in ascending order, for auditing
%
%   Training and test sequences are drawn from the same distribution but from
%   disjoint draws of the stream, so no sequence is shared.
%
%   See also ADDING_INPUT_CELLS, MAKE_ADDING_NARX, RUN_ADDING_NARX.

arguments
    options.SequenceLength (1,1) double {mustBeInteger, mustBePositive} = 30
    options.NumTrain (1,1) double {mustBeInteger, mustBePositive} = 500
    options.NumTest (1,1) double {mustBeInteger, mustBePositive} = 500
    options.Seed (1,1) double = 1
end

finalTime = options.SequenceLength;
assert(finalTime >= 8, ...
    ['The adding problem needs at least 8 time steps for both marks to fit ', ...
     'in the first half, not %d.'], finalTime);

stream = RandStream('mt19937ar', 'Seed', options.Seed);

dataset.sequenceLength = finalTime;
dataset.seed = options.Seed;
dataset.train = draw_split(stream, finalTime, options.NumTrain);
dataset.test = draw_split(stream, finalTime, options.NumTest);

% The reference a model must beat: always predicting the mean of y(T).
dataset.baselineMSE = mean((dataset.test.yFinal - 1).^2);
dataset.baselineMSEBestConstant = ...
    mean((dataset.test.yFinal - mean(dataset.test.yFinal)).^2);
end


function split = draw_split(stream, finalTime, numSequences)
%DRAW_SPLIT One independent set of adding-problem sequences.

split.pValues = rand(stream, finalTime, numSequences);
split.pMarkers = zeros(finalTime, numSequences);

% Hochreiter and Schmidhuber (1997), section 5.4: one mark among the first ten
% pairs, the other among the first T/2 - 1 pairs still unmarked. Both land in
% the first half, so both marked values face a lag of at least T/2.
earlyWindow = min(10, floor(finalTime / 2));
halfWindow = floor(finalTime / 2) - 1;
assert(halfWindow >= 2, ...
    'T = %d leaves no room to place two distinct marks.', finalTime);

earlyMark = randi(stream, [1, earlyWindow], 1, numSequences);

% The second mark is uniform over the first T/2 - 1 pairs excluding the one
% already taken. Draw over the remaining count and step past the taken slot,
% which samples without replacement without a loop.
isTaken = earlyMark <= halfWindow;
available = halfWindow - double(isTaken);
draw = ceil(rand(stream, 1, numSequences) .* available);
halfMark = draw + double(isTaken & draw >= earlyMark);

% Row 1 is the earlier of the two marks, so lag statistics read naturally.
rows = sort([earlyMark; halfMark], 1);
cols = repmat(1:numSequences, 2, 1);
split.pMarkers(sub2ind(size(split.pMarkers), rows, cols)) = 1;
split.markIndex = rows;

split.yFinal = split.pValues(sub2ind(size(split.pValues), rows(1, :), ...
    1:numSequences)) + ...
    split.pValues(sub2ind(size(split.pValues), rows(2, :), 1:numSequences));

assert(all(sum(split.pMarkers, 1) == 2), ...
    'Every sequence must carry exactly two markers.');
assert(all(rows(1, :) < rows(2, :)), 'The two marks must be distinct.');
assert(all(rows(:) <= max(earlyWindow, halfWindow)), ...
    'Both marks must fall in the first half of the sequence.');
end
