function dataset = make_varlen_dataset(minLength, maxLength, numTrain, numTest, run)
%MAKE_VARLEN_DATASET Adding-problem sequences whose length varies from sequence to sequence.
%
%   DATASET = MAKE_VARLEN_DATASET(MINLENGTH, MAXLENGTH, NUMTRAIN, NUMTEST, RUN)
%   draws each sequence's own final time L uniformly from the integers
%   MINLENGTH..MAXLENGTH and builds every length group with
%   GENERATE_ADDING_DATASET, so both marks fall in the first half of that
%   sequence's own L.
%
%   All sequences are zero-padded to MAXLENGTH steps, because MATLAB runs the
%   Q sequences as columns of one matrix. DATASET.TRAIN and DATASET.TEST have
%   the fields of GENERATE_ADDING_DATASET plus LENGTHS (1 x Q), each
%   sequence's L, the only time at which its target y(L) exists.
%
%   Seeds: the length draw uses 8000000 + 1000*MINLENGTH + RUN, and length
%   group L uses 9000000 + 1000*L + RUN.

stream = RandStream('mt19937ar', 'Seed', 8000000 + 1000 * minLength + run);
trainLengths = randi(stream, [minLength maxLength], 1, numTrain);
testLengths = randi(stream, [minLength maxLength], 1, numTest);

dataset.train = empty_split(maxLength, trainLengths);
dataset.test = empty_split(maxLength, testLengths);
for L = minLength:maxLength
    inTrain = find(trainLengths == L);
    inTest = find(testLengths == L);
    if isempty(inTrain) && isempty(inTest), continue; end
    group = generate_adding_dataset('SequenceLength', L, ...
        'NumTrain', max(numel(inTrain), 1), 'NumTest', max(numel(inTest), 1), ...
        'Seed', 9000000 + 1000 * L + run);
    dataset.train = place(dataset.train, group.train, inTrain, L);
    dataset.test = place(dataset.test, group.test, inTest, L);
end
dataset.baselineMSE = mean((dataset.test.yFinal - 1).^2);
end


function split = empty_split(maxLength, lengths)
split.pValues = zeros(maxLength, numel(lengths));
split.pMarkers = zeros(maxLength, numel(lengths));
split.yFinal = zeros(1, numel(lengths));
split.markIndex = zeros(2, numel(lengths));
split.lengths = lengths;
end


function split = place(split, group, columns, L)
k = numel(columns);
split.pValues(1:L, columns) = group.pValues(:, 1:k);
split.pMarkers(1:L, columns) = group.pMarkers(:, 1:k);
split.yFinal(columns) = group.yFinal(1:k);
split.markIndex(:, columns) = group.markIndex(:, 1:k);
end
