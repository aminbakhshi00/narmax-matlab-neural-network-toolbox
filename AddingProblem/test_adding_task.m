function test_adding_task
%TEST_ADDING_TASK Executable checks on the data, the wiring and the loss.
%
%   Run it from this folder. Every check prints "ok" or throws.
%
%   See also GENERATE_ADDING_DATASET, MAKE_ADDING_NARX, RUN_ADDING.

fprintf('\nADDING PROBLEM -- SETUP CHECKS\n');

% ---------------------------------------------------------------- the data --
ds = generate_adding_dataset('SequenceLength', 60, 'NumTrain', 400, ...
    'NumTest', 400, 'Seed', 7);
s = ds.train;
marked = arrayfun(@(q) sum(s.pValues(s.pMarkers(:,q) == 1, q)), 1:size(s.pValues,2));
assert(max(abs(marked - s.yFinal)) < 1e-12);
ok('y(T) is the sum of the two marked p_1 values');

assert(all(s.markIndex(:) <= 30) && all(s.markIndex(1,:) < s.markIndex(2,:)));
assert(all(sum(s.pMarkers, 1) == 2));
ok('both marks are distinct and lie in the first half, so the lag is >= T/2');

again = generate_adding_dataset('SequenceLength', 60, 'NumTrain', 400, ...
    'NumTest', 400, 'Seed', 7);
assert(isequal(again.train.pValues, ds.train.pValues));
assert(~isequal(ds.train.pValues, ds.test.pValues));
ok('the draw is repeatable and the two splits differ');

fprintf('  ..  baseline MSE of always predicting the mean: %.4f (theory 1/6 = %.4f)\n', ...
    ds.baselineMSE, 1/6);

% ------------------------------------------------------------ the seed map --
w = zeros(1,15); d = zeros(1,15);
for run = 1:15
    [d(run), w(run)] = adding_seeds(300, run);
end
assert(isequal(unique(w), 101:105), 'five weight draws');
assert(numel(unique(d)) == 15, 'fifteen distinct datasets');
assert(isequal(w(1:5), w(6:10)) && isequal(w(1:5), w(11:15)), ...
    'each weight draw is reused across the three datasets');
[d400, w400] = adding_seeds(400, 7);
assert(w400 == w(7), 'the weight seed does not depend on T');
assert(d400 ~= d(7), 'the dataset seed does');
ok('adding_seeds gives 5 weight draws x 3 datasets, weights independent of T');

% ------------------------------------------------------------- the network --
net = make_adding_narx('Example', ds.train, 'ActivationFcn', "poslin");
assert(isequal(net.inputWeights{1,1}.delays, 0), 'no input TDL');
assert(isequal(net.layerWeights{1,2}.delays, 1:8), 'feedback taps 1..8');
assert(isequal(size(net.IW{1,1}), [2 2]));
assert(isequal(size(net.LW{1,2}), [2 8]));
assert(isequal(size(net.LW{2,1}), [1 2]));
assert(numel(getwb(net)) == 25, '25 trainable parameters');
assert(isempty(net.divideFcn), 'no validation split, so no early stopping');
ok('narxnet(0, 1:8, 2, ''closed''): p(t) only, feedback through LW^{1,2}, 25 parameters');

half = 0.5 / sqrt(2);
expected = zeros(2, 8); expected(:, 1) = half; expected(:, 8) = half;
assert(max(abs(net.LW{1,2} - expected), [], 'all') < 1e-15);
assert(max(abs(net.LW{2,1} - ones(1,2)/sqrt(2))) < 1e-15);
assert(~any(net.b{1}) && ~any(net.b{2}));
assert(abs(std(net.IW{1,1}(:)) - 0.01) < 0.02);
ok('poslin init: LW^{1,2} and LW^{2,1} as designed, zero biases, small IW^{1,1}');

g = net.LW{2,1} * net.LW{1,2};
assert(max(abs(g - [0.5 zeros(1,6) 0.5])) < 1e-15, 'loop vector');
assert(abs(sum(g) - 1) < 1e-12, 'unit DC gain');
A = [g; eye(7), zeros(7,1)];
assert(abs(max(abs(eig(A))) - 1) < 1e-9, 'spectral radius 1');
assert(abs(polyval([1 -g], 1)) < 1e-12, 'z = 1 is a root of the loop polynomial');
ok('loop vector g = [0.5 0 ... 0 0.5], sum(g) = 1, spectral radius 1, z = 1 a root');

tn = make_adding_narx('Example', ds.train, 'ActivationFcn', "tansig", 'Seed', 101);
assert(strcmp(tn.layers{1}.initFcn, 'initnw') && strcmp(tn.initFcn, 'initlay'));
assert(any(tn.b{1} ~= 0), 'Nguyen-Widrow spreads the biases; ours would be zero');
again = make_adding_narx('Example', ds.train, 'ActivationFcn', "tansig", 'Seed', 101);
assert(isequal(tn.IW{1,1}, again.IW{1,1}) && isequal(tn.b{1}, again.b{1}));
ok('tansig keeps MATLAB''s Nguyen-Widrow default, reproducibly');

% ---------------------------------------------------------------- the loss --
T = 60; Q = size(ds.train.pValues, 2);
y = repmat({zeros(1, Q)}, 1, T); y{end} = ds.train.yFinal;
ew = repmat({zeros(1, Q)}, 1, T); ew{end} = T * ones(1, Q);
a2 = net(adding_input_cells(ds.train));
masked = perform(net, y, a2, ew);
final = mean((ds.train.yFinal - a2{end}).^2);
assert(abs(masked - final) / final < 1e-9);
ok('the masked loss equals the final-time MSE and ignores placeholder targets');

% ----------------------------------------------- the recurrence, explicitly --
hist = zeros(8, Q);
for t = 1:T
    n1 = net.IW{1,1} * [ds.train.pValues(t,:); ds.train.pMarkers(t,:)] ...
         + net.LW{1,2} * hist + net.b{1};
    hist = [net.LW{2,1} * max(n1, 0) + net.b{2}; hist(1:end-1, :)];
end
assert(max(abs(hist(1,:) - a2{end})) < 1e-9);
ok('the explicit zero-history closed-loop recurrence matches sim');

fprintf('\nAll checks passed.\n\n');
end


function ok(what)
fprintf('  ok  %s\n', what);
end
