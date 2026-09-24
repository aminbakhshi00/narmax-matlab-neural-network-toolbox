function net = make_adding_narx(options)
%MAKE_ADDING_NARX The closed-loop NARX used for the adding problem.
%
%   NET = MAKE_ADDING_NARX('Example', SPLIT, 'ActivationFcn', F) builds
%
%       narxnet(0, 1:8, 2, 'closed', 'trainlm')
%
%   so layer 1 sees the current external input p(t) only -- there is no input
%   tapped delay line, so no earlier p is reachable -- and the single
%   recurrent path is the parallel (closed-loop) output feedback
%   a^2(t-1) ... a^2(t-8) through LW^{1,2}. There is no LW^{1,1}
%   self-connection. The feedback history is zero at the start of every
%   sequence.
%
%       n^1(t) = IW^{1,1} p(t) + LW^{1,2} [a^2(t-1); ...; a^2(t-8)] + b^1
%       a^1(t) = f^1(n^1(t))
%       a^2(t) = LW^{2,1} a^1(t) + b^2 = yHat(t)
%
%   f^2 is purelin, so the readout can represent the sum directly. With
%   R^1 = 2, S^1 = 2, S^2 = 1 and D = 8 the network has 25 trainable
%   parameters.
%
%   ACTIVATIONFCN selects f^1 AND, with it, the initialisation. The two are
%   tied together because each activation is paired with the initialisation
%   that suits it, and those are the two models this study compares.
%
%   'poslin' -- the designed lag-1/lag-8 skip initialisation:
%
%       LW^{1,2} = (1/sqrt2) [0.5 0 0 0 0 0 0 0.5      LW^{2,1} = (1/sqrt2) [1 1]
%                             0.5 0 0 0 0 0 0 0.5]     b^1 = b^2 = 0
%                                                      IW^{1,1} ~ N(0, 0.01^2)
%
%     The two 1/sqrt2 factors multiply to 1/2 and the two units sum, so the
%     loop vector is exactly
%
%       g = LW^{2,1} LW^{1,2} = [0.5 0 0 0 0 0 0 0.5],
%
%     meaning a^2(t) reaches back one step and eight steps in the same
%     recurrence: the lag-8 term is a skip connection that delivers credit
%     from t-8 in a single hop instead of eight. The companion
%     characteristic polynomial is z^8 - 0.5 z^7 - 0.5, and because the two
%     taps sum to 1 the point z = 1 is a root of it BY CONSTRUCTION. The loop
%     therefore has unit DC gain from step one: it holds a running total
%     without drift, which is what the task needs. The other seven
%     eigenvalues are strictly inside the unit circle (largest 0.959), all
%     eight are distinct, and cond(V) = 2.17 -- one mode that remembers
%     forever and seven that forget.
%
%     INPUTSTD = 0.01 is the only random part. The scale matters because the
%     network is an accumulator at initialisation: with p_1 ~ U[0,1] the
%     per-step increment has standard deviation about 0.5*InputStd, poslin
%     reflects the walk at zero, and the output drifts as
%     0.5*InputStd*sqrt(T) against a target of mean 1. The working rule is
%     InputStd <~ 2/sqrt(T); at 0.01 the drift is 0.06 to 0.10 across
%     T = 150..400, at 0.1 it is 0.61 to 1.00 and the walk is as large as the
%     target.
%
%   'tansig' -- MATLAB's own default, untouched. NET.INITFCN is 'initlay' and
%     both layers' INITFCN is 'initnw', so CONFIGURE has already applied
%     Nguyen-Widrow to IW^{1,1}, LW^{1,2}, LW^{2,1}, b^1 and b^2. Leaving
%     them alone is the whole implementation. Nguyen-Widrow spreads each
%     unit's active region across the input range, which needs that range to
%     be bounded; it is for tansig, whose ACTIVEINPUTRANGE is [-2 2], and the
%     spread shows up in b^1. The draw comes from the global stream inside
%     CONFIGURE, so SEED is applied before it.
%
%   NAME-VALUE OPTIONS
%     Example      A split from GENERATE_ADDING_DATASET, used to size the
%                  network through CONFIGURE. Required.
%     ActivationFcn  "poslin" or "tansig". Default "poslin".
%     Seed         Weight seed, from ADDING_SEEDS. Default 101.
%     Delay        D, output-feedback taps. Default 8.
%     Neurons      S^1. Default 2.
%     InputStd     sd of IW^{1,1} for poslin. Default 0.01. Unused by tansig,
%                  which draws every weight itself.
%
%   See also RUN_ADDING, ADDING_SEEDS, GENERATE_ADDING_DATASET.

arguments
    options.Example struct
    options.ActivationFcn (1,1) string {mustBeMember(options.ActivationFcn, ...
        ["poslin", "tansig"])} = "poslin"
    options.Seed (1,1) double = 101
    options.Delay (1,1) double {mustBeInteger, mustBePositive} = 8
    options.Neurons (1,1) double {mustBeInteger, mustBePositive} = 2
    options.InputStd (1,1) double {mustBePositive} = 0.01
end

delay = options.Delay;
neurons = options.Neurons;

if options.ActivationFcn == "tansig"
    % CONFIGURE draws the Nguyen-Widrow weights from the global stream, so it
    % has to be seeded before the call. For poslin every weight is overwritten
    % afterwards from a private stream, so the global state never reaches the
    % result and this is deliberately not done there.
    rng(options.Seed, 'twister');
end

net = narxnet(0, 1:delay, neurons, 'closed', 'trainlm');
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
net.layers{1}.transferFcn = char(options.ActivationFcn);   % MATLAB's property name
net.layers{2}.transferFcn = 'purelin';
net.divideFcn = '';                    % no validation split; see RUN_ADDING
net.performFcn = 'mse';

pCells = adding_input_cells(options.Example);
yCells = repmat({zeros(1, size(options.Example.pValues, 2))}, 1, ...
    size(options.Example.pValues, 1));
yCells{end} = options.Example.yFinal;
net = configure(net, pCells, yCells);

if options.ActivationFcn == "poslin"
    taps = size(net.LW{1,2}, 2);
    stream = RandStream('mt19937ar', 'Seed', options.Seed);
    net.IW{1,1} = options.InputStd * randn(stream, size(net.IW{1,1}));
    net.LW{1,2} = zeros(neurons, taps);
    net.LW{1,2}(:, 1) = 0.5 / sqrt(neurons);
    net.LW{1,2}(:, taps) = 0.5 / sqrt(neurons);
    net.LW{2,1} = ones(1, neurons) / sqrt(neurons);
    net.b{1} = zeros(size(net.b{1}));
    net.b{2} = zeros(size(net.b{2}));
end

net.trainParam.min_grad = 1e-7;        % trainlm's own default, restored
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = false;
end
