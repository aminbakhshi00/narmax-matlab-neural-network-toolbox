classdef ARXmodel < NARXmodel
%ARXMODEL Linear ARX model: NARXmodel with the hidden layer removed.
%
%   ARXMODEL(DELAY) builds the linear difference equation
%
%     yHat(t) = sum_{i=1..DELAY} a_i y(t-i)
%             + sum_{i=1..DELAY} b_i p(t-i) + c
%
%   and fits it exactly as NARXMODEL fits the nonlinear network: the same
%   [0, 1] normalisation, the same closed-loop horizon schedule, the same
%   trainlm, and the same closed-loop rollout at prediction time. The model is the only thing that changes, so the
%   difference in held-out error is the value of the nonlinearity.
%
%   There is no hidden layer, and therefore no hidden transfer function to
%   choose. A hidden layer of purelin units would add nothing: the
%   composition of two linear maps is one linear map,
%
%     LW^{2,1}( IW^{1,1} z + b^1 ) + b^2 = ( LW^{2,1} IW^{1,1} ) z + const,
%
%   and since the output is scalar that product is a single 1-by-2*DELAY
%   row of weights. Any linear model it could express is already expressed
%   by one layer. Keeping the hidden layer would carry 501 parameters where
%   2*DELAY+1 suffice and leave trainlm solving a rank-deficient problem,
%   because infinitely many (IW^{1,1}, LW^{2,1}) give the same map.
%
%   Passing [] as narxnet's hidden size is what removes the layer, so the
%   network is one purelin layer reading both tapped delay lines: 25
%   weights at DELAY = 12, 49 at DELAY = 24. Every NARXMODEL property
%   (trainAlg, iterPerRun, earlyStoppage, ...) applies unchanged, and the
%   checkpoints it writes are read by NARXFORECAST like any other.
%
%   The one part of NARXMODEL's procedure that is switched off is the
%   valley-escape heuristic, USEMODIFIEDTRAINING. It exists to free a
%   nonconvex fit that trainlm has stalled on, and it decides it has
%   stalled when mu reaches mu_max. A linear model reaches mu_max at
%   essentially every horizon step because it has simply converged, so the
%   heuristic fires constantly and spends hours re-deriving the same
%   weights. The horizon schedule, the epoch budget and everything else
%   are unchanged.
%
%   Early stopping is off, and trainbr loses nothing by it: Bayesian
%   regularisation replaces early stopping, and given a 70/15/15 split trainbr
%   folds the validation sequences back into training and stops on the epoch
%   count anyway. The split's only real effect was to withhold the inner test
%   block, so switching it off hands that 15% back to the fit -- measured as
%   better at every delay tried: 72.95 against 77.76 MW at 24 taps, 71.48
%   against 71.89 at 48, 75.39 against 76.21 at 12.
%
%   See also NARXMODEL, NARXFORECAST, RUN_LOAD_ARX.

    methods (Access=public)

        function obj = ARXmodel(delay)
            obj@NARXmodel(delay, []);        % [] hidden units = no hidden layer
            obj.useModifiedTraining = false;
            obj.earlyStoppage = false;
        end

    end

end
