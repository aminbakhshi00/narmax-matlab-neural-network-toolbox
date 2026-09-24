function net = make_varlen_narx(example, activationFcn, delay, seed)
%MAKE_VARLEN_NARX The variable-length model: closed-loop NARX, S^1 = 2, D feedback taps.
%
%   NET = MAKE_VARLEN_NARX(EXAMPLE, ACTIVATIONFCN, DELAY, SEED) is
%   MAKE_ADDING_NARX with 'Neurons' 2, so each activation keeps its own
%   initialisation: poslin the lag-1/lag-D skip design, tansig MATLAB's
%   Nguyen-Widrow default.
%
%   One change, for poslin at D = 1 only: the lag-1 and lag-D taps are then
%   the same tap, and MAKE_ADDING_NARX would leave the loop vector
%   g = LW^{2,1} LW^{1,2} at 0.5. Giving that tap the full 1/sqrt(2) restores
%   g = 1, the unit DC gain the skip design has at every other D.

net = make_adding_narx('Example', example, 'ActivationFcn', activationFcn, ...
    'Seed', seed, 'Neurons', 2, 'Delay', delay);
if activationFcn == "poslin" && delay == 1
    net.LW{1,2}(:) = 1 / sqrt(2);
end
end
