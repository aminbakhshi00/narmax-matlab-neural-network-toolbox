function pCells = adding_input_cells(split)
%ADDING_INPUT_CELLS The external input p(t) as MATLAB expects it.
%
%   PCELLS = ADDING_INPUT_CELLS(SPLIT) returns a 1-by-T cell array whose t-th
%   entry is the 2-by-Q matrix
%
%     p(t) = [ p_1(t) ; p_2(t) ]
%
%   across the Q concurrent sequences of SPLIT: row 1 is the uniform value
%   channel and row 2 the marker channel. R^1 = 2.
%
%   See also GENERATE_ADDING_DATASET.

finalTime = size(split.pValues, 1);
pCells = cell(1, finalTime);
for t = 1:finalTime
    pCells{t} = [split.pValues(t, :); split.pMarkers(t, :)];
end
end
