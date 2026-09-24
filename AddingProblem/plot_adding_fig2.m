function plot_adding_fig2(options)
%PLOT_ADDING_FIG2 Le2015 Figure 2, redrawn for the closed-loop NARX.
%
%   Reads the results of RUN_ADDING for both activation functions and draws the
%   four-panel figure of Le2015 Figure 2: held-out final-time MSE against
%   training epoch, one panel per sequence length, with the always-predict-
%   the-mean reference as a dotted line.
%
%   The two curves are labelled by their activation function alone. Each comes
%   with its own initialisation -- the lag-1/lag-8 skip for poslin, MATLAB's
%   Nguyen-Widrow default for tansig -- which the README describes; putting
%   that in the legend would crowd the panels for no gain.
%
%   HOW THE MEDIAN IS TAKEN. The runs stop at different epochs and reach
%   different losses, so there is no "median run" to plot. The bold line is a
%   POINTWISE median instead, built in three steps:
%
%     1. every run is resampled onto the common epoch grid 0, GRIDSTEP, ... ;
%     2. a run that has already ended is held at its final held-out MSE for
%        the rest of the grid, because that is the model it finished with and
%        the number it would be judged on at any later epoch;
%     3. at each grid point the median is taken across the runs.
%
%   So the bold line is the median of 15 numbers at every epoch, not the
%   trajectory of any single run, and it can follow a path no run took. Two
%   consequences are worth reading off it: it flattens once more than half the
%   runs have stopped, and it sits in the solved region only when more than
%   half the runs solved -- where fewer than half did, it lies on the baseline
%   and is reporting the failure rate rather than the achievable accuracy.
%
%   Holding an ended run at its last value rather than dropping it keeps the
%   median over a fixed 15 runs at every epoch. Dropping them would shrink the
%   sample as the grid advances and let the median drift simply because the
%   slower runs are the only ones left.
%
%   THE TWO AXES. Le2015's y axis is linear from 0 to 0.8, kept in the main
%   figure so the shapes are comparable. But a run that solves the task
%   reaches 1e-7 and is indistinguishable from zero on a linear axis, so a
%   second figure repeats the same data on a logarithmic axis. Use that one.
%
%   Curves are resampled onto a grid of GRIDSTEP epochs, which should match
%   the BLOCKEPOCHS the runs were recorded at; a finer grid interpolates
%   between real samples, and on the logarithmic axis a linearly interpolated
%   midpoint does not lie on the line between its neighbours.
%
%   NAME-VALUE OPTIONS
%     Folders    Results folders to read. Default the two RUN_ADDING writes.
%                The figures go into the first.
%     LogFloor   Bottom of the logarithmic axis. Curves are clamped to it, so
%                it must sit below the best run. Default 1e-10.
%     GridStep   Epoch spacing. Default 100.
%
%   See also RUN_ADDING, PLOT_ADDING_HEATMAPS.

arguments
    options.Folders (1,:) string = ["results_poslin", "results_tansig"]
    options.LogFloor (1,1) double {mustBePositive} = 1e-10
    options.GridStep (1,1) double {mustBePositive} = 100
end

here = fileparts(mfilename('fullpath'));
T = table();
for folder = options.Folders
    part = load(fullfile(here, folder, 'results.mat'), 'result').result.table;
    T = [T; part(:, {'sequenceLength', 'activationFcn', 'run', 'baselineMSE', ...
                     'curveEpoch', 'curveTestMSE'})]; %#ok<AGROW>
end

lengths = unique(T.sequenceLength).';
series = ["poslin", "tansig"];
labels = ["poslin", "tansig"];
colours = [0.85 0.33 0.10; 0.29 0.23 0.65];
runsPerLength = sum(T.activationFcn == series(1)) / numel(lengths);

for isLog = [false true]
    fig = figure('Color', 'w', 'Units', 'pixels', ...
        'Position', [60 60 1500 420], 'Visible', 'off');
    tiles = tiledlayout(fig, 1, numel(lengths), 'TileSpacing', 'compact', ...
        'Padding', 'compact');

    for index = 1:numel(lengths)
        ax = nexttile(tiles); hold(ax, 'on'); grid(ax, 'on');
        % This MATLAB release themes axes dark; the figure is for print, so
        % every colour is set explicitly.
        set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', ...
            'GridColor', [0.15 0.15 0.15], 'GridAlpha', 0.15);
        if isLog, set(ax, 'YScale', 'log'); end

        for k = 1:2
            mask = T.activationFcn == series(k) & T.sequenceLength == lengths(index);
            if ~any(mask), continue; end
            block = resample(T(mask, :), options.GridStep);
            grid_ = 0:options.GridStep:(size(block, 2) - 1) * options.GridStep;
            if isLog, block = max(block, options.LogFloor); end

            plot(ax, grid_, block, '-', 'Color', [colours(k,:) 0.22], ...
                'LineWidth', 0.7, 'HandleVisibility', 'off');
            plot(ax, grid_, median(block, 1), '-', 'Color', colours(k,:), ...
                'LineWidth', 2.2, 'DisplayName', labels(k));
        end

        yline(ax, mean(T.baselineMSE(T.sequenceLength == lengths(index))), ':', ...
            'Color', [0.35 0.35 0.35], 'LineWidth', 1.4, 'HandleVisibility', 'off');
        title(ax, sprintf('T = %d', lengths(index)), 'Color', 'k');
        xlabel(ax, 'training epochs', 'Color', 'k');
        if isLog, ylim(ax, [options.LogFloor 1]); else, ylim(ax, [0 0.8]); end
        if index == 1
            ylabel(ax, 'held-out MSE of a^2(T)', 'Color', 'k');
            % The curves fall from the top left, so the free corner is the
            % bottom left on the logarithmic axis and the top right on the
            % linear one, where everything has already collapsed onto zero.
            if isLog, corner = 'southwest'; else, corner = 'northeast'; end
            lg = legend(ax, 'Location', corner, 'Interpreter', 'none', ...
                'FontSize', 9);
            set(lg, 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.6 0.6 0.6]);
        end
    end

    title(tiles, ['Adding two numbers in a sequence of T numbers, ', ...
        'closed-loop NARX'], 'Color', 'k');
    subtitle(tiles, sprintf(['D = 8, S^1 = 2 (25 parameters), 5000 train / ' ...
        '5000 test, trainlm, %d runs per length. Bold = pointwise median ' ...
        'over runs, faint = individual runs, dotted = always predict the ' ...
        'mean.'], runsPerLength), 'Color', [0.25 0.25 0.25]);

    if isLog, name = 'adding_fig2_log.png'; else, name = 'adding_fig2.png'; end
    target = fullfile(here, options.Folders(1), name);
    exportgraphics(fig, target, 'Resolution', 150);
    close(fig);
    fprintf('wrote %s\n', fullfile(options.Folders(1), name));
end
end


function block = resample(rows, step)
%RESAMPLE Put every run on a common epoch grid, holding its last value once
%   it has ended, so curves of different length can be drawn together.
last = max(cellfun(@max, rows.curveEpoch));
grid_ = 0:step:last;
block = nan(height(rows), numel(grid_));
for j = 1:height(rows)
    e = rows.curveEpoch{j};
    v = rows.curveTestMSE{j};
    % A stalled run re-entered by a later call can record the same epoch
    % twice, and INTERP1 rejects repeated sample points.
    [e, keep] = unique(e, 'stable');
    v = v(keep);
    if numel(e) < 2
        block(j, :) = v(1);
    else
        block(j, :) = interp1(e, v, min(grid_, max(e)), 'linear');
    end
end
end
