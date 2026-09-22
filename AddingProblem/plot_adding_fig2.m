function plot_adding_fig2(options)
%PLOT_ADDING_FIG2 Le2015 Figure 2, redrawn for the closed-loop NARX.
%
%   Reads the results of RUN_ADDING for both transfer functions and draws the
%   four-panel figure of Le2015 Figure 2: held-out final-time MSE against
%   training epoch, one panel per sequence length, with the always-predict-
%   the-mean reference as a dotted line.
%
%   Le2015 compares an IRNN, an LSTM, an RNN with tanh and an RNN with ReLUs.
%   This project has no RNN and no LSTM, so the two curves are the NARX
%   counterparts of the two models it does have:
%
%     poslin + lag-1/lag-8 skip   the IRNN analogue
%     tansig + Nguyen-Widrow      MATLAB's default, under a saturating f^1
%
%   TWO DIFFERENCES FROM THE PAPER'S FIGURE ARE WORTH KNOWING.
%
%   First, Le2015 plots "the best result over the grid search", so each of its
%   curves is a single selected run. The bold line here is likewise the BEST
%   run, matching that protocol. The median is drawn thin and dashed and the
%   individual runs faint, so the spread stays visible and it is clear how
%   many runs reached the bold line. Selecting on the best means reporting
%   what a configuration can reach, not what it reaches typically.
%
%   Second, the paper's y axis is linear from 0 to 0.8. That is kept in the
%   main figure so the shapes are comparable, but a run that solves the task
%   reaches 1e-7 and is indistinguishable from zero on a linear axis, so a
%   second figure repeats the same data on a logarithmic axis.
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
    T = [T; part(:, {'sequenceLength', 'transferFcn', 'run', 'baselineMSE', ...
                     'curveEpoch', 'curveTestMSE'})]; %#ok<AGROW>
end

lengths = unique(T.sequenceLength).';
series = ["poslin", "tansig"];
labels = ["poslin + lag-1/lag-8 skip  (IRNN analogue)", ...
          "tansig + Nguyen-Widrow"];
colours = [0.85 0.33 0.10; 0.29 0.23 0.65];
for k = 1:2
    labels(k) = labels(k) + sprintf('  [%d runs per T]', ...
        sum(T.transferFcn == series(k)) / numel(lengths));
end

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
            mask = T.transferFcn == series(k) & T.sequenceLength == lengths(index);
            if ~any(mask), continue; end
            block = resample(T(mask, :), options.GridStep);
            grid_ = 0:options.GridStep:(size(block, 2) - 1) * options.GridStep;
            if isLog, block = max(block, options.LogFloor); end

            plot(ax, grid_, block, '-', 'Color', [colours(k,:) 0.22], ...
                'LineWidth', 0.7, 'HandleVisibility', 'off');
            plot(ax, grid_, median(block, 1), '--', 'Color', colours(k,:), ...
                'LineWidth', 1.0, 'HandleVisibility', 'off');
            [~, best] = min(block(:, end));
            plot(ax, grid_, block(best, :), '-', 'Color', colours(k,:), ...
                'LineWidth', 2.2, 'DisplayName', labels(k));
        end

        yline(ax, mean(T.baselineMSE(T.sequenceLength == lengths(index))), ':', ...
            'Color', [0.35 0.35 0.35], 'LineWidth', 1.4, 'HandleVisibility', 'off');
        title(ax, sprintf('T = %d', lengths(index)), 'Color', 'k');
        xlabel(ax, 'training epochs', 'Color', 'k');
        if isLog, ylim(ax, [options.LogFloor 1]); else, ylim(ax, [0 0.8]); end
        if index == 1
            ylabel(ax, 'held-out MSE of a^2(T)', 'Color', 'k');
            lg = legend(ax, 'Location', 'northeast', 'Interpreter', 'none', ...
                'FontSize', 8);
            set(lg, 'TextColor', 'k', 'Color', 'w', 'EdgeColor', [0.6 0.6 0.6]);
        end
    end

    title(tiles, ['Adding two numbers in a sequence of T numbers, ', ...
        'closed-loop NARX'], 'Color', 'k');
    subtitle(tiles, ['D = 8, S^1 = 2 (25 parameters), 5000 train / 5000 test, ', ...
        'trainlm. Bold = best run, dashed = median, faint = individual runs, ', ...
        'dotted = always predict the mean.'], 'Color', [0.25 0.25 0.25]);

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
