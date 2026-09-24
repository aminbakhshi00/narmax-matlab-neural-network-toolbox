function plot_varlen_fig7(options)
%PLOT_VARLEN_FIG7 Held-out sequences through every solved variable-length network.
%
%   The layout of MIPS Fig. 7, one figure per solved run: a^2(t) (solid)
%   against the partial sum of marked p_1 up to t (dashed), on three held-out
%   sequences. Marks are dots on the dashed line; at each sequence's own final
%   time L a filled marker is a^2(L) and an open one y(L). Only a^2(L) is
%   scored, and the network is never told L.
%
%   The three sequences have lengths MINLENGTH, the midpoint and MAXLENGTH of
%   the trained range, with y(L) nearest the 10th, 50th and 90th percentile
%   of y over the whole held-out set; each trace stops at its own L. The
%   held-out set is regenerated exactly as RUN_VARIABLE_LENGTH drew it.
%
%   NAME-VALUE OPTIONS
%     Folder         Default "results_T300-400".
%     ActivationFcn  Default "poslin".
%     Delays         Only these D. Default [], every D.
%
%   Writes fig7_<tag>.png into the results folder.
%
%   See also RUN_VARIABLE_LENGTH, PLOT_VARLEN_HEATMAPS.

arguments
    options.Folder (1,1) string = "results_T300-400"
    options.ActivationFcn (1,1) string = "poslin"
    options.Delays (1,:) double = []
end

colours = [0.165 0.471 0.839; 0.922 0.408 0.204; 0.106 0.686 0.478];  % blue, orange, aqua
ink = [0.04 0.04 0.04]; muted = [0.32 0.32 0.31];
folder = fullfile(fileparts(mfilename('fullpath')), options.Folder);
T = load(fullfile(folder, 'results.mat'), 'result').result.table;

chosen = T.solved & T.activationFcn == options.ActivationFcn;
if ~isempty(options.Delays), chosen = chosen & ismember(T.delay, options.Delays); end
for k = find(chosen).'
    minLength = T.minLength(k); maxLength = T.maxLength(k);
    tag = sprintf('T%d-%d_%s_D%d_run%02d', minLength, maxLength, ...
        T.activationFcn(k), T.delay(k), T.run(k));
    net = load(fullfile(folder, tag + ".mat"), 'state').state.net;
    test = make_varlen_dataset(minLength, maxLength, 5000, 5000, T.run(k)).test;

    a2Cells = net(adding_input_cells(test));
    a2 = vertcat(a2Cells{:});
    partialSum = cumsum(test.pValues .* test.pMarkers, 1);
    Q = numel(test.lengths);
    mse = mean((test.yFinal - a2(sub2ind(size(a2), test.lengths, 1:Q))).^2);

    ySorted = sort(test.yFinal);
    targets = ySorted(round([0.1 0.5 0.9] * Q));
    showLengths = [minLength, round((minLength + maxLength) / 2), maxLength];
    pick = zeros(1, 3);
    for q = 1:3
        candidates = find(test.lengths == showLengths(q));
        [~, nearest] = min(abs(test.yFinal(candidates) - targets(q)));
        pick(q) = candidates(nearest);
    end

    fig = figure('Color', 'w', 'Units', 'pixels', 'Position', [40 40 900 640], ...
        'Visible', 'off');
    theme(fig, 'light');
    ax = axes(fig);
    hold(ax, 'on');
    band = patch(ax, [0 maxLength + 5 maxLength + 5 0], [0 0 2 2], colours(3, :), ...
        'FaceAlpha', 0.08, 'EdgeColor', 'none');
    trained = patch(ax, [minLength maxLength maxLength minLength], [-10 -10 10 10], ...
        muted, 'FaceAlpha', 0.07, 'EdgeColor', 'none');
    yline(ax, 0, 'Color', muted, 'LineWidth', 0.75);
    lines = gobjects(1, 3);
    for q = 1:3
        s = pick(q); L = test.lengths(s);
        plot(ax, 1:L, partialSum(1:L, s), '--', 'Color', colours(q, :), 'LineWidth', 1.4);
        marks = test.markIndex(:, s);
        scatter(ax, marks, partialSum(marks, s), 36, colours(q, :), 'filled');
        lines(q) = plot(ax, 1:L, a2(1:L, s), '-', 'Color', colours(q, :), 'LineWidth', 2);
        scatter(ax, L, a2(L, s), 70, colours(q, :), 'filled', 'MarkerEdgeColor', ink);
        scatter(ax, L, test.yFinal(s), 90, ink, 'o', 'LineWidth', 1.2);
    end
    openCircle = plot(ax, NaN, NaN, 'o', 'Color', ink, 'LineWidth', 1.2);
    hold(ax, 'off');

    set(ax, 'Color', 'w', 'XColor', muted, 'YColor', muted, 'TickDir', 'out', ...
        'Box', 'off', 'FontSize', 10, 'Layer', 'top');
    grid(ax, 'on');
    shown = [a2(:, pick); partialSum(:, pick)];
    xlim(ax, [1 maxLength + 5]);
    ylim(ax, [min(-0.1, min(shown, [], 'all') - 0.1), max(2.2, max(shown, [], 'all') + 0.1)]);
    xlabel(ax, 'time step t', 'Color', ink);
    ylabel(ax, 'a^2(t) (solid)   |   partial sum of marked p_1 (dashed)', 'Color', ink);
    entries = compose("L = %d:  y(L) = %.2f,  a^2(L) = %.2f", test.lengths(pick).', ...
        test.yFinal(pick).', a2(sub2ind(size(a2), test.lengths(pick), pick)).');
    legend(ax, [lines, band, trained, openCircle], [entries; "range of y(L)"; ...
        sprintf("trained final times L = %d..%d", minLength, maxLength); ...
        "open circle: target y(L)"], 'Location', 'best', 'Box', 'off', ...
        'TextColor', ink, 'FontSize', 9);
    title(ax, sprintf(['%s, S^1 = 2, D = %d, run %d   (closed-loop NARX, trained on ', ...
        'T = %d..%d)\nheld-out MSE %.1e over mixed lengths, %d epochs, stop: %s'], ...
        T.activationFcn(k), T.delay(k), T.run(k), minLength, maxLength, mse, ...
        T.epochs(k), T.stopReason(k)), 'Color', ink, 'FontWeight', 'normal', 'FontSize', 11);

    exportgraphics(fig, fullfile(folder, "fig7_" + tag + ".png"), 'Resolution', 200);
    close(fig);
    fprintf('%-32s  saved MSE %.3e   recomputed %.3e\n', tag, T.testMSE(k), mse);
end
end
