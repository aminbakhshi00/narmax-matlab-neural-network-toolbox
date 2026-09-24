function plot_varlen_heatmaps(options)
%PLOT_VARLEN_HEATMAPS What the successful variable-length networks learned.
%
%   PLOT_ADDING_HEATMAPS for the variable-length results, with the biases
%   added. For every run that solved the task, draws before-and-after
%   heatmaps of IW^{1,1}, LW^{1,2}, LW^{2,1}, b^1 and b^2 with the value
%   printed in every cell, and writes heatmap_<tag>.png into the results
%   folder. Top row: the initialisation. Bottom row: after training.
%
%   Colour is a diverging scale centred on zero and shared between the two
%   rows of a weight. Cells whose magnitude exceeds HIGHLIGHTFACTOR times the
%   median magnitude of that matrix are outlined.
%
%   NAME-VALUE OPTIONS
%     Folders          Default ["results_T150-200" "results_T300-400"].
%     SolvedThreshold  Default 0.01.
%     HighlightFactor  Default 3.
%     Delays           Only these D. Default [], every D.
%
%   See also RUN_VARIABLE_LENGTH, PLOT_ADDING_HEATMAPS.

arguments
    options.Folders (1,:) string = ["results_T150-200" "results_T300-400"]
    options.SolvedThreshold (1,1) double = 0.01
    options.HighlightFactor (1,1) double = 3
    options.Delays (1,:) double = []
end

here = fileparts(mfilename('fullpath'));
fields = ["IW", "LW12", "LW21", "b1", "b2"];
names = ["IW^{1,1}", "LW^{1,2}", "LW^{2,1}", "b^1", "b^2"];
stage = ["before training", "after training"];

for folderName = options.Folders
    folder = fullfile(here, folderName);
    T = load(fullfile(folder, 'results.mat'), 'result').result.table;
    chosen = T.testMSE < options.SolvedThreshold;
    if ~isempty(options.Delays), chosen = chosen & ismember(T.delay, options.Delays); end
    rowsToDraw = find(chosen).';
    fprintf('%s: drawing %d solved runs\n', folderName, numel(rowsToDraw));

    for k = rowsToDraw
        tag = sprintf('T%d-%d_%s_D%d_run%02d', T.minLength(k), T.maxLength(k), ...
            T.activationFcn(k), T.delay(k), T.run(k));
        state = load(fullfile(folder, tag + ".mat"), 'state').state;
        before = state.initial;
        after = struct('IW', state.net.IW{1,1}, 'LW12', state.net.LW{1,2}, ...
            'LW21', state.net.LW{2,1}, 'b1', state.net.b{1}, 'b2', state.net.b{2});

        spans = [3, max(T.delay(k), 2) + 1, 3, 2, 2];   % tile width per matrix
        fig = figure('Color', 'w', 'Units', 'pixels', ...
            'Position', [60 60 max(1300, 140 * sum(spans)) 760], 'Visible', 'off');
        theme(fig, 'light');
        tiles = tiledlayout(fig, 2, sum(spans), 'TileSpacing', 'compact', ...
            'Padding', 'compact');
        for f = 1:numel(fields)
            limit = max(abs([before.(fields(f))(:); after.(fields(f))(:)]));
            if limit == 0, limit = 1; end
            panels = {before.(fields(f)), after.(fields(f))};
            for stageIdx = 1:2
                tile = (stageIdx - 1) * sum(spans) + sum(spans(1:f - 1)) + 1;
                ax = nexttile(tiles, tile, [1 spans(f)]);
                draw_matrix(ax, panels{stageIdx}, limit, options.HighlightFactor);
                title(ax, sprintf('%s  (%d x %d), %s', names(f), ...
                    size(panels{stageIdx}), stage(stageIdx)), 'Color', 'k', 'FontSize', 10);
            end
        end

        % Equal-aspect tiles confuse the layout's own title, so it sits above.
        tiles.OuterPosition = [0 0 1 0.86];
        annotation(fig, 'textbox', [0 0.93 1 0.06], 'String', sprintf(['%s, ', ...
            'S^1 = 2, D = %d, run %d, trained on T = %d..%d  --  held-out MSE %.3g ', ...
            '(%.3fx baseline), %d epochs'], T.activationFcn(k), T.delay(k), T.run(k), ...
            T.minLength(k), T.maxLength(k), T.testMSE(k), T.ratioToBaseline(k), ...
            T.epochs(k)), 'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
            'FontSize', 13, 'Color', 'k');
        annotation(fig, 'textbox', [0 0.88 1 0.05], 'String', sprintf(['Closed-loop ', ...
            'NARX. Diverging scale centred on zero, shared between the two rows of ', ...
            'each weight. Outlined cells exceed %gx the median magnitude of that ', ...
            'matrix.'], options.HighlightFactor), 'EdgeColor', 'none', ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', [0.25 0.25 0.25]);

        exportgraphics(fig, fullfile(folder, "heatmap_" + tag + ".png"), 'Resolution', 150);
        close(fig);
    end
end
end


function draw_matrix(ax, M, limit, highlightFactor)
%DRAW_MATRIX One heatmap with the value printed in every cell.
imagesc(ax, M, [-limit limit]);
colormap(ax, diverging_map());
axis(ax, 'equal', 'tight');
set(ax, 'XColor', 'k', 'YColor', 'k', 'TickLength', [0 0], ...
    'XAxisLocation', 'top', 'FontSize', 8);
xticks(ax, 1:size(M, 2)); yticks(ax, 1:size(M, 1));
cb = colorbar(ax); set(cb, 'Color', 'k');

magnitudes = abs(M(:));
typical = median(magnitudes(magnitudes > 0));
if isempty(typical) || typical == 0, typical = Inf; end

hold(ax, 'on');
for r = 1:size(M, 1)
    for c = 1:size(M, 2)
        value = M(r, c);
        if abs(value) > 0.55 * limit, textColour = 'w'; else, textColour = 'k'; end
        text(ax, c, r, format_cell(value), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'FontSize', 7, 'Color', textColour);
        if abs(value) >= highlightFactor * typical
            rectangle(ax, 'Position', [c - 0.5, r - 0.5, 1, 1], ...
                'EdgeColor', [0.95 0.75 0.05], 'LineWidth', 2);
        end
    end
end
hold(ax, 'off');
end


function s = format_cell(value)
%FORMAT_CELL Short enough to fit, precise enough to read.
if value == 0
    s = '0';
elseif abs(value) >= 100 || abs(value) < 1e-3
    s = sprintf('%.0e', value);
else
    s = sprintf('%.3f', value);
end
end


function map = diverging_map()
%DIVERGING_MAP Blue to white to orange, so zero is the pale midpoint.
n = 128;
blue = [0.16 0.44 0.71]; orange = [0.85 0.37 0.10]; white = [1 1 1];
lower = [linspace(blue(1), white(1), n).', linspace(blue(2), white(2), n).', ...
         linspace(blue(3), white(3), n).'];
upper = [linspace(white(1), orange(1), n).', linspace(white(2), orange(2), n).', ...
         linspace(white(3), orange(3), n).'];
map = [lower; upper(2:end, :)];
end
