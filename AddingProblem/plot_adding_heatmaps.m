function plot_adding_heatmaps(options)
%PLOT_ADDING_HEATMAPS What the successful networks learned.
%
%   For every run that solved the task, draws before-and-after heatmaps of
%   IW^{1,1}, LW^{1,2} and LW^{2,1} with the numeric value printed in every
%   cell, and writes one PNG per run into the results folder.
%
%   Each figure is a 2-by-3 grid: the top row is the initialisation, the
%   bottom row is what training arrived at. Colour is a diverging scale
%   centred on zero and shared between the two rows of a given weight, so a
%   cell that changed reads as a colour change and not only as a number.
%   Cells whose magnitude exceeds HIGHLIGHTFACTOR times the median magnitude
%   of that matrix are outlined.
%
%   NAME-VALUE OPTIONS
%     Folder           Results folder. Default "results_poslin".
%     SolvedThreshold  Held-out MSE below which a run counts as solved.
%                      Default 0.01.
%     HighlightFactor  Outline cells this many times the median magnitude.
%                      Default 3.
%
%   See also RUN_ADDING, PLOT_ADDING_FIG2.

arguments
    options.Folder (1,1) string = "results_poslin"
    options.SolvedThreshold (1,1) double = 0.01
    options.HighlightFactor (1,1) double = 3
end

here = fileparts(mfilename('fullpath'));
outputFolder = fullfile(here, options.Folder);
T = load(fullfile(outputFolder, 'results.mat'), 'result').result.table;

rowsToDraw = find(T.testMSE < options.SolvedThreshold).';
fprintf('\n%d of %d runs solved (held-out MSE < %.3g)\n', numel(rowsToDraw), ...
    height(T), options.SolvedThreshold);
if isempty(rowsToDraw), fprintf('Nothing to draw.\n\n'); return; end

for k = rowsToDraw
    tag = sprintf('T%03d_%s_run%02d', T.sequenceLength(k), T.activationFcn(k), T.run(k));
    state = load(fullfile(outputFolder, tag + ".mat"), 'state').state;
    after = struct('IW', state.net.IW{1,1}, 'LW12', state.net.LW{1,2}, ...
                   'LW21', state.net.LW{2,1});

    fig = figure('Color', 'w', 'Units', 'pixels', ...
        'Position', [60 60 1500 760], 'Visible', 'off');
    tiles = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    panels = {state.initial.IW, state.initial.LW12, state.initial.LW21; ...
              after.IW,         after.LW12,         after.LW21};
    names = {sprintf('IW^{1,1}  (%d x %d)', size(after.IW)), ...
             sprintf('LW^{1,2}  (%d x %d)', size(after.LW12)), ...
             sprintf('LW^{2,1}  (%d x %d)', size(after.LW21))};
    stage = {'before training', 'after training'};

    for col = 1:3
        limit = max(abs([panels{1, col}(:); panels{2, col}(:)]));
        if limit == 0, limit = 1; end
        for rowIdx = 1:2
            ax = nexttile(tiles, (rowIdx - 1) * 3 + col);
            draw_matrix(ax, panels{rowIdx, col}, limit, options.HighlightFactor);
            title(ax, sprintf('%s, %s', names{col}, stage{rowIdx}), ...
                'Color', 'k', 'FontSize', 10);
        end
    end

    title(tiles, sprintf(['%s, T = %d, run %d  --  held-out MSE %.3g ', ...
        '(%.3fx baseline), %d epochs'], T.activationFcn(k), T.sequenceLength(k), ...
        T.run(k), T.testMSE(k), T.ratioToBaseline(k), T.epochs(k)), 'Color', 'k');
    subtitle(tiles, ['Diverging scale centred on zero, shared between the two ', ...
        'rows of each weight. Outlined cells exceed ', ...
        sprintf('%gx', options.HighlightFactor), ...
        ' the median magnitude of that matrix.'], 'Color', [0.25 0.25 0.25]);

    exportgraphics(fig, fullfile(outputFolder, "heatmap_" + tag + ".png"), ...
        'Resolution', 150);
    close(fig);
end
fprintf('wrote %d heatmaps into %s\n\n', numel(rowsToDraw), options.Folder);
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
