function adding_narx_architecture
%ADDING_NARX_ARCHITECTURE Draw the network this folder trains.
%
%   MAKE_ADDING_NARX creates
%
%       narxnet(0, 1:D, S1, 'closed', 'trainlm')
%
%   so the architecture is parallel/closed-loop: the feedback tapped-delay
%   line is fed by the network's own prediction a^2(t) = yHat(t) through
%   LW^{1,2}, never by the measured target.
%
%   This is NARXMODEL_ARCHITECTURE redrawn for the adding problem. Three
%   things differ from the network that file documents:
%
%     1. the input delay argument is 0, so there is NO input tapped delay
%        line -- layer 1 sees only p(t), and no earlier p is reachable;
%     2. R^1 = 2, because the input is a value channel and a marker channel,
%        so IW^{1,1} is S^1 x 2 rather than S^1 x D;
%     3. the data is used in its natural units; NARXmodel.Normalize is not
%        applied.
%
%   Sizes follow RUN_ADDING: D = 8 feedback taps and S^1 = 2 neurons.
%
%   Notation follows Martin Hagan (see AGENTS.md): p(t) is the external
%   input, n^i(t) and a^i(t) are the net input and output of layer i, S^i is
%   the number of neurons in layer i, and a^2(t) = yHat(t) is the prediction.
%
%   Writes adding_narx_architecture.png beside this file.

numDelays  = 8;                       % D, output-feedback taps
numNeurons = 2;                       % S^1
numInputs  = 2;                       % R^1: value channel and marker channel
numParams  = numNeurons*numInputs + numNeurons*numDelays + ...
             numNeurons + numNeurons + 1;                          % 25

outputFile = fullfile(fileparts(mfilename('fullpath')), ...
                      'adding_narx_architecture.png');

% The drawing is hand-laid-out, so confirm every size against the network the
% code really builds. If MAKE_ADDING_NARX changes, this errors rather than
% letting the figure quietly go stale.
probe = generate_adding_dataset('SequenceLength', 20, 'NumTrain', 4, ...
    'NumTest', 4, 'Seed', 1);
built = make_adding_narx('Delay', numDelays, 'Neurons', numNeurons, ...
    'Example', probe.train, 'ActivationFcn', "poslin", 'Seed', 1);
assert(built.inputs{1}.size == numInputs, 'R^1 drawn as %d.', numInputs);
assert(isequal(built.inputWeights{1,1}.delays, 0), 'Drawn with no input TDL.');
assert(isequal(size(built.IW{1,1}), [numNeurons numInputs]), 'IW^{1,1} size.');
assert(isequal(size(built.LW{1,2}), [numNeurons numDelays]), 'LW^{1,2} size.');
assert(isequal(size(built.LW{2,1}), [1 numNeurons]), 'LW^{2,1} size.');
assert(isequal(built.layerWeights{1,2}.delays, 1:numDelays), 'Feedback taps.');
assert(built.layers{2}.size == 1, 'S^2 drawn as 1.');
assert(numel(getwb(built)) == numParams, ...
    'Drawn parameter count %d, network has %d.', numParams, numel(getwb(built)));

% ---------------------------------------------------------------- canvas ---
fig = figure('Color','w','Units','pixels','Position',[60 60 1200 659], ...
             'MenuBar','none','ToolBar','none','Visible','off', ...
             'Name','Adding-problem NARX architecture');
ax  = axes('Parent',fig,'Position',[0 0 1 1]);
hold(ax,'on');
axis(ax,[0 169.2 0 92.9]);
daspect(ax,[1 1 1]);
axis(ax,'off');

lw       = 1.6;
fsTitle  = 17;
fsGroup  = 13;
fsMain   = 12;
fsSize   = 10.5;
fsSmall  = 8.5;
fsNote   = 9.5;

% ----------------------------------------------------------------- title ---
placeText(84.6, 88.7, ['\textbf{Closed-loop (parallel) NARX used for the ' ...
                       'adding problem}'], fsTitle);

% ------------------------------------------------- input / layer groupings --
drawBrace(  6.0,  44.0, 77.0, 2.2, lw);  placeText( 25.0, 82.0, 'External input',  fsGroup);
drawBrace( 46.0, 108.0, 77.0, 2.2, lw);  placeText( 77.0, 82.0, 'Layer 1',         fsGroup);
drawBrace(111.0, 158.0, 77.0, 2.2, lw);  placeText(134.5, 82.0, 'Layer 2',         fsGroup);

% ------------------------------------------- external input path, p(t) -----
% No input tapped delay line: narxnet(0, ...) exposes p(t) only.
placeText(10.5, 70.5, '$p(t)$',              fsMain);
placeText(10.5, 66.0, '$2 \times 1$',        fsSize);
placeText(10.5, 61.5, '$R^1 = 2$',           fsSize);

drawArrow(14.8, 65.0, 19.3, 65.0, lw);

drawVectorBracket(21.3, 34.3, 57.6, 72.4, lw);
placeText(27.8, 69.2, '$p_1(t)$  value',     fsMain);
placeText(27.8, 63.8, '$p_2(t)$  marker',    fsMain);
placeText(27.8, 59.4, '$2 \times 1$',        fsSize);

drawArrow(35.8, 65.0, 46.7, 65.0, lw);
placeText(41.0, 69.6, 'no input TDL',        fsSmall);

drawBox(47.0, 61.0, 10.0, 8.0, lw);
placeText(52.0, 65.0, '$\mathbf{IW}^{1,1}$',                                fsMain);
placeText(52.0, 57.8, sprintf('$%d \\times %d$', numNeurons, numInputs),    fsSize);

% ------------------------------------------------------ summing junction ---
sumX = 72.0; sumY = 45.0; sumR = 3.6;
drawSum(sumX, sumY, sumR, lw);

drawArrowToPoint(57.4, 62.2, sumX, sumY, sumR, lw);   % IW^{1,1} -> sum
drawArrowToPoint(57.4, 33.8, sumX, sumY, sumR, lw);   % LW^{1,2} -> sum
drawArrowToPoint(73.5, 31.2, sumX, sumY, sumR, lw);   % b^1      -> sum

% ------------------------------------------- closed-loop feedback path -----
drawBox(16.5, 27.0, 9.0, 8.0, lw);
placeText(21.0, 32.6, '\textbf{TDL}',                  fsMain);
placeText(21.0, 29.1, sprintf('1:%d', numDelays),      fsSize);

drawArrow(25.5, 31.0, 29.8, 31.0, lw);

drawVectorBracket(32.3, 41.7, 23.3, 38.7, lw);
placeText(37.0, 36.0, '$a^2(t-1)$',                            fsMain);
placeText(37.0, 31.0, '$\vdots$',                              fsMain);
placeText(37.0, 26.0, sprintf('$a^2(t-%d)$', numDelays),       fsMain);
placeText(37.0, 19.7, sprintf('$%d \\times 1$', numDelays),    fsSize);

drawArrow(42.5, 31.0, 46.7, 31.0, lw);

drawBox(47.0, 27.0, 10.0, 8.0, lw);
placeText(52.0, 31.0, '$\mathbf{LW}^{1,2}$',                              fsMain);
placeText(52.0, 23.8, sprintf('$%d \\times %d$', numNeurons, numDelays),  fsSize);

% ---------------------------------------------------------- bias, layer 1 --
drawBox(70.0, 22.0, 7.0, 9.0, lw);
placeText(73.5, 28.3, '$\mathbf{b}^1$',                       fsMain);
placeText(73.5, 24.2, sprintf('$%d \\times 1$', numNeurons),  fsSize);

% --------------------------------------------------- activation function 1 ---
placeText(81.5, 52.0, '$\mathbf{n}^1(t)$',                    fsMain);
placeText(81.5, 47.8, sprintf('$%d \\times 1$', numNeurons),  fsSize);

drawArrow(75.6, 45.0, 85.7, 45.0, lw);

drawBox(86.0, 31.0, 14.0, 23.0, lw);
placeText(93.0, 47.0, '$\mathbf{f}^1$',                       fsMain);
placeText(93.0, 41.5, 'poslin (ReLU)',                        fsSmall);
placeText(93.0, 37.8, 'or tansig',                            fsSmall);
placeText(93.0, 27.6, sprintf('$S^1 = %d$', numNeurons),      fsSize);
placeText(93.0, 23.6, sprintf('(%d neurons)', numNeurons),    fsSmall);

placeText(106.0, 52.0, '$\mathbf{a}^1(t)$',                   fsMain);
placeText(106.0, 47.8, sprintf('$%d \\times 1$', numNeurons), fsSize);

drawArrow(100.0, 45.0, 109.7, 45.0, lw);

% --------------------------------------------------------------- layer 2 ---
drawBox(110.0, 41.6, 11.0, 6.8, lw);
placeText(115.5, 45.0, '$\mathbf{LW}^{2,1}$',                 fsMain);
placeText(115.5, 38.4, sprintf('$1 \\times %d$', numNeurons), fsSize);

placeText(125.0, 56.8, '$\mathbf{n}^2(t)$',                   fsMain);
placeText(125.0, 52.8, '$1 \times 1$',                        fsSize);

sum2X = 129.0; sum2Y = 45.0; sum2R = 3.0;
drawSum(sum2X, sum2Y, sum2R, lw);
drawArrowToPoint(121.0, 45.0, sum2X, sum2Y, sum2R, lw);
drawArrowToPoint(129.0, 30.2, sum2X, sum2Y, sum2R, lw);

drawBox(125.7, 21.0, 6.6, 9.0, lw);
placeText(129.0, 27.3, '$\mathbf{b}^2$',                      fsMain);
placeText(129.0, 23.2, '$1 \times 1$',                        fsSize);

drawArrow(132.0, 45.0, 133.7, 45.0, lw);

drawBox(134.0, 31.0, 8.0, 23.0, lw);
placeText(138.0, 47.0, '$\mathbf{f}^2$',                      fsMain);
placeText(138.0, 41.5, 'purelin',                             fsSmall);
placeText(138.0, 27.6, '$S^2 = 1$',                           fsSize);
placeText(138.0, 23.6, '(1 neuron)',                          fsSmall);

drawArrow(142.0, 45.0, 154.5, 45.0, lw);
placeText(157.5, 51.5, '$\mathbf{a}^2(t) = \hat{\mathbf{y}}(t)$', fsMain);
placeText(157.5, 47.3, '$1 \times 1$',                            fsSize);
placeText(157.5, 40.5, 'scored only',                             fsSmall);
placeText(157.5, 37.2, 'at $t = T$',                              fsSmall);

% ------------------------------------------------- the closed feedback -----
tapX = 147.0; loopY = 13.0;
plot([tapX tapX], [45.0 loopY], 'k-', 'LineWidth', lw);
plot([tapX 11.0], [loopY loopY], 'k-', 'LineWidth', lw);
plot([11.0 11.0], [loopY 31.0],  'k-', 'LineWidth', lw);
drawArrow(11.0, 31.0, 16.2, 31.0, lw);
plot(tapX, 45.0, 'k.', 'MarkerSize', 14);

placeText(84.0, 10.2, ['closed loop: the scalar prediction $a^2(t)=\hat{y}(t)$ is the ' ...
                      'only memory, fed back through the tapped-delay line and $LW^{1,2}$'], fsNote);

% --------------------------------------------------------------- caption ---
placeText(84.6, 5.4, sprintf(['\\texttt{closeloop(narxnet(0, 1:%d, %d))}; parallel ' ...
                              'architecture; %d trainable weights and biases; ' ...
                              'natural units, no normalisation.'], ...
                              numDelays, numNeurons, numParams), fsNote);
placeText(84.6, 2.2, ['poslin initialisation: $LW^{1,2}=\frac{1}{\sqrt{2}}' ...
                      '[\,0.5\,\,0\,\ldots\,0\,\,0.5\,]$ on both rows, ' ...
                      '$LW^{2,1}=\frac{1}{\sqrt{2}}[\,1\,\,1\,]$, ' ...
                      '$b^1=b^2=0$; error weights are zero at every time ' ...
                      'except $t=T$.'], fsNote);

exportgraphics(fig, outputFile, 'Resolution', 200);
close(fig);
fprintf('Wrote %s\n', outputFile);

end

% =========================================================== local helpers ==
function h = placeText(x, y, str, fontSize)
h = text(x, y, str, 'Interpreter','latex', 'FontSize', fontSize, ...
         'HorizontalAlignment','center', 'VerticalAlignment','middle', 'Color','k');
end

function drawBox(x, y, w, h, lw)
rectangle('Position',[x y w h], 'EdgeColor','k', 'FaceColor','w', 'LineWidth', lw);
end

function drawSum(cx, cy, r, lw)
rectangle('Position',[cx-r cy-r 2*r 2*r], 'Curvature',[1 1], ...
          'EdgeColor','k', 'FaceColor','w', 'LineWidth', lw);
plot([cx-0.45*r cx+0.45*r], [cy cy], 'k-', 'LineWidth', lw);
plot([cx cx], [cy-0.45*r cy+0.45*r], 'k-', 'LineWidth', lw);
end

function drawArrow(x1, y1, x2, y2, lw)
headLen  = 1.7;
headHalf = 0.75;
d = [x2-x1, y2-y1];
d = d / norm(d);
n = [-d(2), d(1)];
baseX = x2 - headLen*d(1);
baseY = y2 - headLen*d(2);
plot([x1 baseX], [y1 baseY], 'k-', 'LineWidth', lw);
patch('XData',[x2, baseX+headHalf*n(1), baseX-headHalf*n(1)], ...
      'YData',[y2, baseY+headHalf*n(2), baseY-headHalf*n(2)], ...
      'FaceColor','k', 'EdgeColor','k');
end

function drawArrowToPoint(x1, y1, cx, cy, r, lw)
% Arrow from (x1,y1) that stops on the circle of radius r centred at (cx,cy).
d = [cx-x1, cy-y1];
d = d / norm(d);
drawArrow(x1, y1, cx - r*d(1), cy - r*d(2), lw);
end

function drawVectorBracket(xLeft, xRight, yBottom, yTop, lw)
tick = 0.9;
plot([xLeft+tick xLeft xLeft xLeft+tick], [yTop yTop yBottom yBottom], 'k-', 'LineWidth', lw);
plot([xRight-tick xRight xRight xRight-tick], [yTop yTop yBottom yBottom], 'k-', 'LineWidth', lw);
end

function drawBrace(x1, x2, y, r, lw)
% Over-brace spanning [x1 x2] with the flat run at height y.
xm = 0.5*(x1+x2);
a  = linspace(pi, pi/2, 24);
b  = linspace(-pi/2, 0, 24);
c  = linspace(pi, 3*pi/2, 24);
d  = linspace(pi/2, 0, 24);
X = [x1+r + r*cos(a), xm-r + r*cos(b), xm+r + r*cos(c), x2-r + r*cos(d)];
Y = [y-r  + r*sin(a), y+r  + r*sin(b), y+r  + r*sin(c), y-r  + r*sin(d)];
X = [X(1:24), linspace(x1+r, xm-r, 2), X(25:48), X(49:72), linspace(xm+r, x2-r, 2), X(73:end)];
Y = [Y(1:24), y*[1 1],                 Y(25:48), Y(49:72), y*[1 1],                 Y(73:end)];
plot(X, Y, 'k-', 'LineWidth', lw);
end
