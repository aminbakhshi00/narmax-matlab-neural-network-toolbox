classdef NARXmodel
    
    properties (Access=public)
        
        % properties
        u_pre;
        y_pre;
        u;
        y;
        Ew;
        delay;
        neurons;
        train_k;
        narx;
        
        % normalization
        slope_u;
        int_u;
        slope_y;
        int_y;
        
        % modified training statistics table
        statistics_table;
       
        % user settings
        writeToSave = false;
        writeToConsole = true;
        earlyStoppage = false;
        trainAlg = 'trainlm';
        iterPerRun = 1000;
        iterAfterValley = 400;
        iterAfterSeq = 300;
        maxStep = 20;
        initialTraining = 5;
        zero_input_delay = false;
        hiddenTransferFcn = 'tansig';
        useModifiedTraining = true;
        sequenceSeed = 0;
        segmentStart = 1;
        
    end
    
    methods (Access=public)
       
        function obj = NARXmodel(delay, neurons)
            obj.delay = delay;
            obj.neurons = neurons;
        end
        
        function obj = train(obj, u_pre, y_pre, train_k)
            
            [obj, obj.u_pre, obj.y_pre] = obj.Normalize(u_pre, y_pre);
            
            y = con2seq(obj.y_pre');
            u = con2seq(obj.u_pre');
            Ew = con2seq(ones(1, length(u)));
            
            OLdata.p1 = y;
            OLdata.t1 = u;
            OLdata.Ew = Ew;
            
            obj.y = y;
            obj.u = u;
            obj.Ew = Ew;

            if obj.zero_input_delay
                obj.narx = narxnet(0:obj.delay, 1:obj.delay, obj.neurons);
            else
                obj.narx = narxnet(1:obj.delay, 1:obj.delay, obj.neurons);
            end
            if ~isempty(obj.neurons)
                % With no hidden layer there is no hidden transfer function
                % to set: layer 1 is then the linear output layer itself.
                obj.narx.layers{1}.transferFcn = obj.hiddenTransferFcn;
            end
            
            obj.narx.trainParam.min_grad=0;
            
            % No early stopping means no split at all: every sequence trains.
            obj.narx.divideFcn = '';
            if obj.earlyStoppage
                % divideblock by sequence, never dividerand: consecutive
                % sequences overlap by most of their samples, so a random
                % split would leak near-duplicates into the validation set.
                % Setting divideFcn resets divideParam to that function's
                % defaults, so the ratios must be set after it, not before.
                % Nothing reads the inner test block. Handing its 15% to
                % training instead was tried at 24 taps over three seeds and
                % landed inside the seed scatter, so it stays here. The ARX
                % sidesteps the question by switching early stopping off.
                obj.narx.divideFcn = 'divideblock';
                obj.narx.divideMode = 'sample';
                obj.narx.divideParam.trainRatio = 0.70;
                obj.narx.divideParam.valRatio = 0.15;
                obj.narx.divideParam.testRatio = 0.15;
            end

            obj.narx = closeloop(obj.narx);
            obj.narx.output.processFcns = {};
            obj.narx.input.processFcns = {};
            obj.narx.trainFcn = obj.trainAlg;

            if strcmp(obj.hiddenTransferFcn, 'poslin')
                obj.narx = obj.He_Initialize(obj.narx, u, y);
            end

            if obj.writeToConsole
               disp('Started Training -----> NARX MODEL');
            end

            Prediction_Horizon = train_k;                 % Length of the orginal Sequence
            Horizon_Step = 1;                             % The increment of prediction Horizon
            
            epochs_ST = obj.iterPerRun;                   % Number of epochs
            Fu_Loc_Min_Iter = epochs_ST;
            Lo_Loc_Min_Iter = 200;
            show_ST = 200;                                % Number of timimes shows in command line

            iter1 = obj.iterAfterValley;                  % Number of epochs for passing the valleys
            iter2 = obj.iterAfterSeq;                     % Number of epochs after the seq is back
            Max_step = obj.maxStep;                       % Maximum Prediction Horizon
            Intial_Training = obj.initialTraining;
            counter = Intial_Training;
            reset = true;
            
            Current_Horizon_Step = [];
            Modified_Training_Iterations = [];
            
            Current_Horizon_Step = [Current_Horizon_Step; Horizon_Step];
            Modified_Training_Iterations = [Modified_Training_Iterations; 0];

            Change = 0; 
            while(1)
                switch Change
                    case 0
                        [obj.narx, tr, time] = obj.Standard_Training(obj.narx, u, y, Ew, epochs_ST, obj.delay, Horizon_Step);

                        if obj.writeToSave                             
                            narx = obj.narx;
                            save( sprintf('Net_Close_%s', num2str(Horizon_Step)) , 'narx' , 'tr')
                        end
                        
                        if obj.writeToConsole
                            fprintf('   Standard Training - Horizon Step: %i, Time: %s\n', Horizon_Step, time);
                        end

                        if strcmp(obj.trainAlg, 'trainlm')
                            Mu_Max = obj.narx.trainParam.mu_max;
                        else
                            Mu_Max = 1;
                        end

                        if  Horizon_Step >= Prediction_Horizon
                            if obj.writeToConsole
                                disp('End of Training -----> Full Prediction Horizon Reached');
                            end

                            break; 
                        end

                        if Horizon_Step <= Intial_Training
                            Horizon_Step = Horizon_Step + 1;
                            Change=0;
                            Current_Horizon_Step = [Current_Horizon_Step; Horizon_Step];
                            Modified_Training_Iterations = [Modified_Training_Iterations; 0];
                        elseif ~obj.useModifiedTraining || ~strcmp(obj.trainAlg, 'trainlm') || tr.mu(end) < Mu_Max               
                            [Horizon_Step, ~, epochs_ST] = obj.Horizon_Step_Selection(obj.narx, Horizon_Step, u, y, Ew, obj.delay, ...
                                                                                      Max_step, Lo_Loc_Min_Iter, Fu_Loc_Min_Iter);
                            
                            if Horizon_Step > Prediction_Horizon
                                Horizon_Step = Prediction_Horizon;
                            end
                                                                                 
                            Change = 0;
                            counter = counter + 1;
                            Current_Horizon_Step = [Current_Horizon_Step; Horizon_Step];
                            Modified_Training_Iterations = [Modified_Training_Iterations; 0];
                        elseif tr.mu(end) > Mu_Max
                            Change = 1;
                        end

                    case 1
                        [obj.narx, tr, time, iter, passed] = obj.Modified_Training(obj.narx, u, y, Ew, Horizon_Step, obj.delay, ...
                                                                                   iter1, iter2);

                        if obj.writeToSave                             
                            narx = obj.narx;
                            save( sprintf('Net_Close_%s', num2str(Horizon_Step)) , 'narx' , 'tr')
                        end   
                        
                        if obj.writeToConsole
                            fprintf('   Modified Training - Horizon Step: %i, Epochs: %i, Time: %s\n', Horizon_Step, iter, time)
                        end
                                                             
                        if  Horizon_Step >= Prediction_Horizon
                            if obj.writeToConsole
                                disp('End of Training----->Full Prediction Horizon Reached');
                            end
                            
                            Modified_Training_Iterations(end) = iter;
                            break;
                        end

                        if passed == false
                            Horizon_Step = Horizon_Step - 2;
                            if Horizon_Step <= 0; Horizon_Step = 1; end
                        else
                            [Horizon_Step, ~, epochs_ST] = obj.Horizon_Step_Selection(obj.narx, Horizon_Step, u, y, Ew, obj.delay, ...
                                                                                      Max_step, Lo_Loc_Min_Iter, Fu_Loc_Min_Iter);
                            
                            if Horizon_Step > Prediction_Horizon
                                Horizon_Step = Prediction_Horizon;
                            end
                                                                                  
                            Modified_Training_Iterations(end) = iter;
                            Current_Horizon_Step = [Current_Horizon_Step; Horizon_Step];
                            Modified_Training_Iterations = [Modified_Training_Iterations; 0];
                        end
                        
                        Mu_Max = obj.narx.trainParam.mu_max;                                                  

                        if tr.mu(end) < Mu_Max
                            Change = 0;
                        elseif tr.mu(end) > Mu_Max
                            Change = 1;
                        end

                end
            end
            
            obj.statistics_table = table(Current_Horizon_Step, Modified_Training_Iterations);
            
        end

        function [obj, u_norm, y_norm] = Normalize(obj, u, y)
            
            if isempty(obj.slope_u) || isempty(obj.slope_y) || ...
               isempty(obj.int_u) || isempty(obj.int_y)
                obj.slope_u = max(u) - min(u);
                obj.int_u = min(u);

                obj.slope_y = max(y) - min(y);
                obj.int_y = min(y);
            end
            
            u_norm = (u - obj.int_u) / obj.slope_u;
            y_norm = (y - obj.int_y) / obj.slope_y;
            
        end
        
        function [u_unnorm, y_unnorm] = Unnormalize(obj, u, y)
           
            u_unnorm = obj.slope_u * u + obj.int_u;
            y_unnorm = obj.slope_y * y + obj.int_y;
            
        end
        
        function [u_unnorm] = Unnormalize_U(obj, u)
            
            u_unnorm = obj.slope_u * u + obj.int_u;
            
        end
        
        function [y_unnorm] = Unnormalize_Y(obj, y)
    
            y_unnorm = obj.slope_y * y + obj.int_y;
            
        end

        function [y_est, y] = Sim(obj, u_pre, y_pre, fill)
           
            if nargin < 4
                fill = 2;
            end
            
            steps = length(u_pre) - obj.delay * fill;
            
            [~, u_pre, y_pre] = obj.Normalize(u_pre, y_pre);
            
            K = length(u_pre);
            delay_steps = obj.delay * fill + steps;
            p = floor(K/delay_steps);

            y_tmp = reshape(y_pre(1 : p * delay_steps), delay_steps, p);
            u_tmp = reshape(u_pre(1 : p * delay_steps), delay_steps, p);

            [row, col] = size(y_tmp);
            y = mat2cell(y_tmp, ones(1, row), col)';
            u = mat2cell(u_tmp, ones(1, row), col)';

            y_fshift = y(:, (fill-1) * obj.delay + 1:end);
            u_fshift = u(:, (fill-1) * obj.delay + 1:end);

            [p, Pi, Ai, t] = preparets(obj.narx, u_fshift, {}, y_fshift);
            
            y_est = sim(obj.narx, p, Pi, Ai);
            
            y = obj.Unnormalize_Y(cell2mat(t)');
            y_est = obj.Unnormalize_Y(cell2mat(y_est)');
            
        end
        
    end
    
    methods (Access=private)
        
        function Network = He_Initialize(~, Network, p1, t1)
        %HE_INITIALIZE Draw layer 1 from N(0, 2/fanIn), the He scheme for ReLU.
        %
        %   poslin zeroes every negative net input, which halves the variance
        %   a layer passes on, so its weights are drawn with twice the
        %   variance of a symmetric transfer function's: the factor 2 in
        %   2/fanIn is exactly that compensation. fanIn is the width of
        %   layer 1's tapped delay lines together.
        %
        %   The biases do not start at zero, because He's scheme assumes a
        %   centred input and NORMALIZE does not give one: it maps both
        %   series to [0,1], and the process functions are off, so every
        %   element of layer 1's input is non-negative. A unit's net input
        %   is then w*x with x roughly constant and positive, so its sign is
        %   essentially the sign of sum(w) -- the unit is either active for
        %   almost the whole record or for none of it, and half the layer
        %   starts dead. TRAINLM's first steps kill the rest: with every
        %   unit off the Jacobian is zero, no step can lower the error, mu
        %   climbs to mu_max on the first epoch and every later call ends
        %   the same way, which strands the horizon curriculum.
        %
        %   Each unit is therefore biased by one standard deviation of its
        %   own net input at the mean sample, DEVIATION*norm(meanInput), so
        %   about five units in six start on the active side. That quantity
        %   is sqrt(mean(p1)^2 + mean(t1)^2) whatever DELAY is, so the same
        %   fraction starts active at every tapped-delay length.
        %
        %   Sizing the weights needs a configured network, so the net is
        %   configured on the training sequence first. TRAIN leaves the
        %   weights of an already configured network alone, so these are the
        %   values training starts from.

            [X, ~, ~, T] = preparets(Network, p1, {}, t1);
            Network = configure(Network, X, T);

            fanIn = size(Network.IW{1,1}, 2) + size(Network.LW{1,2}, 2);
            deviation = sqrt(2 / fanIn);
            Network.IW{1,1} = deviation * randn(size(Network.IW{1,1}));
            Network.LW{1,2} = deviation * randn(size(Network.LW{1,2}));

            % The mean sample as layer 1 sees it: the input taps hold p1 and
            % the feedback taps hold t1, so each block takes its own mean.
            meanInput = [mean(cell2mat(p1)) * ones(size(Network.IW{1,1}, 2), 1)
                         mean(cell2mat(t1)) * ones(size(Network.LW{1,2}, 2), 1)];
            Network.b{1} = deviation * norm(meanInput) * ones(size(Network.b{1}));
            disp('   Layer 1 initialized with He scheme for poslin, biases set to one std dev of net input at mean sample and the values are fixed for the rest of training bias is: ' + num2str(Network.b{1}) + ' and weights are: ' + num2str(Network.IW{1,1}) + ' and ' + num2str(Network.LW{1,2}));
        end
        
        function [Net_Close, tr, time_train, Xs, Xi, Ai, Ts] = Standard_Training(obj, Network, p1, t1, Ew, epochs, delay, Horizon_Step)

            Prediction_Horizon = length(p1);

            p1 = cell2mat(p1);
            p1 = reshape(p1, Prediction_Horizon, 1);
            p1 = obj.prepare_data(Horizon_Step, p1, delay);

            t1 = cell2mat(t1);
            t1 = reshape(t1, Prediction_Horizon, 1);
            t1 = obj.prepare_data(Horizon_Step, t1, delay);

            E1 = cell2mat(Ew);
            E1 = reshape(E1, Prediction_Horizon, 1);
            Ew = obj.prepare_data(Horizon_Step, E1, delay);
            
            [Xs, Xi, Ai, Ts, Ews] = preparets(Network, p1, {}, t1, Ew);
            Network.trainFcn = obj.trainAlg;
            Network.trainParam.epochs = epochs;
            Network.trainParam.showWindow = true;
            Network.trainParam.min_grad = 0;

            tic;
            [Net_Close, tr] = train(Network, Xs, Ts, Xi, Ai, Ews, 'useParallel', 'yes');
            time = toc;
            time_train = secs2hms(time);

        end
        
        function [horizon_Step, perf_n, epochs_ST] = Horizon_Step_Selection(obj, Network, Horizon_Step, p1, t1, Ew, delay, Max_step, Lo_Loc_Min_Iter, Fu_Loc_Min_Iter)

            Prediction_Horizon = length(p1);

            B = Horizon_Step + Max_step;               
            if B > Prediction_Horizon
                B = Prediction_Horizon-delay; 
            end

            p1_tmp = p1; t1_tmp = t1; Ew_tmp = Ew;
            for S = Horizon_Step + 1:B
                
                p1 = p1_tmp; t1 = t1_tmp; Ew = Ew_tmp;
                
                p1 = cell2mat(p1);
                p1 = reshape(p1, Prediction_Horizon, 1);
                p1 = obj.prepare_data(S, p1, delay);

                t1 = cell2mat(t1);
                t1 = reshape(t1, Prediction_Horizon, 1);
                t1 = obj.prepare_data(S, t1, delay);

                E1 = cell2mat(Ew);
                E1 = reshape(E1, Prediction_Horizon, 1);
                Ew = obj.prepare_data(S, E1, delay);

                [Xs, Xi, Ai, Ts, ~] = preparets(Network, p1, {}, t1, Ew);
                [~, perf, ~] = netGrad(Network, Xs, Xi, Ai, Ts, [], []);

                perf_n(S) = perf;
                
            end

            dd = diff(perf_n);
            ig = [false (dd(1:end-1)<0 & dd(2:end)>0) false];  % patch to correct length
            inx = find(ig==1);

            pjump = zeros(1,length(inx));
            for k = 1: length(inx)
                if perf_n(inx(k)) <= 0.03
                    pjump(k) = inx(k); 
                end
            end
            
            pjump(~pjump) = [];
            lowest_local_min = {}; furthest_local_min={}; Maximum_Horizon={}; Maximum_Allowed={};
            if isempty(pjump)==0
                furthest_local_min = pjump(end);
            end

            if isempty(inx) == 0
                [~, inxx] = sort(perf_n(inx));
                lowest_local_min = inx(inxx(1)); 
            else
                aa = perf_n < 0.03;
                if all(aa) == 1
                    Maximum_Horizon = length(perf_n);
                else
                    aa1 = find(aa == 0);
                    Maximum_Allowed = aa1(1) ;
                end

            end

            if ~isempty(furthest_local_min) == 1 && ~isempty(lowest_local_min) == 1 
                horizon_Step = furthest_local_min;
                epochs_ST = Fu_Loc_Min_Iter;

            elseif ~isempty(lowest_local_min) == 1    
                horizon_Step = lowest_local_min;
                epochs_ST = Lo_Loc_Min_Iter;

            elseif ~isempty(furthest_local_min) == 1    
                horizon_Step = furthest_local_min;

            elseif ~isempty(Maximum_Allowed) == 1    
                horizon_Step = Maximum_Allowed;
                epochs_ST = Fu_Loc_Min_Iter;

            elseif ~isempty(Maximum_Horizon) == 1    
                horizon_Step = Maximum_Horizon;
                epochs_ST = Fu_Loc_Min_Iter;
            end

        end
        
        function [Net_Close, tr, time_train, epoch, passed, Xs, Xi, Ai, Ts] = Modified_Training(obj, Net_Close, p1, t1, Ew, Horizon_Step, delay, iter1, iter2)
              
            Prediction_Horizon = length(p1);

            p1 = cell2mat(p1);
            p1 = reshape(p1, Prediction_Horizon, 1);
            p1 = obj.prepare_data(Horizon_Step, p1, delay);

            t1 = cell2mat(t1);
            t1 = reshape(t1, Prediction_Horizon, 1);
            t1 = obj.prepare_data(Horizon_Step, t1, delay);

            E1 = cell2mat(Ew);
            E1 = reshape(E1, Prediction_Horizon, 1);
            Ew = obj.prepare_data(Horizon_Step, E1, delay);

            [Xs, Xi, Ai, Ts, Ews] = preparets(Net_Close, p1, {}, t1, Ew);
            [Net_Close, tr] = train(Net_Close, Xs, Ts, Xi, Ai, Ews, 'useParallel', 'yes');
            
            Net_Close.trainParam.showWindow = false;
            
            passed = true;
            
            mu_Max = Net_Close.trainParam.mu_max;
            count = 1;  flag = 0;  q = 1;  q1 = 1; q2 = 1;  epoch = 1; RSC = 1; init = 1;
            tic;

            while epoch <= iter1
                if (tr.mu(end) >= mu_Max && count == 1) || init == 1

                    [~,Number_seq]=size(Ts{1});  Nointerval = Horizon_Step;
                    for l =1:Number_seq   
                        for i= 1:Nointerval;  Xs1{i} = Xs{i}(l);    end; clear i;
                        for i= 1:delay;       Xi1{i} = Xi{i}(l);    end; clear i;
                        for i= 1:Nointerval;  Ts1{i} = Ts{i}(l);    end; clear i;
                        for i= 1:delay ;      Ai1{1,i} = Ai{1,i}(:,l);    Ai1{2,i} = Ai{2,i}(l);   end; clear i;
                        [gWB,~,~] = netGrad(Net_Close, Xs1, Xi1, Ai1, Ts1, [], []);
                        R(l) = norm(gWB);
                    end

                    e{q1} = R ; q1 = q1 + 1;
                    [~,IX] = sort(R,'descend');
                    RS{epoch,count} = IX( RSC);
                    count = count + 1;
                    init = 0;

                else
                    Nointerval = Horizon_Step;
                    Xs2 = Xs; Xi2 = Xi; Ai2 = Ai; Ts2 = Ts;
                    for i= 1:Nointerval;   Xs2{i}( IX(1:RSC) ) = [];    end; clear i;
                    for i= 1:delay;        Xi2{i}( IX(1:RSC) ) = [];    end; clear i;
                    for i= 1:Nointerval;   Ts2{i}( IX(1:RSC) ) = [];    end; clear i;
                    for i= 1:delay ;       Ai2{1,i}( :,IX(1:RSC) ) = [];    Ai2{2,i}( IX(1:RSC) ) = [];   end; clear i;

                    while(1)

                        Net_Close.trainParam.epochs = 1;
                        [Net_Close, tr] = train(Net_Close, Xs2, Ts2, Xi2, Ai2, [], 'useParallel', 'yes');
                        b{q} = tr(1:end-1); q = q + 1;

                        if tr.mu(end) > mu_Max
                            Xs2 = Xs; Xi2 = Xi; Ai2 = Ai; Ts2 = Ts;
                            RSC = RSC + 1;
                            for i= 1:Nointerval;   Xs2{i}(IX (1:RSC) ) = [];    end; clear i;
                            for i= 1:delay;        Xi2{i}(IX (1:RSC) ) = [];    end; clear i;
                            for i= 1:Nointerval;   Ts2{i}(IX (1:RSC) ) = [];    end; clear i;
                            for i= 1:delay ;       Ai2{1,i}( :,IX(1:RSC) ) = [];    Ai2{2,i}( IX(1:RSC) ) = [];   end; clear i;
                            RS{epoch,count} = IX( RSC);
                            count = count + 1;
                            %disp('Another Seq Is Removed')
                        else
                            %disp ('Put the Orginal Seq Back')
                            Net_Close.trainParam.epochs = iter2;
                            Net_Close.trainParam.showWindow = true;
                            [Net_Close, tr] = train(Net_Close, Xs, Ts, Xi, Ai, [], 'useParallel', 'yes');
                            Net_Close.trainParam.showWindow = false;
                            
                            flag = 1;  a{q2} = tr(1:end-1); q2 = q2 + 1;
                        end

                        if RSC == length(IX); passed = false; break; end
                        if tr.mu(end) < mu_Max; break; end
                        if (tr.mu(end) > mu_Max && flag == 1) || RSC == length(IX); count = 1; RSC = 1; epoch = epoch + 1;  break; end

                    end
                    
                end

                flag = 0;
                if tr.mu(end) < mu_Max; disp('We are out of valley'); break;  end

            end
            
            if epoch >= iter1
                passed = false;
            end
            
            time = toc;
            time_train = secs2hms(time);

        end
       
        function Dc1 = prepare_data(obj, interval, YY, numdelay)
        %PREPARE_DATA Cut the series into INTERVAL+NUMDELAY sample sequences.
        %
        %   Each sequence is one training example. Its first NUMDELAY samples
        %   are the warm-up that fills the tapped delay lines -- PREPARETS
        %   consumes exactly those -- and the remaining INTERVAL samples are
        %   the multi-step targets. A 24-tap network therefore spends the
        %   first 24 samples of every sequence loading its delay lines and is
        %   scored on the INTERVAL samples that follow.
        %
        %   The origins are drawn at random, which is the "randomly
        %   overlapping" scheme of Kelley (2024). The two schemes it replaces
        %   both had a defect:
        %
        %     stride 1          the window slid one sample, so consecutive
        %                       sequences were nearly identical and Kelley
        %                       found the redundancy bought nothing.
        %     stride = INTERVAL  the target halves tiled the series, so every
        %                       origin landed INTERVAL apart -- and once
        %                       INTERVAL reaches the period of the data they
        %                       all fall at the same point of the cycle.
        %
        %   As many origins are drawn as the tiling would have given, so the
        %   redundancy of stride 1 is avoided at the same cost in sequences,
        %   but they are placed uniformly instead of on a grid.

            [m,n] = size(YY);
            new_length = interval+numdelay;
            lastStart = m - new_length + 1;            % last origin that fits
            p = floor((m - new_length)/interval) + 1;  % as many as a tiling

            % TRAIN cuts the inputs, the targets and the error weights with
            % three separate calls to this function, and all three have to
            % land on the same origins or the targets stop matching their
            % inputs. Seeding a private stream from the window geometry gives
            % that without carrying any state between the calls, and yields a
            % fresh draw whenever the horizon changes. SEQUENCESEED moves the
            % whole fit onto a different draw.
            stream = RandStream('twister', ...
                'Seed', obj.sequenceSeed + 1000*interval + numdelay);
            starts = sort(randperm(stream, lastStart, min(p, lastStart)));

            index = (1:new_length)' + (starts - 1);

            % A split built from seasons of different years has a join in it.
            % Drop the sequences that straddle one: their delay line would
            % span months of missing record.
            joins = obj.segmentStart(obj.segmentStart > 1);
            for join = joins(:)'
                index(:, index(1,:) < join & index(end,:) >= join) = [];
            end
            p = size(index, 2);
            YY_new = zeros(new_length, p*n);
            for column = 1:n
                series = YY(:,column);
                YY_new(:, (column-1)*p + (1:p)) = series(index);
            end

            Dc1 = mat2cell(YY_new,ones(new_length,1))';
            
        end
        
        
    end
    
end
