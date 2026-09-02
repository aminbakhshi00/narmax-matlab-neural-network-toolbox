classdef NARMAXmodel
    
    properties (Access=public)
        
        % properties
        u_pre;
        y_pre;
        u;
        y;
        delay;
        neurons;
        train_k;
        narmax;
        presetmodel = false;
        
        % normalization
        slope_u;
        int_u;
        slope_y;
        int_y;
        
        % modified training statistics table
        statistics_table;
       
        % user settings
        writeToSave      = false;
        writeToConsole   = true;
        earlyStoppage    = false;
        trainAlg         = 'trainlm';
        iterPerRun       = 1000;
        iterAfterValley  = 400;
        iterAfterSeq     = 300;
        maxStep          = 20;
        initialTraining  = 5;
        fill             = 2;
        zero_input_delay = false;
        
    end
    
    methods (Access=public)
       
        function obj = NARMAXmodel(delay, neurons)
            obj.delay = delay;
            obj.neurons = neurons;
        end
        
        function obj = SetNARMAXmodel(obj, narmax, u_pre, y_pre)
            [obj.narmax] = obj.ConvertOLtoCL(u_pre, y_pre, narmax, obj.delay, obj.neurons); 
            obj.presetmodel = true;
        end
        
        function obj = train(obj, u_pre, y_pre, train_k)
            
            [obj, obj.u_pre, obj.y_pre] = obj.Normalize(u_pre, y_pre);
            
            obj.train_k;
            
            y = con2seq(obj.y_pre');
            u = con2seq(obj.u_pre');
            
            obj.y = y;
            obj.u = u;

            if ~obj.presetmodel
                if obj.zero_input_delay
                    narx = narxnet(0:obj.delay, 1:obj.delay, obj.neurons);
                else
                    narx = narxnet(1:obj.delay, 1:obj.delay, obj.neurons);
                end
                
                [p, Pi, Ai, t] = preparets(narx, u, {}, y);

                if obj.earlyStoppage
                    narx.divideParam.trainRatio=0.70;
                    narx.divideParam.valRatio=0.15;
                    narx.divideParam.testRatio=0.15;
                    narx.trainParam.min_grad=0;
                else
                    narx.divideParam.trainRatio=1;
                    narx.divideParam.valRatio=0;
                    narx.divideParam.testRatio=0;
                    narx.trainParam.min_grad=0;
                end
            end

            if obj.writeToConsole
               disp('Started Training -----> NARMAX MODEL');
            end

            if ~obj.presetmodel
                narx.trainFcn = obj.trainAlg;
                narx.trainParam.epochs = obj.iterPerRun;

                tic
                narx = train(narx, p, t, Pi, Ai, 'useParallel', 'yes');
                time = toc;
                time_train = secs2hms(time);

                if obj.writeToConsole
                   fprintf('   Open Loop NARMAX Training - Time: %s\n', time_train)
                end

                [obj.narmax] = obj.CreateOLNARMAX(narx, obj.u_pre, obj.y_pre, obj.delay, obj.neurons);
                [obj.narmax] = obj.ConvertOLtoCL(obj.u_pre, obj.y_pre, obj.narmax, obj.delay, obj.neurons);
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
            
            Current_Horizon_Step = [];
            Modified_Training_Iterations = [];
            obj.statistics_table = table(Current_Horizon_Step, Modified_Training_Iterations);

            Current_Horizon_Step = [];
            Modified_Training_Iterations = [];
            
            Current_Horizon_Step = [Current_Horizon_Step; Horizon_Step];
            Modified_Training_Iterations = [Modified_Training_Iterations; 0];

            Change = 0; 
            while(1)
                switch Change
                    case 0
                        [obj.narmax, tr, time] = obj.Standard_Training(obj.narmax, obj.u_pre, obj.y_pre, ...
                                                                 obj.neurons, Horizon_Step, ...
                                                                 obj.delay, epochs_ST, show_ST);

                        if obj.writeToSave                             
                            narmax = obj.narmax;
                            save( sprintf('Net_Close_%s', num2str(Horizon_Step)) , 'narmax' , 'tr')
                        end
                        
                        if obj.writeToConsole
                            fprintf('   Standard Training - Horizon Step: %i, Time: %s\n', Horizon_Step, time);
                        end

                        if strcmp(obj.trainAlg, 'trainlm')
                            Mu_Max = obj.narmax.trainParam.mu_max;
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
                        elseif ~strcmp(obj.trainAlg, 'trainlm') || tr.mu(end) < Mu_Max                    
                            [Horizon_Step, ~, epochs_ST] = obj.Horizon_Step_Selection(obj.narmax, Horizon_Step, obj.u_pre, obj.y_pre, ...
                                                                                      obj.neurons, obj.delay, Max_step, ...
                                                                                      Lo_Loc_Min_Iter, Fu_Loc_Min_Iter);
                                                        
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
                        [obj.narmax, tr, time, iter, passed] = obj.Modified_Training(obj.narmax, obj.u_pre, obj.y_pre, ...
                                                                                     obj.neurons, Horizon_Step, obj.delay, ...
                                                                                     iter1, iter2);

                        if obj.writeToSave                             
                            narmax = obj.narmax;
                            save( sprintf('Net_Close_%s', num2str(Horizon_Step)) , 'narmax' , 'tr')
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
                            [Horizon_Step, ~, epochs_ST] = obj.Horizon_Step_Selection(obj.narmax, Horizon_Step, obj.u_pre, obj.y_pre, ...
                                                                                      obj.neurons, obj.delay, Max_step, Lo_Loc_Min_Iter, ...
                                                                                      Fu_Loc_Min_Iter);
                            
                            if Horizon_Step > Prediction_Horizon
                                Horizon_Step = Prediction_Horizon;
                            end
                                                                                  
                            Modified_Training_Iterations(end) = iter;
                            Current_Horizon_Step = [Current_Horizon_Step; Horizon_Step];
                            Modified_Training_Iterations = [Modified_Training_Iterations; 0];
                        end
                        
                        Mu_Max = obj.narmax.trainParam.mu_max;                                                  

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
            
            [obj.narmax] = obj.ConvertCLtoOL(u_pre, y_pre, obj.narmax, obj.delay, obj.neurons);

            K = length(u_pre);
            delay_steps = obj.delay * fill + steps;
            p = floor(K/delay_steps);

            y_tmp = reshape(y_pre(1 : p * delay_steps), delay_steps, p);
            u_tmp = reshape(u_pre(1 : p * delay_steps), delay_steps, p);

            [row, col] = size(y_tmp);
            y = mat2cell(y_tmp, ones(1, row), col)';
            u = mat2cell(u_tmp, ones(1, row), col)';

            u = [u; y];

            % remove all cells after 2 times delay
            y_bshift = y(:, 1:fill * obj.delay);
            u_bshift = u(:, 1:fill * obj.delay);

            [p, Pi, Ai, ~] = preparets(obj.narmax, u_bshift, y_bshift);

            [~,~, Af] = sim(obj.narmax, p, Pi, Ai);

            [obj.narmax] = obj.ConvertOLtoCL(u_pre, y_pre, obj.narmax, obj.delay, obj.neurons);

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

            [p, Pi, Ai, ~] = preparets(obj.narmax, u_fshift, y_fshift);

            Ai(3,:) = Af(3,:);

            y_est = sim(obj.narmax, p, Pi, Ai);

            y = obj.Unnormalize_Y(cell2mat(y_fshift(obj.delay + 1:end))');
            y_est = obj.Unnormalize_Y(cell2mat(y_est)');
            u = obj.Unnormalize_U(cell2mat(u_fshift(obj.delay + 1:end))');
            
        end
        
    end
    
    methods (Access=private)
        
        function [OLnarmax] = CreateOLNARMAX(obj, narx, u_pre, y_pre, delay, neurons)

            y = con2seq(y_pre');
            u = [con2seq(u_pre'); con2seq(y_pre')];

            OLnarmax = feedforwardnet([neurons 1]); 

            OLnarmax.layerConnect = [0 0 1;1 0 0;0 1 0];
            OLnarmax.outputConnect = [0 1 0];
            OLnarmax.numInputs = 2;
            OLnarmax.inputConnect = [1 1; 0 0; 0 1];
            OLnarmax.biasConnect = [1 1 0]';

            OLnarmax.layers{2}.transferFcn = 'purelin';
            OLnarmax.layers{3}.transferFcn = 'purelin';
            OLnarmax.layers{3}.dimensions = 1;

            if obj.zero_input_delay
                OLnarmax.inputWeights{1,1}.delays = [0:delay];
            else
                OLnarmax.inputWeights{1,1}.delays = [1:delay];
            end
            
            OLnarmax.inputWeights{1,2}.delays = [1:delay]; 
            OLnarmax.layerWeights{1,3}.delays = [1:delay]; 

            OLnarmax = configure(OLnarmax, u, y);

            OLnarmax.LW{3,2} = [-1];
            OLnarmax.IW{3,2} = [1];
            OLnarmax.inputWeights{3,2}.learn = 0;
            OLnarmax.layerWeights{3,2}.learn = 0;

            OLnarmax.LW{2,1} = narx.LW{2,1};  
            OLnarmax.IW{1,1} = narx.IW{1,1};
            OLnarmax.IW{1,2} = narx.IW{1,2};
            OLnarmax.b{1} = narx.b{1};
            OLnarmax.b{2} = narx.b{2};
            
            if obj.earlyStoppage
                OLnarmax.divideParam.trainRatio=0.70;
                OLnarmax.divideParam.valRatio=0.15;
                OLnarmax.divideParam.testRatio=0.15;
                OLnarmax.trainParam.min_grad=0;
            else
                OLnarmax.divideParam.trainRatio=1;
                OLnarmax.divideParam.valRatio=0;
                OLnarmax.divideParam.testRatio=0;
                OLnarmax.trainParam.min_grad=0;
            end

        end
        
        function [CLnarmax] = ConvertOLtoCL(obj, u_pre, y_pre, OLnarmax, delay, neurons)

            y = con2seq(y_pre');
            u = [con2seq(u_pre')];

            CLnarmax = feedforwardnet([neurons 1]); 

            CLnarmax.layerConnect = [0 1 1;1 0 0;0 1 0];
            CLnarmax.outputConnect = [0 1 0];
            CLnarmax.numInputs = 1;
            CLnarmax.inputConnect = [1; 0; 0];
            CLnarmax.biasConnect = [1 1 0]';

            CLnarmax.layers{2}.transferFcn = 'purelin';
            CLnarmax.layers{3}.transferFcn = 'purelin';
            CLnarmax.layers{3}.dimensions = 1;

            if obj.zero_input_delay
                CLnarmax.inputWeights{1,1}.delays = [0:delay];
            else
                CLnarmax.inputWeights{1,1}.delays = [1:delay];
            end
            
            CLnarmax.layerWeights{1,2}.delays = [1:delay]; 
            CLnarmax.layerWeights{1,3}.delays = [1:delay]; 

            CLnarmax.output.processFcns = {};
            CLnarmax.input.processFcns = {};

            if obj.earlyStoppage
                CLnarmax.divideParam.trainRatio=0.70;
                CLnarmax.divideParam.valRatio=0.15;
                CLnarmax.divideParam.testRatio=0.15;
                CLnarmax.trainParam.min_grad=0;
            else
                CLnarmax.divideParam.trainRatio=1;
                CLnarmax.divideParam.valRatio=0;
                CLnarmax.divideParam.testRatio=0;
                CLnarmax.trainParam.min_grad=0;
            end

            CLnarmax = configure(CLnarmax, u, y);

            CLnarmax.LW{3,2} = [0];
            CLnarmax.layerWeights{3,2}.learn = 0;

            CLnarmax.LW{2,1} = OLnarmax.LW{2,1};
            CLnarmax.IW{1,1} = OLnarmax.IW{1,1};
            CLnarmax.LW{1,2} = OLnarmax.IW{1,2};
            CLnarmax.LW{1,3} = OLnarmax.LW{1,3};
            CLnarmax.b = OLnarmax.b;

        end
        
        function [OLnarmax] = ConvertCLtoOL(obj, u_pre, y_pre, CLnarmax, delay, neurons)

            y = con2seq(y_pre');
            u = [con2seq(u_pre'); con2seq(y_pre')];

            OLnarmax = feedforwardnet([neurons 1]); 

            OLnarmax.layerConnect = [0 0 1;1 0 0;0 1 0];
            OLnarmax.outputConnect = [0 1 0];
            OLnarmax.numInputs = 2;
            OLnarmax.inputConnect = [1 1; 0 0; 0 1];
            OLnarmax.biasConnect = [1 1 0]';

            OLnarmax.layers{2}.transferFcn = 'purelin';
            OLnarmax.layers{3}.transferFcn = 'purelin';
            OLnarmax.layers{3}.dimensions = 1;

            if obj.zero_input_delay
                OLnarmax.inputWeights{1,1}.delays = [0:delay];
            else
                OLnarmax.inputWeights{1,1}.delays = [1:delay];
            end
            
            OLnarmax.inputWeights{1,2}.delays = [1:delay]; 
            OLnarmax.layerWeights{1,3}.delays = [1:delay]; 

            OLnarmax.output.processFcns = {};
            OLnarmax.inputs{1}.processFcns = {};
            OLnarmax.inputs{2}.processFcns = {};

            if obj.earlyStoppage
                OLnarmax.divideParam.trainRatio=0.70;
                OLnarmax.divideParam.valRatio=0.15;
                OLnarmax.divideParam.testRatio=0.15;
                OLnarmax.trainParam.min_grad=0;
            else
                OLnarmax.divideParam.trainRatio=1;
                OLnarmax.divideParam.valRatio=0;
                OLnarmax.divideParam.testRatio=0;
                OLnarmax.trainParam.min_grad=0;
            end

            OLnarmax = configure(OLnarmax, u, y);

            OLnarmax.LW{3,2} = [-1];
            OLnarmax.IW{3,2} = [1];
            OLnarmax.inputWeights{3,2}.learn = 0;
            OLnarmax.layerWeights{3,2}.learn = 0;

            OLnarmax.LW{2,1} = CLnarmax.LW{2,1};
            OLnarmax.IW{1,1} = CLnarmax.IW{1,1};
            OLnarmax.IW{1,2} = CLnarmax.LW{1,2};
            OLnarmax.LW{1,3} = CLnarmax.LW{1,3};
            OLnarmax.b = CLnarmax.b;


        end

        function [OLnarmax, p, Pi, Af, t] = CreateOLTrainingModel(obj, u_pre, y_pre, OLnarmax, steps, delay)

            K = length(u_pre);
            
            u = reshape(u_pre, K, 1);
            u = obj.prepare_data(steps, u, obj.fill * delay);
            
            y = reshape(y_pre, K, 1);
            y = obj.prepare_data(steps, y, obj.fill * delay);

            u = [u; y];

            % remove all cells after 2 times delay
            y_bshift = y(:, 1:obj.fill * delay);
            u_bshift = u(:, 1:obj.fill * delay);

            [p, Pi, Ai, ~] = preparets(OLnarmax, u_bshift, y_bshift);

            [~,~, Af] = sim(OLnarmax, p, Pi, Ai);

            % remove all cells before delay
            y_fshift = y(:, (obj.fill-1) * delay + 1:end);
            u_fshift = u(:, (obj.fill-1) * delay + 1:end);

            [p, Pi, ~, t] = preparets(OLnarmax, u_fshift, y_fshift);

        end

        function [narmax, p, Pi, Ai, t] = CreateCLTrainingModel(obj, u_pre, y_pre, narmax, steps, delay, Ai)

            K = length(u_pre);

            u = reshape(u_pre, K, 1);
            u = obj.prepare_data(steps, u, obj.fill * delay);
            
            y = reshape(y_pre, K, 1);
            y = obj.prepare_data(steps, y, obj.fill * delay);

            % remove all cells before delay
            y_fshift = y(:, (obj.fill-1) * delay + 1:end);
            u_fshift = u(:, (obj.fill-1) * delay + 1:end);

            [p_CL, Pi_CL, Ai_CL, t_CL] = preparets(narmax, u_fshift, y_fshift);

            p = p_CL; 
            Pi = Pi_CL; 
            Ai_CL(3,:) = Ai(3,:); 
            Ai = Ai_CL; 
            t = t_CL;

        end

        function [Net_Close, tr, time_train, p, Pi, Ai, t] = Standard_Training(obj, Network, p1, t1, neurons, steps, delay, epochs, show)

            if steps > 1
            
                [Network] = obj.ConvertCLtoOL(p1, t1, Network, delay, neurons);
                [Network, ~, ~, Ai, ~] = obj.CreateOLTrainingModel(p1, t1, Network, steps, delay);
                [Network] = obj.ConvertOLtoCL(p1, t1, Network, delay, neurons);
                [Network, p, Pi, Ai, t] = obj.CreateCLTrainingModel(p1, t1, Network, steps, delay, Ai);

                Network.trainFcn = obj.trainAlg;
                Network.trainParam.epochs = epochs;
                Network.trainParam.showWindow = true;
                Network.trainParam.min_grad = 0;

                tic;
                [Net_Close, tr] = train(Network, p, t, Pi, Ai, 'useParallel', 'yes');
                time = toc;
                time_train = secs2hms(time);
            
            else
                
                [Network] = obj.ConvertCLtoOL(p1, t1, Network, delay, neurons);
                [Network, p, Pi, Ai, t] = obj.CreateOLTrainingModel(p1, t1, Network, steps, delay);
                        
                Network.trainFcn = obj.trainAlg;
                Network.trainParam.epochs = epochs;
                Network.trainParam.showWindow = true;
                Network.trainParam.min_grad = 0;

                tic;
                [Net_Open, tr] = train(Network, p, t, Pi, Ai, 'useParallel', 'yes');
                time = toc;
                time_train = secs2hms(time);
                
                [Net_Close] = obj.ConvertOLtoCL(p1, t1, Net_Open, delay, neurons);
                
            end

        end

        function [horizon_Step, perf_n, epochs_ST] = Horizon_Step_Selection(obj, Network, Horizon_Step, p1, t1, neurons, delay, Max_step, Lo_Loc_Min_Iter, Fu_Loc_Min_Iter)

            Prediction_Horizon = length(p1);

            B = Horizon_Step + Max_step;               
            if B > Prediction_Horizon
                B = Prediction_Horizon-delay; 
            end

            for steps = Horizon_Step + 1:B

                [Network] = obj.ConvertCLtoOL(p1, t1, Network, delay, neurons);
                [Network, ~, ~, Ai, ~] = obj.CreateOLTrainingModel(p1, t1, Network, steps, delay);
                [Network] = obj.ConvertOLtoCL(p1, t1, Network, delay, neurons);
                [Network, p, Pi, Ai, t] = obj.CreateCLTrainingModel(p1, t1, Network, steps, delay, Ai);

                [~,perf,~] = netGrad(Network,p,Pi,Ai,t,[],[]);

                perf_n(steps) = perf;
            end

            dd = diff(perf_n);
            ig = [false (dd(1:end-1)<0 & dd(2:end)>0) false];  % patch to correct length
            inx = find(ig==1);

            pjump = zeros(1,length(inx));
            for k = 1: length(inx)
                if perf_n(inx(k)) <= 0.03
                    pjump(k) = inx(k); 
                end; 
            end; 
            
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

        function [Network, tr, time_train, epoch, passed, Xs, Xi, Ai, Ts] = Modified_Training(obj, Network, p1, t1, neurons, steps, delay, iter1, iter2)
              
            [Network] = obj.ConvertCLtoOL(p1, t1, Network, delay, neurons);
            [Network, ~, ~, Ai, ~] = obj.CreateOLTrainingModel(p1, t1, Network, steps, delay);
            [Network] = obj.ConvertOLtoCL(p1, t1, Network, delay, neurons);
            [Network, Xs, Xi, Ai, Ts] = obj.CreateCLTrainingModel(p1, t1, Network, steps, delay, Ai);

            [Network, tr] = train(Network, Xs, Ts, Xi, Ai, 'useParallel', 'yes');

            Network.trainParam.showWindow = false;
            
            passed = true;
            
            first = tr;
            mu_Max = Network.trainParam.mu_max;
            count = 1;  flag = 0;  q = 1;  q1 = 1; q2 = 1;  epoch = 1; RSC = 1; init = 1;
            tic;

            while epoch <= iter1
                if (tr.mu(end) >= mu_Max && count == 1) || init == 1

                    [~,Number_seq]=size(Ts{1});  Nointerval = steps;
                    for l =1:Number_seq   
                        for i= 1:Nointerval;  Xs1{i} = Xs{i}(l);    end; clear i;
                        for i= 1:delay;       Xi1{i} = Xi{i}(l);    end; clear i;
                        for i= 1:Nointerval;  Ts1{i} = Ts{i}(l);    end; clear i;
                        for i= 1:delay ;      Ai1{1,i} = Ai{1,i}(:,l);    Ai1{2,i} = Ai{2,i}(l); Ai1{3,i} = Ai{3,i}(l);  end; clear i;
                        [gWB,~,~] = netGrad(Network, Xs1, Xi1, Ai1, Ts1, [], []);
                        R(l) = norm(gWB);
                    end

                    e{q1} = R ; q1 = q1 + 1;
                    [~,IX] = sort(R,'descend');
                    RS{epoch,count} = IX( RSC);
                    count = count + 1;
                    init = 0;

                else
                    Nointerval = steps;
                    Xs2 = Xs; Xi2 = Xi; Ai2 = Ai; Ts2 = Ts;
                    for i= 1:Nointerval;   Xs2{i}( IX(1:RSC) ) = [];    end; clear i;
                    for i= 1:delay;        Xi2{i}( IX(1:RSC) ) = [];    end; clear i;
                    for i= 1:Nointerval;   Ts2{i}( IX(1:RSC) ) = [];    end; clear i;
                    for i= 1:delay ;       Ai2{1,i}( :,IX(1:RSC) ) = [];    Ai2{2,i}( IX(1:RSC) ) = []; Ai2{3,i}( IX(1:RSC) ) = [];   end; clear i;

                    while(1)

                        Network.trainParam.epochs = 1;
                        [Network, tr] = train(Network, Xs2, Ts2, Xi2, Ai2, [], 'useParallel', 'yes');
                        b{q} = tr(1:end-1); q = q + 1;

                        if tr.mu(end) > mu_Max
                            Xs2 = Xs; Xi2 = Xi; Ai2 = Ai; Ts2 = Ts;
                            RSC = RSC + 1;
                            for i= 1:Nointerval;   Xs2{i}(IX (1:RSC) ) = [];    end; clear i;
                            for i= 1:delay;        Xi2{i}(IX (1:RSC) ) = [];    end; clear i;
                            for i= 1:Nointerval;   Ts2{i}(IX (1:RSC) ) = [];    end; clear i;
                            for i= 1:delay ;       Ai2{1,i}( :,IX(1:RSC) ) = [];    Ai2{2,i}( IX(1:RSC) ) = []; Ai2{3,i}( IX(1:RSC) ) = [];   end; clear i;
                            RS{epoch,count} = IX( RSC);
                            count = count + 1;
                        else
                            Network.trainParam.epochs = iter2;
                            Network.trainParam.showWindow = true;
                            [Network, tr] = train(Network, Xs, Ts, Xi, Ai, [], 'useParallel', 'yes');
                            Network.trainParam.showWindow = false;
                            flag = 1;  a{q2} = tr(1:end-1); q2 = q2 + 1;
                        end

                        if RSC == length(IX); passed = false; break; end;
                        if tr.mu(end) < mu_Max; break; end;
                        if (tr.mu(end) > mu_Max && flag == 1) || RSC == length(IX); count = 1; RSC = 1; epoch = epoch +1;  break; end;

                    end
                    
                end

                Network.trainParam.showWindow = true;

                flag = 0;
                if tr.mu(end) < mu_Max; disp('We are out of valley'); break;  end;

            end
            
            if epoch >= iter1
                passed = false;
            end
            
            time = toc;
            time_train = secs2hms(time);

        end

        function Dc1 = prepare_data(~, interval, YY, numdelay)

            [m,n] = size(YY);
            p = floor((m-numdelay)/interval);

            YY_new = [];
            for i=1:p
                YY_new = [YY_new; YY(1+(i-1)*interval:i*interval+numdelay,:)];
            end

            numseq = p*n;
            new_length = interval+numdelay;
            YY_new = reshape(YY_new,new_length,numseq);
            Dc1 = mat2cell(YY_new,ones(new_length,1))';
            
        end

        function [Dc1, Dc2] = prepare_data_random85(~, interval, UU, YY, numdelay)

            [m,n] = size(YY);
            p = floor((m-numdelay)/interval);

            YY_new = [];
            UU_new = [];
            for i=1:p
                YY_new = [YY_new; YY(1+(i-1)*interval:i*interval+numdelay,:)];
                UU_new = [UU_new; UU(1+(i-1)*interval:i*interval+numdelay,:)];
            end

            numseq = p*n;
            new_length = interval+numdelay;
            YY_new = reshape(YY_new,new_length,numseq);
            UU_new = reshape(UU_new,new_length,numseq);
            r = randperm(size(YY_new, 2), floor(size(YY_new, 2) * 0.85));
            YY_new = YY_new(:, r);
            UU_new = UU_new(:, r);
            Dc2 = mat2cell(YY_new,ones(new_length,1))';
            Dc1 = mat2cell(UU_new,ones(new_length,1))'; 
            
        end

        function [Dc1, Dc2] = prepare_data_with_overlap_random15(~, interval, UU, YY, numdelay)

            [m,n] = size(YY);
            p = floor(m-numdelay-interval);

            YY_new = [];
            UU_new = [];
            for i=1:p
                YY_new = [YY_new; YY(i:i+interval+numdelay-1,:)];
                UU_new = [UU_new; UU(i:i+interval+numdelay-1,:)];
            end

            numseq = p*n;
            new_length = interval+numdelay;
            YY_new = reshape(YY_new,new_length,numseq);
            UU_new = reshape(UU_new,new_length,numseq);
            r = randperm(size(YY_new, 2), floor(size(YY_new, 2) * 0.10));
            YY_new = YY_new(:, r);
            UU_new = UU_new(:, r);
            Dc2 = mat2cell(YY_new,ones(new_length,1))';
            Dc1 = mat2cell(UU_new,ones(new_length,1))'; 
            
        end
        
        function Dc1 = prepare_data_with_overlap(~, interval, YY, numdelay)

            [m,n] = size(YY);
            p = floor(m-numdelay-interval);

            YY_new = [];
            for i=1:p
                YY_new = [YY_new; YY(i:i+interval+numdelay-1,:)];
            end

            numseq = p*n;
            new_length = interval+numdelay;
            YY_new = reshape(YY_new,new_length,numseq);
            Dc1 = mat2cell(YY_new,ones(new_length,1))';
            
        end
        
    end
    
end
