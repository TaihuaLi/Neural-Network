% author: Taihua(Ray) Li
% date: Oct 23, 2016
function [ weights, bias, accuracy, cost ] = MyNetwork(inputs, targets, nodeLayers, numEpochs, batchSize, eta, split, momentum, ActFunc, CostFunc, L2lambda, previousNetwork)
   
   % split training, testing and validation sets
  [train, test, valid] = dividerand(length(inputs), split(1), split(2), split(3));
  
  TrainingInputs = inputs(:, train);
  TestingInputs = inputs(:, test);
  ValidationInputs = inputs(:, valid);
  
  TrainingTargets = targets(:, train);
  TestingTargets = targets(:, test);
  ValidationTargets = targets(:, valid);

  % check if training on a previously trained network
  if previousNetwork == 'None'
      % initialize weigths and biases
      weights = {};
      bias = {};
      weightdelta = {};
      biasdelta = {};

      for layer = 2 : length(nodeLayers)
          weights{layer} = normrnd(0,1/sqrt(length(TrainingInputs)),nodeLayers(layer), nodeLayers(layer-1));
          bias{layer} = normrnd(0,1/sqrt(length(TrainingInputs)),nodeLayers(layer), 1);
      end
  else
      nodeLayers = previousNetwork{0};
      weights = previousNetwork{1};
      bias = previousNetwork{2};
  end
      
  accuracy = {};
  cost = {};

  % batching
  batches = {};
  target_batches = {};
  counter = 1;

  for start_pos = 1 : batchSize : length(TrainingInputs)
      if length(TrainingInputs) - start_pos < batchSize 
          % this avoids index out of range problem
          batches{counter} = TrainingInputs(:, start_pos:end);
          target_batches{counter} = TrainingTargets(:, start_pos:end);
      else
          batches{counter} = TrainingInputs(:, start_pos: start_pos + batchSize - 1);
          target_batches{counter} = TrainingTargets(:, start_pos: start_pos + batchSize - 1);
          counter = counter + 1;
      end 
  end 

  % to store cost for each set
  TrainCost = [];
  TestCost = [];
  ValiCost = [];

  if strcmp(ActFunc, 'relu') == 1
      lastAct = input('ReLu should only be used for the hidden layers. Which activation function do you want to use for the output layer?\n');
  elseif strcmp(ActFunc, 'softmax') == 1
      hidAct = input('Softmax should only be used for the output layer. Which activation function do you want to use for hidden layers?\n');
  end
  
  
  fprintf('\t|\t TRAIN\t\t||\t\tTEST\t ||\tVALIDATION\n');
  fprintf('------------------------------------------------------------------------------------------\n');
  fprintf('Ep\t| Cost |   Corr  |  Acc  ||  Cost |  Corr |  Acc ||  Cost | Corr |  Acc \n');
  fprintf('------------------------------------------------------------------------------------------\n');
  % start epochs
  for epoch = 1 : numEpochs
      
      % minibatch shuffling
      random = randperm(length(batches));
      batch_counter = 1;
      
      for batch = 1 : length(batches)
          z = {};
          activation = {};
          activation{1} = batches{random(batch_counter)};
          delta = {};
          
          for layer = 2 : length(nodeLayers)
              
              z{layer} = bsxfun(@plus,(weights{layer} * activation{layer - 1}), bias{layer});
              if strcmp(ActFunc, 'tanh') == 1
                  activation{layer} = tanh(z{layer}); % calling activation functions below
              elseif strcmp(ActFunc, 'sigmoid') == 1
                  activation{layer} = logsig(z{layer});
              elseif strcmp(ActFunc, 'relu') == 1
                  % only use ReLu in the hidden layers
                  % use user input in the output layer
                  if layer~= length(nodeLayers)
                      activation{layer} = max(0, z{layer});
                  else
                      activation{layer} = act(z{layer}, lastAct);
                  end
              elseif strcmp(ActFunc, 'softmax') == 1
                  % only use softmax in the last layer
                  % use user input in the previous layers
                  if layer ~= length(nodeLayers)
                      activation{layer} = act(z{layer}, hidAct);
                  else
                      activation{layer} = act(z{layer}, ActFunc);
                  end
              end
          end
          
          % normalize the output of tanh() to [0, 1]
          if strcmp(ActFunc, 'tanh') == 1 && size(targets, 1) > 1
              activation{length(nodeLayers)} = (activation{length(nodeLayers)}-min(min(activation{length(nodeLayers)}))) / ...
                                      (max(max(activation{length(nodeLayers)}))-min(min(activation{length(nodeLayers)})));
          end
          
          % calculate sum of square of weights for L2 regularization
          % TotalWeights = 0;
          % for layer=2:length(nodeLayers)
          %     TotalWeights = TotalWeights + sum(sum(weights{layer}.^2));
          % end
          
          % L2 regularization term
          % L2 = L2lambda/(2*length(batches{random(batch_counter)})) * TotalWeights;
          
          % calculate training output error/cost with L2 regularization
          % if strcmp(CostFunc, 'quad') == 1
          %     err = 1/(2*length(batches{random(batch_counter)})) * ...
          %           (0.5*(target_batches{random(batch_counter)} - activation{length(nodeLayers)}).^2) + L2;
          % elseif strcmp(CostFunc, 'cross') == 1
          %     err = -1/length(batches{random(batch_counter)}) .* ...
          %         (target_batches{random(batch_counter)} .* log(activation{length(nodeLayers)}) + ...
          %         (1-target_batches{random(batch_counter)}) .* log(1-activation{length(nodeLayers)}));
          % elseif strcmp(CostFunc, 'log') == 1
          %     err = -log(activation{length(nodeLayers)})/length(batches{random(batch_counter)}) + L2;
          % end
          
          err = (activation{length(nodeLayers)} - target_batches{random(batch_counter)});
          
          % last layer of delta for backprop
          if strcmp(ActFunc, 'tanh') == 1
              prime = 1-tanh(z{length(nodeLayers)}).^2; % see the derivative of activation functions below
              delta{length(nodeLayers)} = err .* prime;
          elseif strcmp(ActFunc, 'relu') == 1 % last layer of ReLu should use the derivative of user input
              if strcmp(lastAct, 'softmax') == 0  
                 prime = dact(z{length(nodeLayers)}, lastAct);
                 delta{length(nodeLayers)} = err .* prime;
              else % if the last layer is softmax
                 delta{length(nodeLayers)} = err .* ones(size(z{length(nodeLayers)}));
              end
          elseif strcmp(ActFunc, 'sigmoid') == 1
              prime = logsig(z{length(nodeLayers)}).*(1-logsig(z{length(nodeLayers)}));
              delta{length(nodeLayers)} = err .* prime;
          elseif strcmp(ActFunc, 'softmax') == 1% delta of last layer if using softmax is constant 1
              delta{length(nodeLayers)} = err.*ones(size(z{length(nodeLayers)}));
          end

          % backpropagation
          for layer = (length(nodeLayers) - 1) : -1 : 2
              if strcmp(ActFunc, 'softmax') == 0
                  delta{layer} = (weights{layer + 1}.' * delta{layer + 1}) .* dact(z{layer}, ActFunc);
              else % if softmax, for hidden layers, the drivative of activation function should be user input
                  delta{layer} = (weights{layer + 1}.' * delta{layer + 1}) .* dact(z{layer}, hidAct);
              end
          end 

          % gradient descent with momentum
          for layer = length(nodeLayers) : -1 : 2
              
              if epoch == 1 && batch == 1
                  weights{layer} = weights{layer} - eta/length(batches{random(batch_counter)}) * delta{layer} * activation{layer - 1}.';
                  bias{layer} = bias{layer} - eta/length(batches{random(batch_counter)}) * sum(delta{layer}, 2);
                  weightdelta{layer} = eta/length(batches{random(batch_counter)}) * delta{layer} * activation{layer - 1}.';
                  biasdelta{layer} = eta/length(batches{random(batch_counter)}) * sum(delta{layer}, 2);
              else
                  weights{layer} = weights{layer} + weightdelta{layer};
                  bias{layer} = bias{layer} + biasdelta{layer};
                  weightdelta{layer} = momentum .* weightdelta{layer} - eta/length(batches{random(batch_counter)}) * delta{layer} * activation{layer - 1}.' ;
                  biasdelta{layer} = momentum .* biasdelta{layer} - eta/length(batches{random(batch_counter)}) * sum(delta{layer}, 2);
              end
            
          end
          
          batch_counter = batch_counter + 1;
          
      end 
      
      % calculate output statistics and display result per epoch
      Trainoutput = {};
      Trainoutput{1} = TrainingInputs;
      Testoutput = {};
      Testoutput{1} = TestingInputs;
      Validoutput = {};
      Validoutput{1} = ValidationInputs;
      for layer = 2 : length(nodeLayers)
          % training set
          z1 = bsxfun(@plus,(weights{layer} * Trainoutput{layer - 1}), bias{layer});
          % testing set
          z2 = bsxfun(@plus,(weights{layer} * Testoutput{layer - 1}), bias{layer});
          % validation set
          z3 = bsxfun(@plus,(weights{layer} * Validoutput{layer - 1}), bias{layer});
          
          if strcmp(ActFunc, 'softmax') == 0
              Trainoutput{layer} = act(z1, ActFunc);
              Testoutput{layer} = act(z2, ActFunc);
              Validoutput{layer} = act(z3, ActFunc);
          elseif strcmp(ActFunc, 'relu') == 1
              if layer ~= length(nodeLayers)
                  Trainoutput{layer} = max(0, z1);
                  Testoutput{layer} = max(0, z2);
                  Validoutput{layer} = max(0, z3);
              else
                  Trainoutput{layer} = act(z1, lastAct);
                  Testoutput{layer} = act(z2, lastAct);
                  Validoutput{layer} = act(z3, lastAct);
              end
          else
              if layer ~= length(nodeLayers)
                  Trainoutput{layer} = act(z1, hidAct);
                  Testoutput{layer} = act(z2, hidAct);
                  Validoutput{layer} = act(z3, hidAct);
              else
                  Trainoutput{layer} = act(z1, ActFunc);
                  Testoutput{layer} = act(z2, ActFunc);
                  Validoutput{layer} = act(z3, ActFunc);
              end
          end    
      end
          
      if size(targets,1) > 1 % multiclass problem
          % training
          [it1,vt1] = max(TrainingTargets);
          [i1,v1] = max(Trainoutput{length(nodeLayers)});
          array1 = vt1 - v1;
          traincorrect = sum(array1(:)==0);
          trainaccuracy = traincorrect/length(TrainingInputs);
          
          % testing
          [it2,vt2] = max(TestingTargets);
          [i2,v2] = max(Testoutput{length(nodeLayers)});
          array2 = vt2 - v2;
          testcorrect = sum(array2(:)==0);
          testaccuracy = testcorrect/length(TestingInputs);
          
          % validation
          [it3,vt3] = max(ValidationTargets);
          [i3,v3] = max(Validoutput{length(nodeLayers)});
          array3 = vt3 - v3;
          validcorrect = sum(array3(:)==0);
          validaccuracy = validcorrect/length(ValidationInputs);
      else
          % training
          array1 = TrainingTargets - round(Trainoutput{length(nodeLayers)});  
          traincorrect = sum(array1(:)==0);
          trainaccuracy = traincorrect/size(TrainingInputs, 2);
          
          % testing
          array2 = TestingTargets - round(Testoutput{length(nodeLayers)});
          testcorrect = sum(array2(:)==0);
          testaccuracy = testcorrect/size(TestingInputs, 2);
          
          % validation
          array3 = ValidationTargets - round(Validoutput{length(nodeLayers)});
          validcorrect = sum(array3(:)==0);
          validaccuracy = validcorrect/size(ValidationInputs, 2);
      end
      
      % L2
      TotalWeights = 0;
      for layer=2:length(nodeLayers)
          TotalWeights = TotalWeights + sum(sum(weights{layer}.^2));
      end
      L2Tr = L2lambda/(2*size(TrainingInputs, 2)) * TotalWeights;
      L2Te = L2lambda/(2*size(TestingInputs, 2)) * TotalWeights;
      L2Va = L2lambda/(2*size(ValidationInputs, 2)) * TotalWeights;
      
      % calculate cost
      if strcmp(CostFunc, 'quad') == 1
          TrCost = 1/(2*size(TrainingInputs, 2)) * sum(sum((0.5*(TrainingTargets - Trainoutput{length(nodeLayers)}).^2)))+L2Tr;
          TeCost = 1/(2*size(TestingInputs, 2)) * sum(sum((0.5*(TestingTargets - Testoutput{length(nodeLayers)}).^2)))+L2Te;
          VaCost = 1/(2*size(ValidationInputs, 2)) * sum(sum((0.5*(ValidationTargets - Validoutput{length(nodeLayers)}).^2)))+L2Va;
      elseif strcmp(CostFunc, 'cross') == 1
          TrCost = -1/length(TrainingInputs) .* ...
                  sum(sum(TrainingTargets .* log(Trainoutput{length(nodeLayers)}+eps) + ...
                  (1-TrainingTargets) .* log(1-Trainoutput{length(nodeLayers)}))+eps)+L2Tr;
          TeCost = -1/length(TestingInputs) .* ...
                  sum(sum(TestingTargets .* log(Testoutput{length(nodeLayers)}+eps) + ...
                  (1-TestingTargets) .* log(1-Testoutput{length(nodeLayers)}))+eps)+L2Te;
          VaCost = -1/length(ValidationInputs) .* ...
                  sum(sum(ValidationTargets .* log(Validoutput{length(nodeLayers)}+eps) + ...
                  (1-ValidationTargets) .* log(1-Validoutput{length(nodeLayers)}))+eps)+L2Va;
      elseif strcmp(CostFunc, 'log') == 1
          TrCost = sum(-log(max(Trainoutput{length(nodeLayers)})+eps)/size(TrainingInputs, 2))+L2Tr;
          TeCost = sum(-log(max(Testoutput{length(nodeLayers)})+eps)/size(TestingInputs, 2))+L2Te;
          VaCost = sum(-log(max(Validoutput{length(nodeLayers)})+eps)/size(ValidationInputs, 2))+L2Va;
      end    
      
      % store Costs for early stoping 
      TrainCost(epoch) = TrCost;
      TestCost(epoch) = TeCost;
      ValiCost(epoch) = VaCost;
      
      fprintf('%d\t| %.3f | %d/%d | %.3f || %.3f | %d/%d | %.3f || %.3f | %d/%d | %.3f\n', ...
              epoch, TrainCost(epoch), traincorrect, size(TrainingInputs, 2), trainaccuracy, ...
              TestCost(epoch), testcorrect, size(TestingInputs, 2), testaccuracy, ...
              ValiCost(epoch), validcorrect, size(ValidationInputs, 2), validaccuracy);
      % training
      accuracy{1}(epoch) = trainaccuracy;
      cost{1}(epoch) = TrainCost(epoch);
      % testing
      accuracy{2}(epoch) = testaccuracy;
      cost{2}(epoch) = TestCost(epoch);
      %validation
      accuracy{3}(epoch) = validaccuracy;
      cost{3}(epoch) = ValiCost(epoch);
          
      % after running at least 75% of total epochs, execute early stopping criteria
      % based on validation set only
      if epoch >= round(numEpochs*0.75)
          if ValiCost(epoch) > ValiCost(epoch-1)
              fprintf('Early Stopping: Validation error increased after %d epochs. \n', round(numEpochs*0.75));
              
              subplot(1, 2, 1);
              plot(accuracy{1}); hold on; plot(accuracy{2}); plot(accuracy{3}); 
              title('Accuracy'); xlabel('epoch'); ylabel('accuracy'); legend('Training', 'Testing', 'Validation'); hold off;
              
              subplot(1, 2, 2);
              plot(cost{1}); hold on; plot(cost{2}); plot(cost{3}); 
              title('Cost'); xlabel('epoch'); ylabel('cost'); legend('Training', 'Testing', 'Validation'); hold off;
              break
          end
      elseif traincorrect == length(TrainingInputs) && testcorrect == length(TestingInputs) && validcorrect == length(ValidationInputs)
          fprintf('Early Stopping: Your model is perfect. No more training is needed. \n');
          
          subplot(1, 2, 1);
          plot(accuracy{1}); hold on; plot(accuracy{2}); plot(accuracy{3}); 
          title('Accuracy'); xlabel('epoch'); ylabel('accuracy'); legend('Training', 'Testing', 'Validation'); hold off;
              
          subplot(1, 2, 2);
          plot(cost{1}); hold on; plot(cost{2}); plot(cost{3}); 
          title('Cost'); xlabel('epoch'); ylabel('cost'); legend('Training', 'Testing', 'Validation'); hold off;
          break
      end
      
      subplot(1, 2, 1);
      plot(accuracy{1}); hold on; plot(accuracy{2}); plot(accuracy{3}); 
      title('Accuracy'); xlabel('epoch'); ylabel('accuracy'); legend('Training', 'Testing', 'Validation'); hold off;
      
      subplot(1, 2, 2);
      plot(cost{1}); hold on; plot(cost{2}); plot(cost{3}); 
      title('Cost'); xlabel('epoch'); ylabel('cost'); legend('Training', 'Testing', 'Validation'); hold off;
  end
end


function f = act(z, type) % activation function
    switch type
        case 'sigmoid'
            f=logsig(z);
        case 'tanh'
            f=tanh(z);
        case 'relu'
            f=max(0,z);
        case 'softmax'
            f=softmax(z);
    end
end


function df = dact(z, type) % derivative of activation function
    switch type
        case 'sigmoid'
            df=act(z,type).*(1-act(z,type)); 
        case 'tanh'
            df=1-act(z,type).^2;
        case 'relu'
            df=double(z>0);
        case 'softmax' % use only if not in the last layer
            df=act(z,type).*(1-act(z,type)); 
    end
end

       
