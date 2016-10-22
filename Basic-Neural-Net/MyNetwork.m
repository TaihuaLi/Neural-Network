% author: Taihua(Ray) Li
% date: Oct 9, 2016
function [ weight, bias ] = MyNetwork(inputs, targets, nodeLayers, numEpochs, batchSize, eta)
  % initialize weigths and biases
  weights = {};
  bias = {};

  for layer = 2 : length(nodeLayers)
      weights{layer} = normrnd(0,1,nodeLayers(layer), nodeLayers(layer-1));
      bias{layer} = normrnd(0,1,nodeLayers(layer), 1);

  end
  
  % batching
  batches = {};
  target_batches = {};
  counter = 1;

  for start_pos = 1 : batchSize : length(inputs)
      if length(inputs) - start_pos < batchSize 
          % this avoids index out of range problem
          mb = inputs(:, start_pos:end);
          batches{counter} = mb;
          target = targets(:, start_pos:end);
          target_batches{counter} = target;
      else
          mb = inputs(:, start_pos: start_pos + batchSize - 1);
          batches{counter} = mb;
          target = targets(:, start_pos: start_pos + batchSize - 1);
          target_batches{counter} = target;
          counter = counter + 1;
      end 
  end 

  % start epochs
  for epoch = 1 : numEpochs
      for batch = 1 : length(batches)
          z = {};
          activation = {};
          activation{1} = batches{batch};
          delta = {};

          for layer = 2 : length(nodeLayers)
              z{layer} = bsxfun(@plus,(weights{layer} * activation{layer - 1}), bias{layer});
              activation{layer} = logsig(z{layer});
          end
          
          % calculate output error aka cost
          err = (activation{length(nodeLayers)} - target_batches{batch});
          sigprime = logsig(z{length(nodeLayers)}) .* (1 - logsig(z{length(nodeLayers)}));
          delta{length(nodeLayers)} = err .* sigprime;
          
          % backpropagation
          for layer = (length(nodeLayers) - 1) : -1 : 2
              delta{layer} = (weights{layer + 1}.' * delta{layer + 1}) .* (logsig(z{layer}) .* (1 - logsig(z{layer})));
          end
            
          % gradient descent
          for layer = length(nodeLayers) : -1 : 2
              w = weights{layer} - eta/length(batches{batch}) * delta{layer} * activation{layer - 1}.';
              weights{layer} = w;
              b = bias{layer} - eta/length(batches{batch}) * sum(delta{layer}, 2);
              bias{layer} = b;
          end

      end 
      
      % calculate output statistics and display a message for each epoch
      output = {};
      output{1} = inputs;
      for layer = 2 : length(nodeLayers)
          z = bsxfun(@plus,(weights{layer} * output{layer - 1}), bias{layer});
          output{layer} = logsig(z);
      end
      
      % calculate MSE, number of correct cases and accuracy rate
      MSE = 1/(2*(length(inputs))) * sum(sum((.5 * (targets - output{length(nodeLayers)}).^2)));
      
      if size(targets,1) > 1
          [it,vt] = max(targets);
          [i,v] = max(output{length(nodeLayers)});
          confusion_array = vt - v;
          correct = sum(confusion_array(:)==0);
          accuracy = correct/length(inputs);
      else
          confusion_array = target - round(output{length(nodeLayers)});
          correct = sum(confusion_array(:)==0);
          accuracy = correct/length(inputs);
      end
      
      if numEpochs <= 100 % print message for each iteration if running a small number of epochs
        fprintf('Epoch %d, MSE: %f, Correct: %d/%d, Acc: %f \n', epoch, MSE, correct, length(targets), accuracy);
    elseif mod(epoch, 100) == 0 && numEpochs > 100
        % print only hundredth iteration's message if running a big number of epochs
        fprintf('Epoch %d, MSE: %f, Correct: %d/%d, Acc: %f \n', epoch, MSE, correct, length(targets), accuracy);
    end
    
    % another stopping criterion: accuracy = 1
    if correct == length(targets)
        fprintf('Epoch %d, MSE: %f, Correct: %d/%d, Acc: %f \n', epoch, MSE, correct, length(targets), accuracy);
        break
    end
  end
  
  if mod(numEpochs, 100) ~= 0 && numEpochs > 100
    % print the result for the last iteration for large
    fprintf('Epoch %d, MSE: %f, Correct: %d/%d, Acc: %f \n', epoch, MSE, correct, length(targets), accuracy);
  end

end
