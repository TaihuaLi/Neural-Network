% author: Taihua(Ray) Li
% date: Oct 9, 2016
function [ weight, bias ] = MyNetwork(inputs, targets, nodeLayers, numEpochs, batchSize, eta)
% this is the project one for CSC 578, Neural Network and Deep Learning
% step one: initilize weights and bias matrices
weight = {}; % initialize weigt matrix with cell
bias = {}; % initialize bias matrix with cell
for toLayer = 2:length(nodeLayers) % for each nodeLayer input
    weight{toLayer} =normrnd(0, 1, nodeLayers(toLayer), nodeLayers(toLayer-1)); % initialize the weights
    bias{toLayer} = normrnd(0, 1, nodeLayers(toLayer), 1); % initialize the bias
    % initialize weights and biases with normal distribution, mean = 0, sd=1
end

% step two: batching
totNumInst = size(inputs, 2); % total number of samples
targetbatches = {}; % to store target batches
batches = {}; % to store batches
numbatch = 1; % to index the position of each batch
for StartInsPos = 1:batchSize:totNumInst % divide the sample into minibatches
    if totNumInst-StartInsPos < batchSize % this avoids index out of range problem
        batch = inputs(:, StartInsPos:end); % a matrix representing the batch
        batches{numbatch} = batch;
        targetbatches{numbatch} = targets(:, StartInsPos:end);
    else
        batch = inputs(:, StartInsPos:StartInsPos+batchSize-1); % a matrix representing the batch
        batches{numbatch} = batch;
        targetbatches{numbatch} = targets(:, StartInsPos+batchSize-1);
        numbatch = numbatch + 1 ;
    end
end

% step three: start epochs
for iteration = 1:numEpochs % for each epochs
    for batchindex = 1:numbatch % for each batch
        z = {}; % to store intermediate values
        activations = {}; % initilize the activation matrix
        activations{1} = batches{batchindex}; % first activation layer is the input layer
        deltas = {}; % to store deltas for gradient descent
        for layer = 2:length(nodeLayers)
            z{layer} = bsxfun(@plus, (weight{layer} * activations{layer-1}), bias{layer});
            activations{layer} = logsig(z{layer}); % calculate and store the activation for next layer
        end
        % calculate output error aka cost
        % criterion: quadratic cost function
        error = activations{length(nodeLayers)} - targetbatches{batchindex};
        sigprime = logsig(z{length(nodeLayers)}).*(1-logsig(z{length(nodeLayers)}));
        cost = error.*sigprime; % this is the delta of last layer
        
        % step four: backpropagation
        deltas{length(nodeLayers)} = cost; % last layer of delta is the quadratic cost function output
        for layer = (length(nodeLayers)-1):-1:2 % skip the last layer
            deltas{layer} = (weight{layer+1}.'*deltas{layer+1}).*(logsig(z{layer}).*(1-logsig(z{layer})));
        end
        
        % step five: gradient descent
        for layer = length(nodeLayers):-1:2 % start from the last layer
            w = weight{layer} - eta/length(batches{batchindex})*deltas{layer}*activations{layer-1}.';
            weight{layer} = w;
            b = bias{layer} - eta/length(batches{batchindex})*sum(deltas{layer}, 2);
            bias{layer} = b;
        end
    end
    
    % step six: calculate output statistics and display a message for each epoch
    output = {}; % to store the output calculated using updated weights
    output{1} = inputs;
    for layer = 2:length(nodeLayers)
        z2 = bsxfun(@plus,(weight{layer} * output{layer-1}), bias{layer});
        output{layer} = logsig(z2); % calculate and store the activation for next layer
    end
    
    % calculate MSE, number of correct cases and accuracy rate
    error = output{length(nodeLayers)}-targets;
    MSE = sqrt(sum(sum(error.^2)))/(2*length(inputs)); % mean() only divide by n, but we want to divde by 2n
    
    if size(targets, 1) > 1 % if it is a multi class problem
        [mx, loc] = max(output{length(nodeLayers)}); % max value of each column/observation
        [mx2, loct] = max(targets);
        t = loct - loc;
        correct = sum(t(:)==0);
    else % if it is a binary classification problem, we set the threshold at 0.5
        positive = round(output{length(nodeLayers)});
        correct = sum(abs(targets-positive)==0); % incorrect cases will have a sum greater than zero
    end
                            
    totNumCase = size(inputs, 2);
    accuracy = correct/totNumCase;
    
    if numEpochs <= 100 % print message for each iteration if running a small number of epochs
        fprintf('Epoch %d, MSE: %f, Correct: %d/%d, Acc: %f \n', iteration, MSE, correct, totNumCase, accuracy);
    elseif mod(iteration, 100) == 0 && numEpochs > 100
        % print only hundredth iteration's message if running a big number of epochs
        fprintf('Epoch %d, MSE: %f, Correct: %d/%d, Acc: %f \n', iteration, MSE, correct, totNumCase, accuracy);
    end
    
    % another stopping criterion: accuracy = 1
    if correct == totNumCase
        fprintf('Epoch %d, MSE: %f, Correct: %d/%d, Acc: %f \n', iteration, MSE, correct, totNumCase, accuracy);
        break
    end
end

if mod(numEpochs, 100) ~= 0 && numEpochs > 100
    % print the result for the last iteration for large
    fprintf('Epoch %d, MSE: %f, Correct: %d/%d, Acc: %f \n', numEpochs, MSE, correct, totNumCase, accuracy);
end

end
