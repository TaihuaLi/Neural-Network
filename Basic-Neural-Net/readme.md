# Neural Network Implementation One
## With Gradient Descent Using Backpropagation

Transformation function: Sigmoid

### Pseudocode
Initialize a cell for weights
Initialize a cell for biases

For each layer starting from the second layer
Initialize weights with normally distributed values and fill in its corresponding position in the weight cell
Initialize biases with normally distributed values and fill in its corresponding position in the biases cell

Initialize a cell to store batch values
Initialize a cell to store target values
Initialize a counter to keep tracking batch indexes

For each starting index of the instance of each batch
	If desired batch size is greater than the actual batch size
		Insert the batch with actual size into the batch cell
		Insert the corresponding target batch into the target cell
	Else
		Insert the batch with desired size into the batch cell
		Insert the corresponding target batch into the target cell
		Increment the batch index counter by one

For each epoch
	For each batch
		Initialize a cell to store intermediate values
		Initialize a cell to store activations
		Set the first element of activation cell to the input batch

		For each layer starting from the second layer
			Intermediate value is calculated 
Activation values is calculated by transforming the intermediate value which is then stored into the activation cell

		Calculate error which is the difference between activations and targets
		Calculate sigmoid gradient with respect to the intermediate values
		Calculate the cost by multiply the error and sigmoid gradient
		Initiate a cell for deltas
		Set the last element of the delta cell to the cost

		For each layer starting from the second to the last layer to the second layer
Delta of that layer is calculated by multiplying weights with deltas and multiplying the product with sigmoid gradient 

		For each layer starting from the last layer to the second layer
Update the weights by subtracting learning rate divided by the total number of observations in the batch times the result of delta of the layer multiplies with the next layer’s activations, from current weights
Update the biases by subtracting learning rate divided by the total number of observations in the batch and then multiplying by the sum of the deltas in the layer, from the current biases

Initialize the final output cell
Set the first element of the output cell to inputs

For each layer starting from the second layer to the end
	Caculate intermediate values
	Store transformed intermediate values into the output cell

Calculate the mean squared of error which is the squared errors divided by two times the total observations

If it is a multiclass problem
	Get the matrix location of the max value of the final output
	Get the matrix location of the max value of the targets
	Subtract the two location vectors
	Count the number of zeros in the vector which are the correct cases
Else
	Round up or down the final output vecotr
	Count the number of zeros in the vector which are the correct cases

Calculate the overall accuracy
	
	If the number of input epoch is less or equal to 100
		Display iteration number, MSE, correct cases over total cases and accuracy
Else if the number of input epoch is greater or equal to 100 and the MOD of epoch and 100 and equal to 0
Display iteration number, MSE, correct cases over total cases and accuracy of that epoch
	
	If the accuracy is 1
Display iteration number, MSE, correct cases over total cases and accuracy of that epoch
Stop the function

If number of epoch is greater than 100 and the MOD of number of total epochs and 100 does not equal to 
Display iteration number, MSE, correct cases over total cases and accuracy of that epoch
