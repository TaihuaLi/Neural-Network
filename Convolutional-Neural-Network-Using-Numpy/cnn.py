# -*- coding: utf-8 -*-
# author: Taihua Li
# convolutional neural network built with Python 2.7

import numpy as np
from scipy import signal
import time

# miscellaneous functions (activation, etc.)
def sigmoid(x):
    return np.array(1 / (1 + np.exp(-x)))

def sigmoid_prime(x):
    return np.array(sigmoid(x)*(1.0-sigmoid(x)))

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.gradient(x)[0]
    
# main functions
def wb_init(shape):
    # weight is initialized to be between [-1, 1] with mean 0 and std 1
    # bias is initialized to be between [-1, 1] with mean 0 and std 1
    weights = []
    if len(shape) == 4: # convolutional layer
        for i in range(shape[-1]):
            weights.append(np.random.uniform(-1, 1, shape[0:2]))
            biases = np.random.uniform(-1, 1, shape[-1])
        return weights, biases
    elif len(shape) == 2: # fully connected layer
        weights = np.random.uniform(-1, 1, shape)
        biases = np.random.uniform(-1, 1, shape[1])
        return weights, biases

def convolution(in_img, w, b, convo=True, stride=1):
    if convo:
        # if it is a convolutional layer
        # zero pad all feature maps/input images
        padded = []
        for feature in in_img:
            padded_img = np.lib.pad(feature, int(w[0].shape[0]/2), 'constant')
            padded.append(padded_img)
        
        out_fea = []
        
        for i in range(len(b)):
            out_img = np.zeros(in_img[0].shape)
            for j in range(len(padded)):
                for row in range(0, in_img[0].shape[0], stride):
                    for col in range(0, in_img[0].shape[1], stride):
                        # multiply weights with regions from all feature maps and sum them up
                        # assign the sum of all feature maps to one new pixel in one output map
                        out_img[row, col] +=  np.sum(np.multiply(padded[j][row:row+w[i].shape[0], \
                                                                 col:col+w[i].shape[1]], w[i]))
            # add bias term
            out_img = np.add(b[i], out_img)            
            # activate the output
            # out_img = activation(out_img)
            # store the feature map
            out_fea.append(out_img)
    else:
        # if it is a affine layer
        out_fea = np.dot(in_img, w)+b
    return out_fea

def pooling(in_fea, alpha, pool_type = 'max'):
    # initialize a new output feature map set
    out_fea = []
    # take each input feature map and shrink them by alpha
    for i in range(len(in_fea)):
        out_img = np.zeros([in_fea[i].shape[0]/alpha, in_fea[i].shape[1]/alpha])
        for row in range(0, in_fea[i].shape[0], alpha):
            for col in range(0, in_fea[i].shape[1], alpha):
                if pool_type == 'max':
                    out_img[row/alpha, col/alpha] = np.max(in_fea[i][row:row+alpha, col:col+alpha])
                elif pool_type == 'min':
                    out_img[row/alpha, col/alpha] = np.min(in_fea[i][row:row+alpha, col:col+alpha])
                elif pool_type == 'average':
                    out_img[row/alpha, col/alpha] = np.average(in_fea[i][row:row+alpha, col:col+alpha])
        out_fea.append(out_img)
    return out_fea

def train(train_x, train_y, epoch = 1, batch_size = 10, eta=0.1, stride=1, poolind=2, poolme='max', act=sigmoid, actprime =sigmoid_prime):

    # initialize placeholders for weights and biases
    weights = {}
    biases = {}
    al_index = [] # keep tracking which layer is the affine layer so we don't pool
    
    # ask user to set up the network (filter size and dimensions of each layer)
    # how many feature map as output???
    # weight & bias initialization
    counter = 1
    while True:
        print 'For layer {}, please enter the layer shape:'.format(counter)
        print 'Convolutional layer: [filter height, filter width, input dim, output dim]'
        print 'Affine layer: [input dim, output dim]'
        print 'Warning: last layer output dimension has to agree with the label dimension'
        print '(Enter START to start training......)'
        s = raw_input()
        if s.lower() == 'start':
            break
        if len(eval(s)) == 4: # convolutional layer
            w, b = wb_init(eval(s))
            weights[counter] = w
            biases[counter] = b
            counter += 1
            print 'Initializing weights and biases... \n'
        elif len(eval(s)) == 2: # affine layer
            w, b = wb_init(eval(s))
            weights[counter] = w
            biases[counter] = b
            al_index.append(counter)
            counter += 1
            print 'Initializing weights and biases... \n'      
    
    x_batches = {}
    y_batches = {}
    batch_counter = 1
    # batching
    for i in range(0, len(train_x), batch_size):
        if len(train_x) - i < batch_size: # last batch
            x_batch = train_x[i:]
            x_batches[batch_counter] = x_batch
            y_batch = train_y[i:]
            y_batches[batch_counter] = y_batch
        else:
            x_batch = train_x[i:i+batch_size-1]
            x_batches[batch_counter] = x_batch
            y_batch = train_y[i:i+batch_size-1]
            y_batches[batch_counter] = y_batch
            batch_counter += 1
    
    for iteration in range(epoch):   
        for batch_index in range(len(x_batches.keys())):
            # each element in train_x is an image
            batch_to_use = x_batches[batch_index]
            batch_label = y_batches[batch_index]
            preact = {} # to store pre-activation for each image for backprop
            activation = {} # to store feature maps for each image
            image_index = 0
            for image in batch_to_use:
                # forward pass
                activation[image_index] = {}
                activation[image_index][0] = image
                preact[image_index] = {}
                for i in range(1, counter):
                    if i not in al_index:
                        a = convolution(activation[image_index][i-1], weights[i], biases[i], True, stride)
                        preact[image_index][i] = a # save the preactivation for backprop
                        activated = []
                        for s in a:
                            activated.append(act(s)) # activate it
                        p = pooling(activated, 2, poolme)
                        activation[image_index][i] = p
                    else:
                        reshaped = np.array(activation[image_index][i-1]).flatten() # vecotrize the input
                        a = convolution(reshaped, weights[i], biases[i], False, stride)
                        preact[image_index][i] = a
                        activated = []
                        for s in a:
                            activated.append(act(s)) # activate it
                        activation[image_index][i] = activated
                image_index += 1
            
            # activation[image_index] to select which image you want
            # activation[image_index][layer_index] to select the output of a specific layer for a specific image
                
            # calculate output error (MSE) -- can change later
            tot_error = []
            error = []
            for i in range(image_index):
                tot_error.append(np.sum((activation[i][counter-1] - batch_label[i])**2))
            error = [x/2 for x in tot_error]
            
            # backpropagation
            delta = {}
            delta[counter-1] = {}
            # calculate the delta for the output layer
            for i in range(image_index):
                delta[counter-1][i] = error[i] * np.array(actprime(np.array(preact[i][counter-1])))
            
            # start calculating the delta for each layer backward
            for i in range(counter-2, 0,-1):   
                start_point = time.time()
                delta[i] = {}
                for img in range(image_index):
                    if i in al_index: # if it is an affine layer
                        delta[i][img] = np.multiply(np.dot(weights[i+1], delta[i+1][img].flatten()), actprime(preact[img][i]).flatten())
                    elif i == min(al_index) - 1: # if it is the layer between affine and convolution layer
                        delta[i][img] = {}
                        meh = []
                        for fp in activation[img][i]:
                            meh.append(fp.flatten())
                        meh2 = [item for sublist in meh for item in sublist]
                        delta[i][img] = np.multiply(np.dot(weights[i+1], np.array(delta[i+1][img]).flatten()), actprime(np.array(meh2)))
                        delta[i][img] = np.reshape(np.array(delta[i][img]).flatten(), [len(activation[img][i])]+list(activation[img][i][0].shape))
                        # pad each feature map and calculate the delta to return it to 16x16 size
                        new = {}
                        for fp in range(len(delta[i][img])):
                            ahh = np.pad(delta[i][img][fp], int((delta[i][img][fp].shape[0]*poolind-delta[i][img][fp].shape[0])/2), 'edge')
                            new[fp] = np.multiply(signal.convolve2d(ahh, np.rot90(weights[i][fp], 2), mode='same'), np.array(actprime(preact[img][i][fp])))
                        delta[i][img] = new
                    
                    #from this step, the delta will be organized as delta[layer index][input image index][feautre map index]
                    #in this case, there are 16x16 feature maps (before pooling) produced, so the delta matrix size for each feature map
                    #will be 16x16. in the next layer, the matrix size will be 32x32, which is the preactivation and pre-pooling size
                    
                    else: # if it is a convolution layer
                        for img in range(image_index):
                            delta[i][img] = {} # dimensionality hell
                            for fp in range(len(activation[img][i])): # output 32 delta matrices
                                temp_sum = np.zeros((np.lib.pad(delta[i+1][img][0], int((delta[i+1][img][0].shape[0]*poolind-delta[i+1][img][0].shape[0])/2), 'constant')).shape)
                                for deltafp in range(len(delta[i+1][img].keys())): # for each of the 64 feature maps
                                    # resize it to 32x32
                                    temp = np.pad(delta[i+1][img][deltafp], int((delta[i+1][img][deltafp].shape[0]*poolind-delta[i+1][img][deltafp].shape[0])/2), 'edge')
                                    temp2 = np.multiply(signal.convolve2d(temp, np.rot90(weights[i][fp], 2), mode='same'), actprime(preact[img][i][fp]))
                                    temp_sum = np.add(temp_sum, temp2)
                                temp4 = np.divide(temp_sum, len(delta[i+1][img].keys()))
                                delta[i][img][fp] = temp4
            
            # update weights & biases for affine layers
            for i in range(len(delta.keys()), min(al_index) - 1, -1): # for each affine layer
                affine_w_d = np.zeros(weights[i].shape)
                affine_b_d = np.zeros(biases[i].shape)
                for img in range(image_index):
                    temp = np.dot(np.multiply(eta/image_index, np.matrix(np.array(activation[img][i-1]).flatten()).T), np.matrix(delta[i][img]))
                    affine_w_d = np.add(affine_w_d, temp)
                    temp2 = np.multiply(eta/image_index, np.matrix(delta[i][img]))
                    affine_b_d = np.add(affine_b_d, temp2)
                weights[i] = np.subtract(weights[i], affine_w_d)
                biases[i] = np.subtract(biases[i], affine_b_d)
            
            # update weights & biases for convolutional layers
            # step one: average the deltas that are on the same feature map and multiplies with learning rate and calculate the average
            new_delta = {}
            for i in range(min(al_index) - 1, 0, -1): # for each convolutional layer
                new_delta[i] = {}
                for fp in range(len(weights[i])):
                    temp_sum = np.zeros(delta[i][0][fp].shape)
                    for img in range(image_index):
                        temp_sum = np.add(temp_sum, delta[i][img][fp])
                    new_delta[i][fp] = np.average(np.multiply(eta/image_index, temp_sum))
            # step two: calculate cost gradient
            convo_cg = {}
            for i in new_delta.keys():
                convo_cg[i] = {}
                for fp in range(len(weights[i])):
                    temp1 = 0
                    for in_img in range(len(activation[i-1])):
                        temp0 = np.multiply(new_delta[i][fp], np.sum(actprime(np.array(activation[in_img][i-1]))))
                        temp1 = temp1+temp0
                    convo_cg[i][fp] = temp1
            # step three: update weights and biases
            for i in convo_cg.keys():
                for j in range(len(convo_cg[i])):
                    weights[i][j] = np.subtract(weights[i][j], convo_cg[i][j])
                    biases[i][j] = np.subtract(biases[i][j], new_delta[i][j])
        
        # end an epoch, evaluate that epoch
        activation = {}
        image_index = 0
        for image in train_x:
            # forward pass
            activation[image_index] = {}
            activation[image_index][0] = image
            for i in range(1, counter):
                if i not in al_index:
                    a = convolution(activation[image_index][i-1], weights[i], biases[i], True, stride)
                    activated = []
                    for s in a:
                        activated.append(act(s)) # activate it
                    p = pooling(activated, 2, poolme)
                    activation[image_index][i] = p
                else:
                    reshaped = np.array(activation[image_index][i-1]).flatten() # vecotrize the input
                    a = convolution(reshaped, weights[i], biases[i], False, stride)
                    activated = []
                    for s in a:
                        activated.append(act(s)) # activate it
                    activation[image_index][i] = activated
            image_index += 1
        
        # calculate MSE
        tot_error = []
        prediction = []
        for i in range(image_index):
            tot_error.append(np.sum((activation[i][counter-1] - train_y[i])**2))
            prediction.append(np.argmax(activation[i][counter-1]))
        error = sum([x/2 for x in tot_error])
        prediction = np.array(prediction)
        truth = np.argmax(train_y, 1)
        accurate = np.sum(prediction==truth)
        accuracy = accurate/float(len(truth))
        end_point = time.time()
        time_spent = end_point - start_point
        
        # print 'For epoch {}, cost: {}, accuracy: {0:.5f}, correct: {}/{}'.format(iteration, error, accuracy, accurate, len(truth))
        print 'For epoch', str(iteration+1), ' cost: %.7f'%round(error, 7), ' accuracy: %.3f'%round(accuracy, 3), ' correct:', str(accurate), '/', str(len(truth)), ' time spent: %.2f'%round(time_spent,2),'seconds'                
    return weights, biases, activation
    
# data processing function
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
    
def get_label(lst, low=0, high=9): #specify class labels
    m = np.zeros([len(lst), high-low+1])
    m[np.arange(len(lst)), lst] = 1
    return list(m)
    
def get_shape(i, shape):
    lst = []
    for img in i:
        temp = np.reshape(img, shape)
        lst.append(temp)
    return lst
    
    
    