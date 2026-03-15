# Project1_Breast_Cancer_Classification
Classification of people having breast cancer with pytorch

## BASIC TERMINOLOGIES 

Deep Learning is a subset of ML which is a subset of ML. ML teaches computers to recognize patterns in data. 

Deep learning is ML technique that learns features and tasks directly from data by running inputs through neural networks. NNs form the basis of DL. These NNs have hidden layers- for data processing, making connections and weighing inputs. 

## NEURAL NETWORKS
Take data as input > train themselves to understand patterns in data > output useful predictions. NNs consist of input layer, output layer and several hidden layers between the two. 

### LEARNING PROCESS OF A NN
Two processes: forward propagation and backward propagation. 

1. FORWARD PROPAGATION: Propagation of info from input layer to output layer. Input neurons are connected to the next layer through channels which are assigned a weight. The i/ps are mutliplied to weights and their sum is sent as i/p to neurons in hidden layer where each neuron in turn is associated with a numerical value called bias which is then added to input sum. This weighted sum is then passed through a non linear function called activation function which decides if the particular neuron can contribute to the next layer. The output layer is a form of probability. The neuron with the highest value determines what the output finally is.
  - Weight: how important the neruon is
  - Bias: allows for the shifting of activation function to the right or left

2. BACK PROPAGATION: Info here is passed from o/p layer to the hidden layers. The NN evaluates its performance and checks if it is right or wrong. The network uses a loss function to quantify the deviation from the expected output. This info is sent back to the hidden layers for weights and biases to be adjusted so n/w's accuracy levels are increased. 

### STEPS
1. Initialize a NN w/ random weights and biases
2. Move i/p data through NN. Take inputs, multiply by the weights and add the bias
3. Run this weighted sum thru an activation function
4. The output of the activation function is then multiplied by weights, added to bias and the process repeats until you reach o/p layer
5. Start backpropagation: quantify diff between expected result and predicted o/p using a loss function
6. Go backwards and adjust initial weights and biases i.e., propagate this loss back thru the n/w and update params w/ gradient descent algo
7. Continue iterating until you have a good enough model

The more data you give to the NN the better it'll be at predicting the right output. Too much data => overfitting. 

## ACTIVATION FUNCTIONS
Introduces non linearity in the network and also decides whether a neuron can contribute to the next layer. How to decide when to activate the neuron:

1. Step function: If value > 0 -> Activate, Else -> Do not activate. Problem- it just outputs activated or not. Better to activate with some probability. Eg: if more than 1 neuron activates, we can find neuron fires based on which has the highest probability. 
2. Linear function: Stragiht line function where activation is proportional to input by the slope of the line. This way activation is not binary. Problem- in gradient descent teh derivative of a linear function is a constant. Eg: y = mx + b, so y' = m. Here the derivative has no relationship with x. Hence during backpropagation, the adjustments made to the weights and biases aren't dependent on x. So, activation function of final layer is just a linear function of input of first layer- hence entire NN can be replaced by single layer (we cant stack layers). O/p is in range of - infinity to + infinity.
3. Sigmoid function: a(x) = 1/(1+e^-z) is activation function. It is non linear in nature- so we can stack layers. We also get analog output and has a smoother gradient. O/p is in range of 0 to 1. Hence, activations wont blow up. Problem- in the range of x = -2 to x = 2, the y vals are very steep i.e., small changes in vals of x, lead to drastic y val changes. Moreover, towards either end of the function, y vals respond very little to changes in x i.e., gradients in those regions will be very small =>  VANISHING GRADIENT PROBLEM (says that if i/p of activation function is large or small, the sigmoid squishes it down to a val b/w 0 and 1 and the gradient of this function becomes very small)
4. Tanh function: Shifted sigmoid function. It is non linear (so we can stack layers), bound to range from -1 to 1. So activations dont blow up. The derivatives of the tanh function is steeper than the sigmoid. Problem- similar to sigmoid, vanishing gradient.
5. ReLU Function (Rectified Linear Unit): a(x) = max(0,x). Non linear, ranges from 0 to infinity so there is a chance of blowing up the activation. Since it outputs 0 for negative vals of x, nearly 50% of all neurons fire- sparse activation- making the n/w lighter. However, gradient is 0 in the region of negative x vals. During back propagation, the weights will not get adjusted during descent. Hence, neurons going into the state of deactivation will not respond to variations in the error because the gradient is 0. This is called DYING RELU PROBLEM. Neurons in the n/w die and not respond hence making a substantial part of the n/w passive than active.
6. Leaky ReLU function: To solve the dying ReLU problem, we simply make the horizontal line in the negative x axis into a non horizontal line by adding a slope around 0.001. Hence, gradient doesnt become zero. Note: in parametric ReLU, the slop of the horizontal line is adjusted to y = ax. 


Sparsity of an activation: In a big NN w/ several neurons. A sigmoid or tanh function will cause almost all neurons to fire in an analog way => almost all activations will be processesd to get the n/w o/p. Hence, the activation is dense. This is costly. We ideally want only a few neurons in the NN to activate, making the activation sparse and efficient.

### WHICH ACTIVATION FUNCTION TO USE AND WHY WE USE NON LINEAR AF?

A sigmoid activation function works well for binary classification probelms. Else you can use ReLU (or modified ReLU). This leads to faster training processes or larger convergence. 

Activation functions serve to introduce non linearity in the n/w. Introducing non linearity means that the activation function should be non linear (not a straight line) or polynomials of degree > 1. 

If we use a linear activation function to model our data, no matter how many hidden layers our model has, it will always become equivalent to a single layer n/w. 

## LOSS FUNCTIONS
Way to quantify the deviation of the predicted output by the neural  network to the expected o/p. Different loss functions: 
  - Regression: Squared error, huber loss
  - Binary classification: Binary cross esntropy, hinge loss
  - Multi class classification: Multi class cross entropy, Kullback Divergence

## OPTIMIZERS
During training, we adjust the params to minize the loss function and make our model as optimized as possible. Optimizers tie together the loss function and model params by updating the n/w based on the o/p of the los function. Loss functions guide optimizers and tell it whether its moving in the right or wrong direction.  

### GRADIENT DESCENT
Iterative algo that starts off at a random poitn on the loss function and travels down its slope in steps until it reaches the lowest point (minimum) of the function. Most popular optimizer. Its working:
  1. Calculate what a small change in each individual weight would do to the loss function
  2. Adjust each parameter based on its gradient (i.e., take a small step in the determined direction)
  3. Repeat steps 1 and 2 until the loss functions is as low as possible

The gradient of a function is the vector of partial derivates wrt all independent variables. It always points in the direction of the steepest increase in the function. 

For high dimensional datasets(lots of variables), we ideally want to find the global min of the loss function. To avoid getting stuck in a local minima, we need to use a proper LEARNING RATE. Learning rate is usually a small number like .001 which are mutliplied TO SCALE GRADIENTS. This ensures that changes made to the weights are small and at the right pace.

We don't want large learning rate where algo will overshoot global minimum. Similarly, we don't want very small learning rate where algo takes a long time to converge to global min. Steps that are too small might lead to the optimizer converging on a local min for the loss function but never absolute min. 

### STOCHASTIC GRADIENT DESCENT
Like gradient descent, except uses a SUBSET of training examples rather than the entire lot. SGD is an implementation of GD that uses batches on each pass. It uses momentum to accumulate gradients (of past steps to determine what may happen in future steps) and is less expensive computationally (since we dont include entire training set). 

Backpropagation is basically gradient descent implemented on a n/w. Other kinds of optimizers based on gradient descent are: Adagrad, RMSprop, Adam.

### ADAGRAD
  - Adapts learning rate to individual features
  - Hence, some weights have diff learning rates
  - Ideal for sparse datasets with many input examples missing
  - Issue: adaptive learning rate tends to get very small with time

### RMSprop
  - Speicalized version of Adagrad
  - Instead of letting all gradients accumulate from momentum, it accumulates gradients in a fixed window
  - Similar to Adaprop

### Adam
  - Stands for Adaptive Moment Estimation. Uses past gradients to calculate current gradient
  - Uses the concept of momentum. Tells NN whether we want past changes to affect new changes by adding fractions of prev gradients to current one

## PARAMETERS VS HYPERPARAMETERS
Model parameters: variables internal to the NN. Value can be estimated from the data. Points are:
  - Required by model to make predictions
  - Values define the skill of the model
  - Estimated directly from data
  - Not set manually
  - Saved as part of the learned model
  - Eg: Weights, Biases

Model hyperparams: Configurations external to the NN. Value cant be estimated right from the data. Points are"
  - No clear way to find the best value
  - When a DL algo is tuned (using grid search or random search), you are tuning the hyperparams of model to discover the params that result in better predictions
  - If you have to manually specify a param, it is a hyperparam
  - Eg: Learning rate

Model params are estimated from the data. Model hyperparams cant be estimated from the data. 

## EPOCHS, BATCHES, BATCH SIZES AND ITERATIONS

EPOCHS
  - One epoch is when the entire dataset is passeed forward and backward thru the NN only ONCE
  - We use multiple epochs to help our model generalize better. GD is an iterative algo and updating params and back propagation in a single pass or 1 epoch isnt enough
  -  As no. of epochs increases, the more the params are adjusted leading to a better performing model
  -  Too many epochs can lead to overfitting where a model has memorized patterns in training data and performs bad on data it hasnt seen before

BATCH AND BATCH SIZE
  - Need these if dataset is large
  - Break down the dataset into smaller chunks or batches and feed those chunks to the NN one by one and update the weights of the nn at the end of every step to fit it to the data given
  - Batch size: total no. of training examples in a batch

ITERATIONS
No. of batches to complete one epoch. Number of batches = Number of iterations for one epoch. Eg: Suppose we have a dataset of 34000 training examples and you divide the dataset into batches of 500. To complete 1 epoch, it would take 68 iterations. 
