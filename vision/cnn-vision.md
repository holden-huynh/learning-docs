# CNN - Vision
## <span style="color:yellow">1. Basic</span>
* Visualization
	* http://cs231n.github.io/understanding-cnn/
	* https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
* Cheatsheet: https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-deep-learning
* Loss / Optimization
	* sanity check: when W ~ 0
### Training NN:
* one time setup: activation func, preprocessing, weight init, regularization, gradient checking
* dynamics: babysitting learning process, params update, hyper-params optimization
* evaluation
### Activation  function:
* sigmoid / tanh:
    * squash functions: simulates neuron activations
    * saturation -> vanishing grads
    * tanh > sigmoid: non zero-centered converges slower (inputs to all neurons always > 0 -> grads always either > 0 or < 0 -> zigzag updates)
* ReLU
    * not saturated, computationally efficient, converges faster
    * non zero-centered
    * dead neurons  —> init with slightly positive bias + monitor activations (statistics over entire dataset)
### Data preprocessing
* normalize mean
    *  subtract mean image (per pixel mean)
    * subtract per-channel mean
* References:
    * https://pdfs.semanticscholar.org/475b/16a20c71bd1648f79cfcb9d75db94166778d.pdf
    * https://papers.nips.cc/paper/4421-a-convergence-analysis-of-log-linear-training.pdf
### Weight init
* if all biases b = 0:
* if all W = 0 —> all neurons activate the same —> same grads —> not symmetry breaking
* instead small random numbers: works ok for small networks, but lead to non-homogenous distributions of activations across layers:
    * W ~ N(0, 0.001): X = activations[l] ~ 0 —> grad(W[l+1]) = X ~ 0
    * dL / dX ~ W -> behavior similar to forward pass: all collapse to 0s, grads of lower layers vanished
    * W ~ N(0, 1.0) : saturated —> grad(W) = 0
* Xavier init:
    * W = 1/sqrt(fan_in)
    * works with tanh
    * breaks with RELU —> He init
* He init:
    * RELU halves the variance
    * -> /2 factor: W = 1/sqrt(fan_in / 2)
### Batch norm
* batch norm layer *before* nonlinearities
* normalization is differentiable
* free to cancel or take advantage of normalization
* improve grad flow + allows higher lr + reduces dependence on init + regularization
* at test time: use fixed mean/std
* Related works:
    * [Mean-normalized SGD](http://www-i6.informatik.rwth-aachen.de/publications/download/903/Wiesler-ICASSP-2014.pdf)
    * http://yann.lecun.com/exdb/publis/pdf/raiko-aistats-12.pdf
    * [Natural NN](https://pdfs.semanticscholar.org/5941/5994e8a1a5fce32fef32929e74d3138697b6.pdf?_ga=2.15411702.1945782316.1564575345-754707387.1564575345)
### Babysitting Learning process:
References:
- https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/
- 
#### Sanity check:
* disable regularization, make sure loss correct. e.g. loss of Cifar10 is ~ -log(1/10) ~ 2.3
* crank up regularization, loss went up
* take small piece of training data, make sure NN overfits (turn off regularization)
#### Hyperparams CV
* find lr: on small piece of data (start with small regularization)
* with softmax, even loss changed slightly, accuracy can change much more
* coarse to fine CV for some epochs. 
    * first stage: a few epochs, rough idea of what params work
    * second stage: longer running time, finer search 
    * e.g. detect explosions (e.g cost > 3 * original cost) in solver, break out early
* Best to optimize (sampling) in log scale:
    * params act multiplicatively in the dynamics of backprop?
    * params are not sensitive wrt performance
* Tips: care about best params at the edges of the range —> maybe need to expand the range
* random sampling better than grid search!
#### Monitor:
* loss functions
* accuracy: interpretable (v.s loss)
* track the ratio: weight update / weight magnitute ~ 1e-3
## 2. Optimization
### GD:
* NN loss surface: all local minima are almost the same in large NN!
* momentum: like moving average —> stablize grads
* Nesterov momentum
	* inconvenience: forward and grad backward at different points
* AdaGrad
	* per feature learning rate: accumulated grad^2
	* problem: lr decays toward 0 in the long run
* —> RMSProp remedies the lr decay of AdaGrad: leaky accumulated grad^2
* Adam: combines AdaGrad and momentum
	* momentum (m): 1st order moment
	* adagrad (v): 2nd order
	* bias correction: m, v estimates may be incorrect at the beginning (m, v = 0)
* lr decay: applied for all
	* step decay
	* exponential decay
### 2nd-order method
### References:
- https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
## 3. Regularization
### dropout
* intuition
	* redundant representation
	* ensemble
* training time
* test time: 
	* ideal: integrate out all the noise (or MC approx)
	* practice: single forward pass (no dropout)
		* —> account for the change in scales of activations btw training and test: scale activations so that *for each neuron output at test time = expected output at training*
		* more common: inverted dropout, test time untouched
		* what about non-linearity?
## 4. ConvNet
### Convolution
#### Convolution
* input W1 x H1 x D1
* K filters: F x F
* stride S
* zero padding P
* Formula:
	* W2 = (W1 - F + 2P) / S + 1
	* H2 = (H1 - F + 2P) / S + 1
	* D2 = K
	* Weights: K x F x F x D1 (+1 bias)
* is linear transformation

$ X: Nx1 $
$ kernel: kx1 $
-> conv matrix: K (N-k+1)xN 

#### Transpose convolution
* Equivalent to (fractional) strided convolution
* upsample: K^T
### Design
* padding: preserves size, keeps feature maps from collapsing quickly
* K: power of 2 (computational reasons)
* pooling: throw away a bit spatial information
### Models
* VGG:
	* memory ~ 100MB, 138m params
	* most memory in early CONV
	* most params in late FC —> instead of FC, should use average pooling across spatial dim (7 x 7 x 512 —> 1 x 1 x 512)
* Resnet:
	* skip connections —> easy to learn early layers
### Visualization
* First layer: visualize filters (no biases), almost the same for different architectures, datasets etc.
* Higher layers: visualizing filters make less senses
* Last layer: features before the classifier (e.g. 4096-dim in AlexNet)
    * feed a lot of images (test set)
    * NN in feature space -> semantic similarities
    * interesting coz loss functions don't explicitly constrain features to be close
    * PCA / t-SNE
* Activation maps of intermediate layers
    * visualize every channel as a gray scale image, e.g. highly activated cells / neurons in feature maps corresponding to human faces
* Maximally activating patches of input images: which patches cause maximum activation in different neurons
    * Pick a channel of a layer
    * Pass images and record activations of the selected channel
    * Visualize patches corresponding to maximal activations
* Occlusion experiments: which part of the input image cause the network to make classification decision
* Saliency maps: which pixels in the image are important for classification
    * masking
    * salency maps: __gradient__ of predicted class score wrt pixels of input image
* Guided backprop: similar to saliency maps, but compute grads for an intermediate neuron
## 5. RNN
RNN is able to generalize to sequences longer than those in training!
### LSTM
* W: 4n x 2n
* (x; h): 2n
* —> (i, f, o, g): 4n (sigm, sigm, sigm, tanh)
	* 4 gates are named w.r.t cell state, e.g input / output / forget applied to cell states
	* i, f, o: binary gates (sigm to be differentiable)
	* g: how much we want 
	* c_t = f * c_{t-1} + i * g : forget + new updates
	* h_t = o * tanh(c_t)  *or* o * c_t: cell states leak to hidden states
### Gradient flow dynamics: vanishing gradients
* control by additive interactions
### Tricks:
* init forget gates: small positive bias to turn off forget gates at beginning
## 6. Detection
* Loss: weighted sum of classification and detection head's losses (weight is hyperparam)
* CV for the weight: using another performance metric
### Region-based
#### R-CNN
* Separate CNN computation for each region proposal

#### Fast R-CNN
* Share expensive CNN computation for all regions
* ROI pooling to fixed size --> FC layer
* Bottleneck is Selective search (Region proposal)

#### Faster R-CNN
* Region Proposal Network (RPN)
* Jointly train with 4 losses:
    * RPN classify objectness (binary)
    * RPN regress boundingbox coordinates
    * Final classification score (object class)
    * Final boundingbox coordinates
* Balancing multi-task losses is tricky!
### Single-Shot
#### YOLO
* grid G x G
* each cell: B boxes
    * regress from each of B base boxes: (dx, dy, dw, dh, confidence)
    * predict scores for each of C classes
* -> outputs: G x G x (5B + C)

#### SSD
* abc
### Comparison:
* Faster R-CNN:
    * Region proposal: fixed end-to-end regression proble
    * separate per region processing
* Single-shot:
    * Only 1 single step
## 7. Segmentation
