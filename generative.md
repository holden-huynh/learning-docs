# Generative Models
## Materials:
* [CS294-158 Deep Unsupervised Learning Spring 2018](https://sites.google.com/view/berkeley-cs294-158-sp19/home?fbclid=IwAR0MPHcveiSGnD8X8860hixqvVFr0dtOsOQ3udtVoIjv26-RVSTpAGGM9h8)
* [Eric Jang: Normalizing Flows Tutorial, Part 1: Distributions and Determinants](https://blog.evjang.com/2018/01/nf1.html)
* [Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#what-is-normalizing-flows)
* [CS 236: Deep Generative Models | Fall 2018-2019](https://deepgenerativemodels.github.io/)
# <span style="color:yellow"> Maximum Likelihood</span>
* https://sander.ai/2020/09/01/typicality.html
* https://www.inference.vc/maximum-likelihood-for-representation-learning-2/
# <span style="color:yellow"> 1. Autoregressive</span>
* Classical
	* RNN, e.g char-RNN: sequential
* Masking-based: parallelized
	* 1D masked convolution
		* limited receptive field —> dilated convolution w. exponential dilation
		* implementation: padding instead of masking kernel
	* 2D (PixelCNN)
		* blind spot —> gated PixelCNN: vertical + horizontal stack 
		* gated resnet block
		* PixelCNN++: beyond softmax —> better generalization
* (Masked) attention:
	* unlimited receptive field in a single layer —> slow
	* O(1) param scaling wrt data dimension (e.g., v.s MLP)
	* v.s convnet: limited receptive field —> hard to capture long-range
	* v.s RNN: parallelized
* autoregressive *ordering matters*
* Pros & cons:
	* pros: fast evaluation of p(x), great compression performance, good samples with careful design of dependence structure
	* cons: slow sampling, discrete data only
    
### Compression
* prefix-free ~ binary tries
* Shannon, Kraft - McMillan: $H(X) <= l_a(C)$ = average code length
* Huffman: 
	* uses <= H(X) + 1 bits per symbol on average
	* optimal lossless prefix-free code
* approximate p by p^: l_a = KL(p || p^) + H(p)
* conditional entropy:
	* H(X | C): average entropy over all contexts
	* AR models do conditional entropy
# <span style="color:yellow">2. Flow Models</span>
* histogram, GMM: simplistic
* flows can be composed: not possible with discrete AR
* how to use NN to model density —> shift perspective to CDF
* sampling: noise to data
    * $z$ ~ $U(0, 1)$ (or any base distribution - noise)
	* pass z through inverse CDF $f^{-1}(z)$ (CDF is invertible, differentiable from X to [0, 1])
* training: data to noise
	* flow is invertible —> x and z have same dimension
	* ML training: p(x) from p(z):  x —> z —> p(z) —> p(x)
	* —> challenges: determinant of Jacobian —> diagonal
* simple flows: affine flow (multivariate Gaussian), element-wise flow
* RealNVP
	* first half: identity
	* second half: parametrized (by first half) element-wise transformation
	* insights:
		* Bayes net, e.g. AR flows: sampling process of a Bayes net is a flow
		* DAG structure —> triangular Jacobian when vars are topological sorted
		* inference x —> z: fully parallelized
		* sampling z —> x:
* generalization of AR (continuous vs discrete, being stacked)
* flows for discrete data: 
	* dequantization
	* fitting lower bound of the true (discrete) distribution
# <span style="color:yellow">3. Latent Variables</span>
# Math block
# $\begin{aligned}
# E[logp(x|z) - logq(z|NN(x)) + logp(z)] 
# \\ w.r.t z \~\ q(z|NN(x))
# \\ core problem: optimize E[logq(z|NN(x))] w.r.t z \~{} q(z|NN(x))
# \end{aligned}$
### Maximize VLB 
$E[logp(x|z) - logq(z|NN(x)) + logp(z)]$ w.r.t $z$ ~ $q(z|NN(x))$

—> core problem: optimize $E[logq(z|NN(x))]$ w.r.t $z$ ~ $q(z|NN(x))$
* in SGD: optimize theta w.r.t data sampled from a true dist p_data
* this case: z is sampled from q(z | NN(x)) !!!
* logp(x|z),  logp(z) : easy
### Wake-sleep algorithm
* NOTE:
	* minimize KL(q || p) —> sample from q(z)
	* if minimize KL(p || q) —> sample from p(z | x) (but unknown!!!)
* Trick: 
	* generate z ~ p(z), 
	* generate x ~ p(x|z) ( caveat: p_model , not true p_data )
	* —> 
* Problem:
### References
* https://jmtomczak.github.io/blog/9/9_hierarchical_lvm_p1.html
* 
* https://sander.ai/2022/01/31/diffusion.html
# <span style="color:yellow">4. Diffusion - Score-based Models</span>
* https://jmtomczak.github.io/blog/10/10_ddgms_lvm_p2.html
* https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
* https://scorebasedgenerativemodeling.github.io/
# <span style="color:yellow">4. General Maths</span>
* log-derivative trick:
    * https://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/
    * https://andrewcharlesjones.github.io/journal/log-derivative.html