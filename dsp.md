### concepts
* time / freq: scale-free
* digital frequency: like normalized freq
* power: average energy over a period
* finite signals: embedded into infinite signals (periodic or aperiodic)
* Hilbert space 
	* complete inner product space
	* finite signals
	* periodic signals
	* aperiodic infinite signals: ~ finite energy signals l_2(Z)
		* Note: space of finite support signals not Hilbert (not complete)
### LTI
* Linearity: e.g. multitrack recording
* Time-invariance: system’s behavior is independent of the time it’s turned on
* —> completely characterized by its IR h[n]
* causal (system): h[n] = 0 for n < 0
* stable (filter): bounded l_1(h[n])
* FIR
	* 
* IIR
	* difference equation (AR?): y[n] = lambda * y[n-1] + (1 - lambda) * x[n]
	* —> LTI if zero initial
	* limited memory (v.s moving average)
	* can be causal or anti-causal depending on re-arrangement —> ROC analysis
	* leaky integrator
	* how to obtain IR (time / freq) in general case? —> z-transform
* complex exponentials are eigenfunctions of LTI systems
* realizable filters: FIR, leaky integrator etc.
### Fourier
* DFT
* DFS
* DTFT
### z transform
* linearity
* time-shift
* invertible up to a causality specification —> different RoC
* solving CCDE:
	* Y(z) = H(z) X(z)
	* H(z): transfer function (rational function) is z-transform of IR
	* H(z) evaluated on unit circle is the freq response
	* RoC defines causality / anti-causality
* ROC
	* region of absolute convergence
	* circular symmetric
	* ROC for finite support sequences is the entire C plane
	* ROC for causal sequences: outside of disk
	* ROC for anti-causal sequences: inside disk
	* a system is stable if ROC of transfer function includes unit circle 
		* —> for causal: all poles must be inside unit circle
		* —> for anti-causal: all poles must be outside unit circle
* Pole-zero plot: