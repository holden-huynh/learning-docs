* DL: unstructured environment
* RL: mechanism for behavior
## 1. Imitation Learning
Learn policies from examples
### Formalism
- sample $o_t$ from $p_{data}$
- sample action $a_t$ from optimal policy $\pi^*(a_t|o_t)$ (supervised)
- supervised learning: ignore dependencies between $o_t$
- use sampled data to train $\pi_\theta(a|o)$
- -> inference: run $\pi_\theta(a|o)$ -> $p_{\pi_\theta}(o_t)$
Problem: 
* $p_{data}(o_t)$ and $p_{\pi_\theta}(o_t)$  discrepancy
* Causual confusion
### Hacks
#### 1. Sample from more stable trajectories, not fit better

__DAGGER (Dataset Aggregation) algorithm__
* collect training data from $p_{\pi_\theta}(o_t)$ instead of $p_{data}(o_t)$
* how? just run $\pi_\theta(a_t|o_t)$
* need humans to label data
#### 2. Instead of more data, fitting better?
Why fail to fit expert?
__Fit expert is hard__

__Non-Markovian behavior:__ 
* $\pi(a_t|o_t)$ behavior depends on current observation, same observations same bahavior regardless of what happened before
* Include history exacerbate causual confusion?
* Does DAGGER mitigate causual confusion?

__Multimodal behavior__
* actions are discrete but modeled by MLE with continuous distributions, e.g. Normal. Solutions:
    * Mixture of Gaussians: easy to implement
    * Latent variables: theoretically cool, hard to implement
    * Autoregressive discretization: balanced
### Demo: drone in forest trails
* 10 layers convnet
* 20k images
### Imitation learning: what's the problem?
* humans provide data
* humans not good at proving some actions
- -> can machine learn autonomously?
- -> __optimize expected reward / cost__, not imitating
    - e.g. $\underset{\theta}{min}\left( \underset{\pi(a|s)}{E}\underset{p(s'|s,a)}{E}[\delta(eaten-by-tiger)] \right)$
    - sequence: $\underset{\theta}{min}\left( \underset{s1:T,a1:T)}{E}[\sum^T\delta(s_t=eaten-by-tiger)] \right)$
- -> RL: replace $\delta$ by some cost function $c(s,a)$
    - $\underset{\theta}{min}\left( \underset{s1:T,a1:T)}{E}[\sum^Tc(s_t,a_t)] \right)$
### A cost for imitation?
* log loss: $r(s,a)=\log p(a=\pi^*(a)|s)$
* zero-one loss: $r(s,a)=I(a=\pi^*(a))$
-> DAGGER actually optimizes zero-one loss
## 2. Intro to RL
### 2.1. Concepts
#### Definitions
* Markov chain
* MDP, partial observed MDP
#### Objective
* Once nailed down policy $\pi$, extended state space $(s,a)$ forms a Markov chain
* Marginal state-action distribution

In RL, always deal with __expectations__ -> expectation of non-smooth functions can be smooth!
#### Structure of RL algorithms:
* generate samples
* fit a model / estimate return
* improve policy
### 2.2. Details
#### How to deal with expectations