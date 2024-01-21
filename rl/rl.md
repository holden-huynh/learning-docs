## Materials
- [Deep RL Bootcamp - Lectures](https://sites.google.com/view/deep-rl-bootcamp/lectures?fbclid=IwAR1L_GTpvUsHhYgTx58Lwc0DwwLGMh_oUsfzx2-cx-Nf8DrmsYmO2th8PtI)
- RLSS: [Reinforcement Learning - VideoLectures.NET](http://videolectures.net/deeplearning2017_pineau_reinforcement_learning/)
- UCB: [CS294-112 Fa18 - YouTube](https://www.youtube.com/playlist?list=PLkFD6_40KJIxJMR-j5A1mkxK26gh_qg37)
- Deepmind: [Advanced Deep Learning & Reinforcement Learning - YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)
- David Silver: [Introduction to reinforcement learning - YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- [Deep Reinforcement Learning Course](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)
- https://spinningup.openai.com/en/latest/spinningup/keypapers.html
- [Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
- https://www.stat.berkeley.edu/~bartlett/courses/2014fall-cs294stat260/readings.html
## 1. Basics
* agent, environment
	* agent -> env: action
	* env -> agent: state + reward
* challenges:
	* need access to environment
	* jointly learning and planning from correlated samples
	* data distribution changes with action choice
### 1.1. MDP
* states: $S_i$ (can be infinite / continuous) -> __Markov property__
* Observations -> __non Markov property__ (see Graphical Models basics)
* actions: $A_i$ (can be infinite / continuous)
* dynamics of the env: $T(s, a, s’) = Pr(s’ | a, s)$   (*probabilistic*)
* rewards: $R(s, a, s’)$ (deterministic - dependent on both source and target states)
* trajectory: sequence of states
* discount factor
* horizon

#### Goal 
* find policy to maximize expected discounted reward
* episodic: $U_t$ = sum of rewards
* continuing: discount factor to avoid infinite return, tradeoff immediate and long-term rewards
### 1.2. Policy & Value function
* Policy: action selection strategy at every state (*deterministic* or *probabilistic*). not abt env, have to learn
	* $\pi: S —> A$
	* $\pi(s, a) = Pr(a_t = a | s_t = s)$ (or $\pi(o,a)$ if not __fully observed__)
	* —> find policy that maximizes expected total reward
	* return: sum of reward for a trajectory
	* value: __expected__ sum of reward (over policy and/or transition)
	* value function: $V_\pi(s)$ expected sum of reward from state $s_0 = s$
	* —> optimal value $V^*(s)$
* Value
	* $V^*(s)$: sum of (expected) discounted rewards when starting from s and acting optimally (i.e. optimal $\pi$)
### 1.3. Value iteration
* algo: for k: 0 —> H (deterministic policy)
    * $V^*_k(s)$: optimal value func when k time steps left —> Bellman eq.:
    * $V^*_k(s) = \underset{a}{max} \left(\underset{s':P(s'|s,a)}{E}[R(s,s',a) + \gamma V^*_{k-1}(s')]\right)$&nbsp;&nbsp;&nbsp;&nbsp;(1.1)
    * $\pi^*_k(s) = \underset{a}{argmax} \left(\underset{s':P(s'|s,a)}{E}[R(s,s',a) + \gamma V^*_{k-1}(s')]\right)$ &nbsp;&nbsp;&nbsp;&nbsp;(1.2)
* $V^*_k(s)$ converges to $V^*(s)$ as $k —> \inf$
    * run value iteration till convergence
    * —> optimal policy for infinite horizon $\pi^*(s)$
* Q function
    * $Q^*(s, a)$: expected utility starting at s, taking action a, and thereafter acting optimally (i.e. a needs not to be optimal)
    * $V^*(s) = max_a Q^*(s, a)$
    * Bellman equation: $Q^*_k(s, a) = \underset{s':P(s'|s,a)}{E}[R(s,s',a) + \gamma \underset{a'} {max} Q^*_{k-1}(s',a')]$&nbsp;&nbsp;&nbsp;&nbsp;(1.3)
    * —> Q-value iteration: $Q^*_k(s, a)$ 
### 1.4. Policy iteration
* Policy evaluation: $V^\pi_k(s)$ value of the policy
    * sum of expected discounted reward
    * $V^\pi_k(s) = \underset{s':P(s'|s,\pi(s))}{E}[R(s,s',\pi(s)) + \gamma V^\pi_{k-1}(s')]$ 
    (action $a=\pi(s)$)&nbsp;&nbsp;&nbsp;&nbsp;(1.4)
    * stochastic policy: expectation over 2 distributions
* convergence: $V^\pi(s)$
* Policy iteration:
    * evaluate *current* policy: iterate until convergence ($V^\pi(s)$) OR solve a linear equation
    * Improve: find the best action according to 1-step look ahead
        * $\pi'(s) = \underset{a}{argmax}(\underset{s':P(s'|s,a)}{E}[R(s,s',a) + \gamma V^\pi(s')])$&nbsp;&nbsp;&nbsp;&nbsp;(1.5)
    * repeat until policy converges
    * under certain conditions, converges faster than value iteration
## 2. Sample-based approximation & fitted learning 
### Model-free method
* Model-based RL: reconstruct transition function, then use planning to find policy
* Model-free: relies on estimates to improve policy directly
### Limitations of exact methods
* access to dynamics model -> __sampling-based approximation__
* iterate over all states / actions: intractable -> __Q / V function fitting__
### 2.1. Tabular Q-learning 
* Off-policy method: decoupling sampling strategy from policy
* Algorithm:
    * sample action / state —> estimate of expectation (eq. 1.3)
    * incorporate old estimate —> running average of Q(s, a)
    * action sampling: (multi-arm bandit?)
        * random
        * greedy: a that maximizes $Q_k(s, a)$
        * $\epsilon$-greedy: decaying epsilon
* Properties
	* Q-learning converges to optimal policy even if acting suboptimally (e.g. greedy actions, or even randomly)
	* Caveats:
        * Have to explore enough
        * Have to eventually make lr small enough (but not too fast!)