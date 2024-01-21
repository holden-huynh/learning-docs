## env
* time_step_spec(): observation + reward
    * observation:
        * global
        * per_arm
        * num_actions
* action_spec()
* reset() -> first time_step
    * _reset()
* step(action) -> next time_step
    * _step():
* bandit env specific: 
    * _observe(): returns observation for _step(), _reset()
    * _apply_action(): reward for _step()

## agent
* 2 policies: policy, collect_policy
* accepts_per_arm_features=True/False: for __contextual only__
* _train()

## policy
* action(time_step) -> PolicyStep(action, state, info)
* accepts_per_arm_features=True/False: for __contextual only__
* _distribution()

## dynamic actions
* fixed max_num_actions
* network predicts for all max_num_actions, masking using mask
* mask:
    * 1st priority: observation_and_action_constraint_splitter: 
        * used with non-contextual, contextual non-per-arm
        * not allowed for per-arm bandits
    * 2nd priority: num_actions -> masked first num_actions: 
        * used with contextual (per-arm or not)
        * not allowed for non-contextual (BTS )
    * -> hard to user mask for per-arm contextual, have to handle manually

## contextual bandits
* reward pred: 
    * greedy_reward_pred agent - greedy reward pred policy
    * custom reward pred network
    * neural_linucb agent
    * neural e-greedy agent - e-greedy policy
* linear: 
    * LinUCBAgent(LinearBanditAgent) - LinearBanditPolicy, 
        * exploration_strategy=optimistic
    * LinTSAgent(LinearBanditAgent) - LinearBanditPolicy
        * exploration_strategy=sampling
        
## non-contextual bandits
* bernoulli TS agent - bernoulli TS policy, 
    * alpha, beta: List[Variable] (length= max_num_actions)
    * -> need to modify to support dynamic actions
* e-greedy