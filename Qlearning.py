"""
Tabular Q-learning with epsilon-greedy exploration (Section 5 of the handout).

Student code goes here. Complete the functions marked TODO X.X. Do not
change function names, argument lists, or return types.

IMPORTANT: this file must not read mdp.P or mdp.R to perform Bellman
backups. Interact with the environment only through mdp.reset()/mdp.step().
"""

import torch


class QLearning:
    def __init__(self, mdp, seed=0):
        self.mdp = mdp
        self.discount = mdp.discount
        self.nStates = mdp.nStates
        self.nActions = mdp.nActions
        self.device = mdp.device

        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def epsilonGreedy(self, Q, state, epsilon):
        """
        TODO 5.1.1
        With probability epsilon, return a uniformly random action.
        Otherwise, return argmax_a Q(state, a).

        Use self.generator for all randomness, in this order:
          1. Always draw u = torch.rand(1, generator=self.generator,
             device=self.device) -- even when epsilon is 0.
          2. If u < epsilon, return a random action drawn with
             torch.randint(self.nActions, (1,), generator=self.generator,
             device=self.device), uniform over all actions (it may equal
             the greedy one).
        Break ties in argmax toward the lowest action index (UP=0), as
        torch.argmax does.

        Parameter Q: Tensor of shape (nStates, nActions)
        Parameter state: int, the current state
        Parameter epsilon: float in [0, 1]

        Returns: int, the selected action
        """
        # Always draw u first so the random stream is identical regardless of epsilon.
        u = torch.rand(1, generator=self.generator, device=self.device)
        if u.item() < epsilon:
            a = torch.randint(self.nActions, (1,), generator=self.generator, device=self.device)
            return int(a.item())
        return int(Q[state].argmax().item())

    def qLearningUpdate(self, Q, state, action, reward, next_state, done, alpha):
        """
        TODO 5.1.2
        Apply the one-step tabular Q-learning update (Section 5.3) to Q
        in place:

            target = reward                                     if done
                   = reward + discount * max_a' Q(next_state, a') otherwise
            Q(state, action) <- Q(state, action) + alpha * (target - Q(state, action))

        Parameter Q: Tensor of shape (nStates, nActions), modified in place
        Parameter state, action, next_state: ints
        Parameter reward: float
        Parameter done: bool, whether the episode ended on this transition
          (the goal was reached, or the episode timed out -- see mdp.step())
        Parameter alpha: float, the learning rate

        Returns: Q (the same tensor, after the update)
        """
        if done:
            # Terminal transition: no future return, target is just the reward.
            target = reward
        else:
            target = reward + self.discount * Q[next_state].max().item()
        # TD error, then in-place update of the single visited entry.
        Q[state, action] += alpha * (target - Q[state, action])
        return Q

    def qLearning(
        self,
        initialQ,
        num_episodes=1000,
        alpha=0.5,
        epsilon=0.1,
    ):
        """
        TODO 5.1.3
        Starting from initialQ, run num_episodes episodes of tabular
        Q-learning (Section 5.4):
          1. reset the environment (self.mdp.reset())
          2. select actions with self.epsilonGreedy
          3. observe the resulting transition (self.mdp.step())
          4. update the corresponding Q-value with self.qLearningUpdate
          5. continue until the episode terminates (done=True)

        self.mdp.step() already bounds every episode at
        self.mdp.max_episode_steps transitions -- returning done=True with
        a large negative reward if the goal hasn't been reached by then --
        so you don't need a separate step cap here.

        Parameter initialQ: Tensor of shape (nStates, nActions)
        Parameter num_episodes: int, number of training episodes
        Parameter alpha: float, learning rate
        Parameter epsilon: float in [0, 1], exploration rate

        Returns: (Q, pi, episode_rewards)
          Q: the learned Q-function, Tensor of shape (nStates, nActions)
          pi: the greedy policy w.r.t. Q, LongTensor of shape (nStates,),
            ties broken toward the lowest action index (as torch.argmax does)
          episode_rewards: list of length num_episodes of Python floats, the
            total (undiscounted) reward collected in each episode
        """
        Q = initialQ.clone()
        episode_rewards = []

        for _ in range(num_episodes):
            state = self.mdp.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.epsilonGreedy(Q, state, epsilon)
                next_state, reward, done = self.mdp.step(action)
                self.qLearningUpdate(Q, state, action, reward, next_state, done, alpha)
                total_reward += reward
                state = next_state
            episode_rewards.append(float(total_reward))

        pi = Q.argmax(dim=1)
        return Q, pi, episode_rewards
