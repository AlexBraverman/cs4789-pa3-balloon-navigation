"""
Finite-horizon dynamic programming (Section 3 of the handout).

Student code goes here. Complete the functions marked TODO X.X. Do not
change function names, argument lists, or return types.
"""

import torch


class DynamicProgramming:
    def __init__(self, mdp):
        self.R = mdp.R  # (nStates, nActions)
        self.P = mdp.P  # (nStates, nActions, nStates), P[s, a, s'] = Pr(s' | s, a)
        self.horizon = mdp.horizon
        self.nStates = mdp.nStates
        self.nActions = mdp.nActions
        self.device = mdp.device

    def computeQfromV(self, V_next):
        """
        TODO 3.1.1
        Given V_{t+1}, return Q_t using the Bellman equation

            Q_t(s, a) = R(s, a) + E_{s' ~ P(.|s,a)}[V_{t+1}(s')]

        Parameter V_next: value function at time t+1
        Precondition: a Tensor of shape (nStates,)

        Returns: a Tensor of shape (nStates, nActions)
        """
        # E_{s'~P(.|s,a)}[V_{t+1}(s')] = sum_{s'} P[s, a, s'] * V_next[s']
        # P has shape (S, A, S') and V_next has shape (S',), so the matmul
        # contracts over s' and returns a (S, A) tensor.
        return self.R + self.P @ V_next

    def dynamicProgramming(self):
        """
        TODO 3.1.2
        Run the backward recursion of Section 3.3, starting from
        V*_H(s) = 0 for all s and working back to t = 0. Use
        self.computeQfromV at each step.

        Returns: (pi, V)
          pi: a LongTensor of shape (horizon, nStates), pi[t] = pi*_t, with
            ties broken toward the lowest action index (UP=0), as
            torch.argmax does
          V: a Tensor of shape (horizon, nStates), V[t] = V*_t
        """
        pi = torch.zeros((self.horizon, self.nStates), dtype=torch.long, device=self.device)
        V = torch.zeros((self.horizon, self.nStates), device=self.device)

        # Terminal condition: V*_H(s) = 0 for all s.
        V_next = torch.zeros(self.nStates, device=self.device)

        # Single backward pass t = H-1, ..., 0.
        for t in range(self.horizon - 1, -1, -1):
            Q_t = self.computeQfromV(V_next)   # (S, A)
            V[t] = Q_t.max(dim=1).values       # V*_t(s)  = max_a Q*_t(s, a)
            pi[t] = Q_t.argmax(dim=1)          # pi*_t(s) = argmax_a Q*_t(s, a)
            V_next = V[t]

        return pi, V
