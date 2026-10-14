"""
Infinite-horizon value iteration (Section 4 of the handout).

Student code goes here. Complete the functions marked TODO X.X. Do not
change function names, argument lists, or return types.
"""

import torch


class ValueIteration:
    def __init__(self, mdp):
        self.R = mdp.R  # (nStates, nActions)
        self.P = mdp.P  # (nStates, nActions, nStates), P[s, a, s'] = Pr(s' | s, a)
        self.discount = mdp.discount
        self.nStates = mdp.nStates
        self.nActions = mdp.nActions
        self.device = mdp.device

    def computeVfromQ(self, Q):
        """
        TODO 4.1.1
        V(s) = max_a Q(s, a)

        Parameter Q: Tensor of shape (nStates, nActions)
        Returns: Tensor of shape (nStates,)
        """
        return Q.max(dim=1).values

    def computeQfromV(self, V):
        """
        TODO 4.1.2
        Q(s, a) = R(s, a) + discount * E_{s' ~ P(.|s,a)}[V(s')]

        Parameter V: Tensor of shape (nStates,)
        Returns: Tensor of shape (nStates, nActions)
        """
        # P @ V contracts over s', giving E_{s'~P(.|s,a)}[V(s')] of shape (S, A).
        return self.R + self.discount * (self.P @ V)

    def extractMaxPifromQ(self, Q):
        """
        TODO 4.1.3
        pi(s) = argmax_a Q(s, a)

        Break ties toward the lowest action index (UP=0), as torch.argmax
        does -- e.g. at the goal state, where both actions have Q = 0.

        Parameter Q: Tensor of shape (nStates, nActions)
        Returns: LongTensor of shape (nStates,)
        """
        return Q.argmax(dim=1)

    def valueIterationStep(self, Q):
        """
        TODO 4.1.4
        One Bellman optimality backup:

            Q^{k+1}(s, a) = R(s, a) + discount * E_{s'~P(.|s,a)}[max_a' Q^k(s', a')]

        Parameter Q: Q^k, Tensor of shape (nStates, nActions)
        Returns: Q^{k+1}, Tensor of shape (nStates, nActions)
        """
        # Greedy value estimate from Q^k, then one Bellman backup.
        return self.computeQfromV(self.computeVfromQ(Q))

    def valueIteration(self, initialQ, tolerance=0.01, max_iterations=10000):
        """
        TODO 4.1.5
        Repeatedly apply valueIterationStep, starting from initialQ, until
        ||Q^{k+1} - Q^k||_inf <= tolerance (or max_iterations is reached).
        Then extract the greedy policy and value function from the final Q.

        Parameter initialQ: Tensor of shape (nStates, nActions)
        Parameter tolerance: float >= 0 (default: 0.01)
        Parameter max_iterations: safety cap on the number of updates

        Returns: (pi, V, iterId, epsilon)
          pi: LongTensor of shape (nStates,), the converged greedy policy
          V: Tensor of shape (nStates,), the converged value function
          iterId: number of valueIterationStep updates performed
          epsilon: the final ||Q^{k+1} - Q^k||_inf
        """
        Q = initialQ
        iterId = 0
        epsilon = float("inf")

        while iterId < max_iterations:
            Q_next = self.valueIterationStep(Q)
            iterId += 1
            epsilon = (Q_next - Q).abs().max().item()  # ||Q^{k+1} - Q^k||_inf
            Q = Q_next
            if epsilon <= tolerance:
                break

        pi = self.extractMaxPifromQ(Q)
        V = self.computeVfromQ(Q)
        return pi, V, iterId, epsilon
