"""
The balloon navigation gridworld described in Section 2 of the handout.

This file is fully implemented -- you do not need to modify it. DP.py,
VI.py, and Qlearning.py all consume a `BalloonMDP` instance built here.
"""

import torch

UP, DOWN = 0, 1
ACTIONS = (UP, DOWN)
ACTION_NAMES = {UP: "UP", DOWN: "DOWN"}
ACTION_DELTAS = {UP: (0, 1), DOWN: (0, -1)}

LEFT, NO_WIND, RIGHT = -1, 0, 1


def release_wind_grid():
    """
    This is the wind grid we train our policy on. 
    """

    rows_top_to_bottom = [
        [NO_WIND, RIGHT, RIGHT, NO_WIND, LEFT, NO_WIND],      # j = 5
        [NO_WIND, NO_WIND, LEFT, NO_WIND, LEFT, NO_WIND],      # j = 4
        [NO_WIND, NO_WIND, LEFT, NO_WIND, LEFT, NO_WIND],   # j = 3
        [NO_WIND, NO_WIND, LEFT, NO_WIND, RIGHT, RIGHT],  # j = 2
        [LEFT, NO_WIND, LEFT, RIGHT, NO_WIND, NO_WIND], # j = 1
        [NO_WIND, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT], # j = 0
    ] 

    grid_size = len(rows_top_to_bottom)
    wind = torch.zeros((grid_size, grid_size), dtype=torch.long)
    for row_from_top, row in enumerate(rows_top_to_bottom):
        j = grid_size - 1 - row_from_top
        for i, w in enumerate(row):
            wind[i, j] = w
    return wind



class BalloonMDP:
    """
    States are encoded as s = i * grid_size + j, where i is the horizontal
    coordinate and j is the vertical coordinate (both zero-indexed, j=0 at
    the bottom of the grid). Actions are UP (0) and DOWN (1).

    At every step the balloon first applies its vertical action, then the
    wind at the resulting cell pushes it horizontally; both moves are
    clipped to stay on the grid (Section 2.3). The goal state is absorbing:
    once reached, both actions self-loop back to the goal with 0 reward, so
    no further reward can be earned (Section 2.5).

    P and R (used by DP.py/VI.py) describe the un-truncated MDP and are not
    affected by max_episode_steps. Episode truncation is purely a property
    of the reset()/step() sampling interface used by Qlearning.py: if a
    policy (e.g. an under-trained or badly exploring one) hasn't reached the
    goal within max_episode_steps transitions, step() ends the episode with
    a large negative timeout_reward instead of letting it run forever. This
    matters if you redesign the map in a way that makes some states dead
    ends or the goal hard to reach.

    Attributes:
      P: transition tensor, shape (nStates, nActions, nStates)
      R: reward tensor, shape (nStates, nActions)
      discount / horizon: exactly one is not None, matching the setting
        (infinite-horizon discounted vs. finite-horizon) you're using it for
    """

    def __init__(
        self,
        wind,
        initial_coord,
        goal_coord,
        goal_reward=20.0,
        max_episode_steps=50,
        timeout_reward=-20.0,
        discount=None,
        horizon=None,
        device="cpu",
    ):
        assert wind.dim() == 2 and wind.shape[0] == wind.shape[1], (
            "wind must be a square (grid_size, grid_size) tensor"
        )
        assert not (discount is None and horizon is None), (
            "Can't have both discount and horizon be None"
        )

        self.grid_size = wind.shape[0]
        self.nStates = self.grid_size * self.grid_size
        self.nActions = len(ACTIONS)
        self.device = device

        self.wind = wind.to(device=self.device, dtype=torch.long)
        self.initial_state = self.coord_to_state(*initial_coord)
        self.goal_state = self.coord_to_state(*goal_coord)
        self.step_reward = self._release_state_to_reward().reshape(-1)
        self.goal_reward = goal_reward
        self.max_episode_steps = max_episode_steps
        self.timeout_reward = timeout_reward
        self.discount = discount
        self.horizon = horizon

        self.P, self.R = self._build_dynamics()
        self.current_state = None
        self.episode_step_count = 0

    # ---- coordinate <-> state-index helpers ----
    def coord_to_state(self, i, j):
        return i * self.grid_size + j

    def state_to_coord(self, s):
        return s // self.grid_size, s % self.grid_size

    def is_goal(self, s):
        return s == self.goal_state

    def _release_state_to_reward(self):
        """
        This is the state-to-reward mapping we train our policy on.
        """
        rows_top_to_bottom = [
            [-1, -1, -1, -1, -1, 20],      # j = 5
            [-1, -1, -1, -1, -1, -1],      # j = 4
            [-1, -1, -1, -1, -1, -1],      # j = 3
            [-1, -1, -1, -1, -1, -1],      # j = 2
            [-1, -1, -1, -1, -1, -1],      # j = 1
            [-1, -1, -5, -5, -5, -5],      # j = 0
        ]

        grid_size = len(rows_top_to_bottom)
        reward = torch.zeros((grid_size, grid_size), dtype=torch.float32)
        for row_from_top, row in enumerate(rows_top_to_bottom):
            j = grid_size - 1 - row_from_top
            for i, r in enumerate(row):
                reward[i, j] = r
        return reward

    # ---- dynamics table construction (vectorized over all states/actions,
    # so this is cheap to rebuild on a GPU for a larger grid) ----
    def _build_dynamics(self):
        g = self.grid_size
        ii, jj = torch.meshgrid(
            torch.arange(g, device=self.device),
            torch.arange(g, device=self.device),
            indexing="ij",
        )  # ii[i, j] = i, jj[i, j] = j, each shape (g, g)
        src_states = (ii * g + jj).reshape(-1)  # (nStates,)

        P = torch.zeros((self.nStates, self.nActions, self.nStates), device=self.device)
        R = torch.zeros((self.nStates, self.nActions), device=self.device)

        for a in ACTIONS:
            di, dj = ACTION_DELTAS[a]
            i_tilde = (ii + di).clamp(0, g - 1)
            j_tilde = (jj + dj).clamp(0, g - 1)
            wind_dir = self.wind[i_tilde, j_tilde]
            i_next = (i_tilde + wind_dir).clamp(0, g - 1)
            j_next = j_tilde
            next_state = (i_next * g + j_next).reshape(-1)  # (nStates,)

            reached_goal = next_state == self.goal_state
            reward_a = torch.where(
                reached_goal,
                torch.full_like(next_state, self.goal_reward, dtype=torch.float32),
                self.step_reward[next_state],
            )

            P[src_states, a, next_state] = 1.0
            R[src_states, a] = reward_a

        # The goal is absorbing: both actions self-loop with 0 reward.
        P[self.goal_state, :, :] = 0.0
        P[self.goal_state, :, self.goal_state] = 1.0
        R[self.goal_state, :] = 0.0

        return P, R

    # ---- single-transition dynamics, used by step() below ----
    def apply_wind_dynamics(self, i, j, a):
        """Returns (i_next, j_next) for the action-then-wind dynamics of Section 2.3."""
        g = self.grid_size
        di, dj = ACTION_DELTAS[a]
        i_tilde = min(max(i + di, 0), g - 1)
        j_tilde = min(max(j + dj, 0), g - 1)
        wind_dir = int(self.wind[i_tilde, j_tilde].item())
        i_next = min(max(i_tilde + wind_dir, 0), g - 1)
        j_next = j_tilde
        return i_next, j_next

    # ---- sampling interface for Q-learning: reset()/step() only, no P/R ----
    def reset(self):
        self.current_state = self.initial_state
        self.episode_step_count = 0
        return self.current_state

    def step(self, action):
        """
        Advances the environment by one transition from self.current_state.

        An episode also ends -- with reward self.timeout_reward instead of
        self.step_reward -- if the goal hasn't been reached within
        self.max_episode_steps transitions since the last reset(). This
        keeps episodes bounded even if a policy (or a redesigned map) can
        get stuck without ever reaching the goal.

        Returns (next_state, reward, done).
        """
        assert self.current_state is not None, "Call reset() before step()."
        s = self.current_state
        self.episode_step_count += 1
        if self.is_goal(s):
            next_state, reward, done = s, 0.0, True
        else:
            i, j = self.state_to_coord(s)
            i_next, j_next = self.apply_wind_dynamics(i, j, action)
            next_state = self.coord_to_state(i_next, j_next)
            if next_state == self.goal_state:
                reward, done = self.goal_reward, True
            elif self.episode_step_count >= self.max_episode_steps:
                reward, done = self.timeout_reward, True
            else:
                reward = float(self.step_reward[next_state].item())
                done = False
        self.current_state = next_state
        return next_state, reward, done


def build_default_balloon_mdp(discount=None, horizon=None, device="cpu"):
    """
    Builds the release-version 6x6 balloon MDP using the custom wind field,
    initial state s(1,0), and goal state s(5,5).
    """
    wind = release_wind_grid()
    return BalloonMDP(
        wind,
        initial_coord=(1, 0),
        goal_coord=(5, 5),
        discount=discount,
        horizon=horizon,
        device=device,
    )
