"""
Command-line entry point for running DP, VI, or Q-learning on the balloon
gridworld, once DP.py / VI.py / Qlearning.py are implemented. Prints an
ASCII rendering of the resulting policy/value function -- useful for a
quick sanity check before visualize.py / the notebook exist.

Usage:
  python run.py --method dp --horizon 10
  python run.py --method vi --discount 0.9 --tolerance 0.01
  python run.py --method qlearning --discount 0.9 --num-episodes 500 --alpha 0.5 --epsilon 0.1
"""

import argparse
import random

import numpy as np
import torch

from DP import DynamicProgramming
from MDP import UP, build_default_balloon_mdp
from Qlearning import QLearning
from VI import ValueIteration

def seed_everything(seed):
    """
    Reset every RNG a Q-learning implementation might draw from. Called right
    before each Q-learning run here and in visualize.py, so the same settings
    and seed give identical results in both scripts even if a solution uses
    the global RNG instead of QLearning.generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# ASCII grid rendering
# ---------------------------------------------------------------------------


def _grid_lines(mdp, cell_fn):
    """
    cell_fn(state) -> short string for each state. Renders j=grid_size-1
    (top) down to j=0 (bottom), i=0 (left) to i=grid_size-1 (right), the
    same orientation as Figure 1.
    """
    lines = []
    for j in reversed(range(mdp.grid_size)):
        row = [cell_fn(mdp.coord_to_state(i, j)) for i in range(mdp.grid_size)]
        lines.append(" ".join(row))
    return lines


def _policy_grid(mdp, pi):
    def cell(s):
        if s == mdp.goal_state:
            return " G "
        arrow = "^" if int(pi[s].item()) == UP else "v"
        return f"S{arrow} " if s == mdp.initial_state else f" {arrow} "

    return _grid_lines(mdp, cell)


def _value_grid(mdp, V):
    return _grid_lines(mdp, lambda s: f"{V[s].item():6.1f}")


def _print_grid(title, lines):
    print(f"\n{title}")
    for line in lines:
        print(" ", line)


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------


def run_dp(args, mdp):
    dp = DynamicProgramming(mdp)
    pi, V = dp.dynamicProgramming()

    print(f"Finite-horizon dynamic programming: horizon = {args.horizon}")
    _print_grid("pi*_0 (policy at t=0)", _policy_grid(mdp, pi[0]))
    _print_grid("V*_0 (value at t=0)", _value_grid(mdp, V[0]))
    print(f"\nV*_0(initial state) = {V[0, mdp.initial_state].item():.2f}")

    if args.verbose:
        for t in range(args.horizon):
            _print_grid(f"pi*_{t}", _policy_grid(mdp, pi[t]))


def run_vi(args, mdp):
    vi = ValueIteration(mdp)
    initialQ = torch.zeros(mdp.nStates, mdp.nActions, device=mdp.device)
    pi, V, n_iters, eps = vi.valueIteration(initialQ, tolerance=args.tolerance)

    print(f"Value iteration: discount = {args.discount}, tolerance = {args.tolerance}")
    print(f"Converged after {n_iters} iterations (epsilon = {eps:.2e})")
    _print_grid("pi* (greedy policy)", _policy_grid(mdp, pi))
    _print_grid("V* (value function)", _value_grid(mdp, V))
    print(f"\nV*(initial state) = {V[mdp.initial_state].item():.2f}")


def run_qlearning(args, mdp):
    seed_everything(args.seed)
    ql = QLearning(mdp, seed=args.seed)
    initialQ = torch.zeros(mdp.nStates, mdp.nActions, device=mdp.device)
    Q, pi, episode_rewards = ql.qLearning(
        initialQ, num_episodes=args.num_episodes, alpha=args.alpha, epsilon=args.epsilon
    )

    print(
        f"Q-learning: discount = {args.discount}, episodes = {args.num_episodes}, "
        f"alpha = {args.alpha}, epsilon = {args.epsilon}"
    )
    window = min(50, len(episode_rewards))
    if window > 0:
        print(f"Average reward, first {window} episodes: {sum(episode_rewards[:window]) / window:.2f}")
        print(f"Average reward, last {window} episodes:  {sum(episode_rewards[-window:]) / window:.2f}")
    _print_grid("learned greedy policy", _policy_grid(mdp, pi))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(description="Run DP / VI / Q-learning on the balloon gridworld.")
    parser.add_argument("--method", required=True, choices=["dp", "vi", "qlearning"])
    parser.add_argument("--seed", type=int, default=0)

    dp_group = parser.add_argument_group("dp")
    dp_group.add_argument("--horizon", type=int, default=7, help="finite horizon H")
    dp_group.add_argument("--verbose", action="store_true", help="also print pi_t for every t")

    vi_group = parser.add_argument_group("vi")
    vi_group.add_argument("--tolerance", type=float, default=0.01)

    shared_group = parser.add_argument_group("vi & qlearning")
    shared_group.add_argument("--discount", type=float, default=0.9)

    ql_group = parser.add_argument_group("qlearning")
    ql_group.add_argument("--num-episodes", dest="num_episodes", type=int, default=1000)
    ql_group.add_argument("--alpha", type=float, default=0.5)
    ql_group.add_argument("--epsilon", type=float, default=0.1)

    return parser


def main():
    args = build_parser().parse_args()

    seed_everything(args.seed)
    device = "cpu"

    if args.method == "dp":
        mdp = build_default_balloon_mdp(horizon=args.horizon, device=device)
        run_dp(args, mdp)
    elif args.method == "vi":
        mdp = build_default_balloon_mdp(discount=args.discount, device=device)
        run_vi(args, mdp)
    elif args.method == "qlearning":
        mdp = build_default_balloon_mdp(discount=args.discount, device=device)
        run_qlearning(args, mdp)


if __name__ == "__main__":
    main()
