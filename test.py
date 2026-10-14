"""
Basic tests for DP.py, VI.py, and Qlearning.py.

These tests only check that each TODO function runs and returns outputs
with the expected shapes/types. They do NOT check that your values are
correct -- passing them does not mean your implementation is right. Inspect
the value functions, policies, and learning curves yourself (Section 6).

Run with: python test.py
"""

import torch

from DP import DynamicProgramming
from MDP import UP, DOWN, build_default_balloon_mdp
from Qlearning import QLearning
from VI import ValueIteration

# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

_PASS = 0


def _banner(title):
    width = 64
    print()
    print("=" * width)
    print(f" {title}")
    print("=" * width)


def _section(title):
    print()
    print(f"-- {title}")


def _ok(msg):
    global _PASS
    _PASS += 1
    print(f"  [PASS] {msg}")


def _summary(suite_name):
    print()
    print("-" * 64)
    print(f" {suite_name}: {_PASS} checks passed")
    print("-" * 64)


def _check_tensor(name, x, shape, dtype=None):
    """Assert x is a Tensor with the given shape (and dtype), with a fix-it hint."""
    assert isinstance(x, torch.Tensor), (
        f"{name}: expected output type torch.Tensor, got {type(x).__name__} "
        f"-- change output into 'torch.Tensor'"
    )
    assert tuple(x.shape) == tuple(shape), (
        f"{name}: expected output shape {tuple(shape)}, got {tuple(x.shape)} "
        f"-- change output shape into {tuple(shape)}"
    )
    if dtype is not None:
        assert x.dtype == dtype, (
            f"{name}: expected output dtype {dtype}, got {x.dtype} "
            f"-- change output dtype into {dtype} (e.g. .long())"
        )


def _check_type(name, x, typ, hint):
    """Assert x is a Python `typ` (not a Tensor), with a fix-it hint."""
    assert isinstance(x, typ) and not isinstance(x, bool), (
        f"{name}: expected output type {typ.__name__}, got {type(x).__name__} "
        f"-- change output into '{typ.__name__}' ({hint})"
    )



# ---------------------------------------------------------------------------
# Dynamic programming (finite horizon)
# ---------------------------------------------------------------------------


def test_dp():
    global _PASS
    _PASS = 0
    _banner("Dynamic Programming (finite horizon) sanity tests")

    horizon = 8
    mdp = build_default_balloon_mdp(horizon=horizon, device=torch.device("cpu"))
    dp = DynamicProgramming(mdp)

    _section("computeQfromV")
    Q = dp.computeQfromV(torch.randn(mdp.nStates))
    _check_tensor("computeQfromV", Q, (mdp.nStates, mdp.nActions))
    _ok("returns a Tensor of shape (nStates, nActions)")

    _section("dynamicProgramming")
    pi, V = dp.dynamicProgramming()
    _check_tensor("dynamicProgramming (pi)", pi, (horizon, mdp.nStates), torch.long)
    _check_tensor("dynamicProgramming (V)", V, (horizon, mdp.nStates))
    _ok("pi is a LongTensor and V a Tensor, both of shape (horizon, nStates)")

    _summary("DP suite")


# ---------------------------------------------------------------------------
# Value iteration (infinite horizon)
# ---------------------------------------------------------------------------


def test_vi():
    global _PASS
    _PASS = 0
    _banner("Value Iteration (infinite horizon) sanity tests")

    mdp = build_default_balloon_mdp(discount=0.99, device=torch.device("cpu"))
    vi = ValueIteration(mdp)

    Q_rand = torch.randn(mdp.nStates, mdp.nActions)
    V_rand = torch.randn(mdp.nStates)

    _section("computeVfromQ / computeQfromV / extractMaxPifromQ")
    _check_tensor("computeVfromQ", vi.computeVfromQ(Q_rand), (mdp.nStates,))
    _ok("computeVfromQ returns shape (nStates,)")

    _check_tensor("computeQfromV", vi.computeQfromV(V_rand), (mdp.nStates, mdp.nActions))
    _ok("computeQfromV returns shape (nStates, nActions)")

    _check_tensor("extractMaxPifromQ", vi.extractMaxPifromQ(Q_rand), (mdp.nStates,), torch.long)
    _ok("extractMaxPifromQ returns a LongTensor of shape (nStates,)")

    _section("valueIterationStep")
    _check_tensor("valueIterationStep", vi.valueIterationStep(Q_rand), (mdp.nStates, mdp.nActions))
    _ok("returns shape (nStates, nActions)")

    _section("valueIteration")
    initialQ = torch.zeros(mdp.nStates, mdp.nActions)
    pi, V, iterId, epsilon = vi.valueIteration(initialQ, tolerance=1e-4)
    _check_tensor("valueIteration (pi)", pi, (mdp.nStates,), torch.long)
    _check_tensor("valueIteration (V)", V, (mdp.nStates,))
    _check_type("valueIteration (iterId)", iterId, int, "e.g. int(x) or x.item()")
    _check_type("valueIteration (epsilon)", epsilon, float, "e.g. float(x) or x.item()")
    _ok("returns (pi, V, iterId, epsilon) with the right shapes/types")

    _summary("VI suite")


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------


def test_qlearning():
    global _PASS
    _PASS = 0
    _banner("Q-learning sanity tests")

    mdp = build_default_balloon_mdp(discount=0.99, device=torch.device("cpu"))
    ql = QLearning(mdp)

    _section("epsilonGreedy")
    Q = torch.randn(mdp.nStates, mdp.nActions)
    for eps in (0.0, 1.0):
        a = ql.epsilonGreedy(Q, 0, epsilon=eps)
        _check_type("epsilonGreedy", a, int, "e.g. int(x) or x.item()")
        assert 0 <= a < mdp.nActions, f"epsilonGreedy: action {a} is out of range [0, {mdp.nActions})"
    _ok("returns an int action in [0, nActions)")

    _section("qLearningUpdate")
    Q = torch.zeros(mdp.nStates, mdp.nActions)
    for done in (False, True):
        Q_out = ql.qLearningUpdate(Q, state=0, action=1, reward=1.0, next_state=5, done=done, alpha=0.5)
        _check_tensor("qLearningUpdate", Q_out, (mdp.nStates, mdp.nActions))
    _ok("returns a Tensor of shape (nStates, nActions)")

    _section("qLearning (full training loop)")
    num_episodes = 5
    initialQ = torch.zeros(mdp.nStates, mdp.nActions)
    Q_learned, pi_learned, episode_rewards = ql.qLearning(
        initialQ, num_episodes=num_episodes, alpha=0.5, epsilon=0.2
    )
    _check_tensor("qLearning (Q)", Q_learned, (mdp.nStates, mdp.nActions))
    _check_tensor("qLearning (pi)", pi_learned, (mdp.nStates,), torch.long)
    assert isinstance(episode_rewards, list), (
        f"qLearning (episode_rewards): expected output type list, got "
        f"{type(episode_rewards).__name__} -- change output into 'list'"
    )
    assert len(episode_rewards) == num_episodes, (
        f"qLearning (episode_rewards): expected length {num_episodes}, got {len(episode_rewards)}"
    )
    _ok("returns Q, pi, and per-episode rewards with the right shapes")

    _summary("Q-learning suite")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    _banner("Running all sanity tests")
    test_dp()
    test_vi()
    test_qlearning()
    print()
    print("=" * 64)
    print(" ALL SUITES PASSED")
    print("=" * 64)


if __name__ == "__main__":
    main()
