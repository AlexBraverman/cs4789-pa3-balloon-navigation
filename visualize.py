"""
Visualization utilities for the balloon gridworld (Section 6 of the handout).

Plot the environment, value functions, Q-functions, policies, and Q-learning
progress. Also provides experiment runners matching the Section 6 checklist.

Examples:
  python visualize.py --experiment all
  python visualize.py --experiment dp --save-dir figures
  python visualize.py --experiment vi
  python visualize.py --experiment qlearning

Hyperparameters are the DEFAULT_* constants below; edit them to change runs.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from DP import DynamicProgramming
from MDP import UP, ACTION_NAMES, build_default_balloon_mdp
from Qlearning import QLearning
from run import seed_everything
from VI import ValueIteration

# Default experiment settings from Section 6.
DEFAULT_DP_HORIZONS = (1,3,5,7,9)
DEFAULT_DP_TIME_HORIZON = 7
DEFAULT_VI_DISCOUNTS = (0.9, 0.99, 0.999)
DEFAULT_VI_TOLERANCE = 0.01
DEFAULT_QL_EPSILONS = (0.0, 0.1, 0.5)
DEFAULT_QL_EPISODES = 1000
DEFAULT_QL_ALPHA = 0.5
DEFAULT_QL_DISCOUNT = 0.9
DEFAULT_QL_EPSILON = 0.1
DEFAULT_QL_LOG_EVERY = 25


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=float)


def values_to_grid(mdp, values) -> np.ndarray:
    """
    Reshape a length-nStates vector into a (grid_size, grid_size) array
    indexed as grid[row_from_top, col], matching Figure 1 (j = grid_size-1
    at the top, i = 0 on the left).
    """
    g = mdp.grid_size
    vals = _to_numpy(values).reshape(-1)
    grid = np.zeros((g, g), dtype=float)
    for s in range(mdp.nStates):
        i, j = mdp.state_to_coord(s)
        row = g - 1 - j
        grid[row, i] = vals[s]
    return grid


def policy_to_grid(mdp, pi) -> np.ndarray:
    """Same layout as values_to_grid, but entries are action indices."""
    return values_to_grid(mdp, pi)


# ---------------------------------------------------------------------------
# Core plotting primitives
# ---------------------------------------------------------------------------


def _mark_terminals(ax, mdp, fontsize: int = 11):
    g = mdp.grid_size
    si, sj = mdp.state_to_coord(mdp.initial_state)
    gi, gj = mdp.state_to_coord(mdp.goal_state)
    ax.text(si, g - 1 - sj, "S", ha="center", va="center", fontsize=fontsize, fontweight="bold", color="darkorange")
    ax.text(gi, g - 1 - gj, "G", ha="center", va="center", fontsize=fontsize, fontweight="bold", color="royalblue")


def plot_environment(mdp, ax=None, title: str = "Balloon environment (wind)", save_path: Optional[str] = None):
    """Draw the wind field with S / G markers (Figure 1 orientation)."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    g = mdp.grid_size
    wind = mdp.wind.detach().cpu().numpy()
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(-0.5, g - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(g))
    ax.set_yticks(range(g))
    ax.set_xticklabels(range(g))
    ax.set_yticklabels(list(reversed(range(g))))
    ax.set_xlabel("i (horizontal)")
    ax.set_ylabel("j (vertical)")
    ax.set_title(title)
    ax.grid(True, which="major", color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)

    for i in range(g):
        for j in range(g):
            w = int(wind[i, j])
            row = g - 1 - j
            if w == 0:
                ax.plot(i, row, "o", color="0.55", markersize=4)
            else:
                ax.annotate(
                    "",
                    xy=(i + 0.28 * w, row),
                    xytext=(i - 0.28 * w, row),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.6),
                )

    _mark_terminals(ax, mdp)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    if created:
        return fig, ax
    return ax


def plot_value_function(
    mdp,
    V,
    ax=None,
    title: str = "Value function",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    annotate: bool = True,
    save_path: Optional[str] = None,
):
    """Heatmap of a length-nStates value vector."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure

    grid = values_to_grid(mdp, V)
    if vmin is None:
        vmin = float(np.min(grid))
    if vmax is None:
        vmax = float(np.max(grid))
    im = ax.imshow(grid, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(mdp.grid_size))
    ax.set_yticks(range(mdp.grid_size))
    ax.set_xticklabels(range(mdp.grid_size))
    ax.set_yticklabels(list(reversed(range(mdp.grid_size))))
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_title(title)
    if annotate:
        for r in range(mdp.grid_size):
            for c in range(mdp.grid_size):
                ax.text(c, r, f"{grid[r, c]:.1f}", ha="center", va="center", fontsize=7, color="w")
    _mark_terminals(ax, mdp, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    if created:
        return fig, ax
    return ax


def plot_trajectory(
    ax,
    mdp,
    states: Sequence,
    color: str = "#ff6b35",
    linewidth: float = 2.2,
    markersize: float = 6.5,
    annotate_steps: bool = True,
    zorder: int = 5,
):
    """
    Overlay a rolled-out state sequence as a polyline on a grid axes
    (same i / row layout as plot_policy and plot_environment).
    """
    g = mdp.grid_size
    xs, ys = [], []
    for s in states:
        i, j = mdp.state_to_coord(int(s))
        xs.append(float(i))
        ys.append(float(g - 1 - j))
    if not xs:
        return ax

    ax.plot(
        xs,
        ys,
        "-",
        color=color,
        linewidth=linewidth,
        zorder=zorder,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    ax.plot(
        xs,
        ys,
        "o",
        color=color,
        markersize=markersize,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=zorder + 1,
    )
    if annotate_steps:
        for t, (x, y) in enumerate(zip(xs, ys)):
            ax.text(
                x + 0.22,
                y - 0.22,
                str(t),
                color=color,
                fontsize=7,
                fontweight="bold",
                ha="left",
                va="top",
                zorder=zorder + 2,
            )
    return ax


def save_trajectory_gif(
    mdp,
    states: Sequence,
    save_path: str,
    pi=None,
    background_V=None,
    actions: Optional[Sequence] = None,
    rewards: Optional[Sequence] = None,
    title: str = "Trajectory",
    fps: float = 2.0,
    hold_final_frames: int = 3,
    dpi: int = 120,
):
    """
    Animate a rolled-out trajectory growing one state at a time over the
    policy (or wind) map. Writes an animated GIF via Pillow.
    """
    states = [int(s) for s in states]
    if not states:
        return None

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    g = mdp.grid_size
    n_path_frames = len(states)
    n_frames = n_path_frames + max(0, hold_final_frames)

    fig, ax = plt.subplots(figsize=(5, 4.5))

    def _frame_title(k: int) -> str:
        parts = [title, f"t={k}"]
        if actions is not None and k >= 1 and k - 1 < len(actions):
            parts.append(f"a={ACTION_NAMES[int(actions[k - 1])]}")
        if rewards is not None and k >= 1:
            ret = float(sum(rewards[:k]))
            parts.append(f"return={ret:.1f}")
        return " | ".join(parts)

    def _draw(frame_idx: int):
        ax.clear()
        k = min(frame_idx, n_path_frames - 1)
        prefix = states[: k + 1]
        frame_title = _frame_title(k)
        if pi is not None:
            plot_policy(
                mdp,
                pi,
                ax=ax,
                title=frame_title,
                background_V=background_V,
                trajectory=None,
            )
        else:
            plot_environment(mdp, ax=ax, title=frame_title)
        plot_trajectory(ax, mdp, prefix, annotate_steps=True)
        i, j = mdp.state_to_coord(prefix[-1])
        ax.plot(
            i,
            g - 1 - j,
            "o",
            color="white",
            markersize=11,
            markeredgecolor="#ff6b35",
            markeredgewidth=2.4,
            zorder=12,
        )

    anim = FuncAnimation(fig, _draw, frames=n_frames, interval=1000.0 / fps, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved {save_path}")
    return save_path


def plot_policy(
    mdp,
    pi,
    ax=None,
    title: str = "Policy",
    background_V=None,
    trajectory: Optional[Sequence] = None,
    save_path: Optional[str] = None,
):
    """Arrow plot of a deterministic policy (UP = ^, DOWN = v)."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure

    g = mdp.grid_size
    if background_V is not None:
        grid = values_to_grid(mdp, background_V)
        ax.imshow(grid, origin="upper", cmap="viridis", alpha=0.85)
    else:
        ax.set_xlim(-0.5, g - 0.5)
        ax.set_ylim(g - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_facecolor("#f7f7f7")

    pi_np = _to_numpy(pi).astype(int).reshape(-1)
    for s in range(mdp.nStates):
        if s == mdp.goal_state:
            continue
        i, j = mdp.state_to_coord(s)
        row = g - 1 - j
        a = int(pi_np[s])
        dy = 0.28 if a == UP else -0.28
        ax.annotate(
            "",
            xy=(i, row - dy),
            xytext=(i, row + dy * 0.15),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )

    if trajectory is not None:
        plot_trajectory(ax, mdp, trajectory)

    ax.set_xticks(range(g))
    ax.set_yticks(range(g))
    ax.set_xticklabels(range(g))
    ax.set_yticklabels(list(reversed(range(g))))
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_title(title)
    _mark_terminals(ax, mdp)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    if created:
        return fig, ax
    return ax


def plot_value_and_policy(
    mdp,
    V,
    pi,
    title: str = "",
    trajectory: Optional[Sequence] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Side-by-side value heatmap and policy arrows (optional trajectory overlay)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_value_function(mdp, V, ax=axes[0], title=f"{title} value".strip())
    plot_policy(
        mdp,
        pi,
        ax=axes[1],
        title=f"{title} policy".strip(),
        background_V=V,
        trajectory=trajectory,
    )
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    if show:
        plt.show()
    return fig, axes


def plot_q_function(
    mdp,
    Q,
    title: str = "Q-function",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Side-by-side heatmaps of Q(:, UP) and Q(:, DOWN)."""
    Q_np = _to_numpy(Q)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    vmin = float(Q_np.min())
    vmax = float(Q_np.max())
    for a, ax in enumerate(axes):
        plot_value_function(
            mdp,
            Q_np[:, a],
            ax=ax,
            title=f"{title}: {ACTION_NAMES[a]}",
            vmin=vmin,
            vmax=vmax,
        )
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    if show:
        plt.show()
    return fig, axes


def plot_learning_curve(
    episode_rewards: Sequence[float],
    window: int = 25,
    title: str = "Q-learning episode reward",
    ax=None,
    label: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Raw (faint) and moving-average episode returns."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    rewards = np.asarray(episode_rewards, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)
    ax.plot(episodes, rewards, color="0.75", linewidth=0.8, alpha=0.7)
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(rewards, kernel, mode="valid")
        ax.plot(
            np.arange(window, len(rewards) + 1),
            smooth,
            linewidth=2.0,
            label=label or f"moving avg (w={window})",
        )
    elif label is not None:
        ax.plot([], [], label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Undiscounted return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if label is not None or len(rewards) >= window:
        ax.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    if show and created:
        plt.show()
    if created:
        return fig, ax
    return ax


def plot_metric_curve(
    xs: Sequence[float],
    ys: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    ax=None,
    label: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    ax.plot(xs, ys, linewidth=2.0, marker="o", markersize=3, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if label is not None:
        ax.legend()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    if show and created:
        plt.show()
    if created:
        return fig, ax
    return ax

def dp_trajectory(mdp, pi):
    """
    Roll out the time-varying DP policy from the initial state (deterministic dynamics).
    Returns (states, actions, rewards): states has one more entry than actions/rewards
    (states[k] is the state before actions[k], reached via rewards[k]).
    """
    state = mdp.initial_state
    states = [state]
    actions = []
    rewards = []
    for t in range(pi.shape[0]):
        if mdp.is_goal(state):
            break
        i, j = mdp.state_to_coord(state)
        a = int(pi[t, state].item())
        i_next, j_next = mdp.apply_wind_dynamics(i, j, a)
        next_state = mdp.coord_to_state(i_next, j_next)
        actions.append(a)
        rewards.append(float(mdp.R[state, a].item()))
        state = next_state
        states.append(state)
    return states, actions, rewards

def greedy_trajectory(mdp, pi):
    """
    Roll out the greedy Q policy from the initial state (deterministic dynamics).
    Returns (states, actions, rewards): states has one more entry than actions/rewards
    (states[k] is the state before actions[k], reached via rewards[k]).
    """
    mdp.reset()
    state = mdp.initial_state
    states = [state]
    actions = []
    rewards = []
    done=False
    while not done:
        a = int(pi[state].item())
        state, reward, done = mdp.step(a)
        actions.append(a)
        rewards.append(reward)
        states.append(state)
    
    return states, actions, rewards


# ---------------------------------------------------------------------------
# Section 6 experiment runners
# ---------------------------------------------------------------------------


def run_dp_horizon_sweep(
    horizons: Sequence[int] = DEFAULT_DP_HORIZONS,
    device="cpu",
    save_dir: str = "figures",
    show: bool = False,
):
    """
    Section 6.1: for each H, plot V*_0 and pi*_0.
    Returns list of (horizon, pi, V, V0_initial).
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []
    n = len(horizons)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.2 * n))
    if n == 1:
        axes = np.array([axes])

    for row, H in enumerate(horizons):
        mdp = build_default_balloon_mdp(horizon=H, device=device)
        pi, V = DynamicProgramming(mdp).dynamicProgramming()
        V0 = V[0]
        pi0 = pi[0]
        # Full-horizon rollout under time-varying π*; overlay on π*_0 for path context.
        trajectory, traj_actions, traj_rewards = dp_trajectory(mdp, pi)
        plot_value_function(mdp, V0, ax=axes[row, 0], title=f"H={H}: V*_0")
        plot_policy(
            mdp,
            pi0,
            ax=axes[row, 1],
            title=f"H={H}: π*_0 (+ traj)",
            background_V=V0,
            trajectory=trajectory,
        )
        v_init = float(V0[mdp.initial_state].item())
        results.append({"horizon": H, "pi": pi, "V": V, "V0_initial": v_init})
        print(f"DP H={H}: V*_0(s0) = {v_init:.2f}")
        print(f"DP H={H}: trajectory (i,j)   = {[mdp.state_to_coord(s) for s in trajectory]}")
        print(f"DP H={H}: actions            = {[ACTION_NAMES[a] for a in traj_actions]}")
        print(f"DP H={H}: rewards            = {traj_rewards}")
        save_trajectory_gif(
            mdp,
            trajectory,
            save_path=os.path.join(save_dir, f"dp_traj_H{H}.gif"),
            pi=pi0,
            background_V=V0,
            actions=traj_actions,
            rewards=traj_rewards,
            title=f"DP H={H}",
        )

    fig.tight_layout()
    path = os.path.join(save_dir, "dp_horizon_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return results


def run_dp_time_slices(
    horizon: int = DEFAULT_DP_TIME_HORIZON,
    times: Optional[Sequence[int]] = None,
    device="cpu",
    save_dir: str = "figures",
    show: bool = False,
):
    """
    Section 6.1: for one longer horizon, plot (V*_t, pi*_t) at selected t.
    """
    os.makedirs(save_dir, exist_ok=True)
    mdp = build_default_balloon_mdp(horizon=horizon, device=device)
    pi, V = DynamicProgramming(mdp).dynamicProgramming()

    if times is None:
        # A few representative remaining-horizon slices, including late times.
        candidates = [0, max(1, horizon // 4), max(2, horizon // 2), horizon - 2, horizon - 1]
        times = sorted({t for t in candidates if 0 <= t < horizon})

    n = len(times)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.0 * n))
    if n == 1:
        axes = np.array([axes])

    for row, t in enumerate(times):
        remaining = horizon - t
        plot_value_function(
            mdp, V[t], ax=axes[row, 0],
            title=f"H={horizon}: V*_{t} (remaining horizon {remaining})",
        )
        plot_policy(
            mdp, pi[t], ax=axes[row, 1],
            title=f"H={horizon}: π*_{t} (remaining horizon {remaining})",
            background_V=V[t],
        )

    fig.tight_layout()
    path = os.path.join(save_dir, f"dp_time_slices_H{horizon}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"horizon": horizon, "times": list(times), "pi": pi, "V": V}


def run_vi_propagation(
    discount: float = DEFAULT_QL_DISCOUNT,
    tolerance: float = DEFAULT_VI_TOLERANCE,
    snapshot_iters: Sequence[int] = (0, 1, 2, 5, 10, 25),
    device="cpu",
    save_dir: str = "figures",
    show: bool = False,
):
    """
    Section 6.2: visualize successive VI updates (goal information propagating).
    Iteration 0 is the initial all-zero Q-table before any Bellman backup.
    """
    os.makedirs(save_dir, exist_ok=True)
    mdp = build_default_balloon_mdp(discount=discount, device=device)
    vi = ValueIteration(mdp)
    Q = torch.zeros(mdp.nStates, mdp.nActions, device=mdp.device)

    wanted = set(int(k) for k in snapshot_iters)
    snapshots = {}
    if 0 in wanted:
        snapshots[0] = (vi.computeVfromQ(Q).clone(), vi.extractMaxPifromQ(Q).clone())
    max_iter = max(wanted) if wanted else 0
    for k in range(1, max_iter + 1):
        Q = vi.valueIterationStep(Q)
        if k in wanted:
            snapshots[k] = (vi.computeVfromQ(Q).clone(), vi.extractMaxPifromQ(Q).clone())

    # Also run to convergence for the final panel.
    Q0 = torch.zeros(mdp.nStates, mdp.nActions, device=mdp.device)
    pi_star, V_star, n_iters, eps = vi.valueIteration(Q0, tolerance=tolerance)
    snapshots["final"] = (V_star, pi_star)

    keys = list(snapshot_iters) + ["final"]
    n = len(keys)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.8 * n))
    if n == 1:
        axes = np.array([axes])

    for row, key in enumerate(keys):
        V_k, pi_k = snapshots[key]
        label = f"iter {key}" if key != "final" else f"converged ({n_iters} iters)"
        plot_value_function(mdp, V_k, ax=axes[row, 0], title=f"VI γ={discount}: V after {label}")
        plot_policy(mdp, pi_k, ax=axes[row, 1], title=f"π after {label}", background_V=V_k)

    fig.tight_layout()
    path = os.path.join(save_dir, f"vi_propagation_gamma{discount}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    print(f"VI γ={discount}: converged in {n_iters} iters (ε={eps:.2e}), V*(s0)={V_star[mdp.initial_state].item():.2f}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "discount": discount,
        "snapshots": snapshots,
        "pi_star": pi_star,
        "V_star": V_star,
        "n_iters": n_iters,
        "epsilon": eps,
        "Q_star": vi.computeQfromV(V_star),
    }


def run_vi_discount_sweep(
    discounts: Sequence[float] = DEFAULT_VI_DISCOUNTS,
    tolerance: float = DEFAULT_VI_TOLERANCE,
    device="cpu",
    save_dir: str = "figures",
    show: bool = False,
):
    """
    Section 6.2: VI for multiple γ; record iterations to the same tolerance.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []
    n = len(discounts)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.2 * n))
    if n == 1:
        axes = np.array([axes])

    for row, gamma in enumerate(discounts):
        mdp = build_default_balloon_mdp(discount=gamma, device=device)
        vi = ValueIteration(mdp)
        Q0 = torch.zeros(mdp.nStates, mdp.nActions, device=mdp.device)
        pi, V, n_iters, eps = vi.valueIteration(Q0, tolerance=tolerance)
        trajectory, traj_actions, traj_rewards = greedy_trajectory(mdp, pi)
        plot_value_function(mdp, V, ax=axes[row, 0], title=f"γ={gamma}: V* ({n_iters} iters)")
        plot_policy(
            mdp,
            pi,
            ax=axes[row, 1],
            title=f"γ={gamma}: π* (+ traj)",
            background_V=V,
            trajectory=trajectory,
        )
        entry = {
            "discount": gamma,
            "pi": pi,
            "V": V,
            "n_iters": n_iters,
            "epsilon": eps,
            "V_initial": float(V[mdp.initial_state].item()),
            "Q_star": vi.computeQfromV(V),
        }
        results.append(entry)
        print(
            f"VI γ={gamma}: iters={n_iters}, ε={eps:.2e}, "
            f"V*(s0)={entry['V_initial']:.2f}"
        )
        print(f"VI discount={gamma}: trajectory (i,j)   = {[mdp.state_to_coord(s) for s in trajectory]}")
        print(f"VI discount={gamma}: actions            = {[ACTION_NAMES[a] for a in traj_actions]}")
        print(f"VI discount={gamma}: rewards            = {traj_rewards}")
        save_trajectory_gif(
            mdp,
            trajectory,
            save_path=os.path.join(save_dir, f"vi_traj_gamma{gamma}.gif"),
            pi=pi,
            background_V=V,
            actions=traj_actions,
            rewards=traj_rewards,
            title=f"VI γ={gamma}",
        )

    fig.tight_layout()
    path = os.path.join(save_dir, "vi_discount_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")

    # Compact summary bar chart of iteration counts.
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.bar([str(g) for g in discounts], [r["n_iters"] for r in results], color="steelblue")
    ax2.set_xlabel("Discount γ")
    ax2.set_ylabel(f"Iterations to ||ΔQ||_∞ ≤ {tolerance}")
    ax2.set_title("Value iteration: iterations vs discount")
    ax2.grid(True, axis="y", alpha=0.3)
    path2 = os.path.join(save_dir, "vi_iterations_vs_discount.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved {path2}")

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)
    return results


def _q_star_from_vi(mdp, tolerance: float = DEFAULT_VI_TOLERANCE):
    vi = ValueIteration(mdp)
    Q0 = torch.zeros(mdp.nStates, mdp.nActions, device=mdp.device)
    pi, V, n_iters, eps = vi.valueIteration(Q0, tolerance=tolerance)
    Q_star = vi.computeQfromV(V)
    return Q_star, pi, V, n_iters, eps


def _reachable_states(mdp) -> List[int]:
    """States reachable from the fixed initial state under some action sequence."""
    seen = {mdp.initial_state}
    frontier = [mdp.initial_state]
    while frontier:
        s = frontier.pop()
        for a in range(mdp.nActions):
            ns = int(torch.argmax(mdp.P[s, a]).item())
            if ns not in seen:
                seen.add(ns)
                frontier.append(ns)
    return sorted(seen)


def train_qlearning_with_metrics(
    mdp,
    Q_star,
    pi_star,
    num_episodes: int = DEFAULT_QL_EPISODES,
    alpha: float = DEFAULT_QL_ALPHA,
    epsilon: float = 0.1,
    log_every: int = DEFAULT_QL_LOG_EVERY,
    seed: int = 0,
):
    """
    Runs QLearning.qLearning exactly as run.py does (same seeding, same
    single call), and logs ||Q - Q*_VI||_∞ and greedy-policy agreement vs VI
    every `log_every` episodes (Section 6.3).

    Logging only observes training: mdp.reset and ql.qLearningUpdate are
    wrapped to read the Q-table at episode boundaries, and never touch any
    RNG, so the learned Q and rewards match `python run.py --method qlearning`
    with the same seed, discount, num_episodes, alpha, and epsilon.

    Metrics are restricted to states reachable from the initial state
    (Q-learning never updates the rest). The absorbing goal is excluded
    from policy agreement because both actions are optimal there.
    """
    seed_everything(seed)
    ql = QLearning(mdp, seed=seed)
    initialQ = torch.zeros(mdp.nStates, mdp.nActions, device=mdp.device)
    logged_episodes: List[int] = []
    q_errors: List[float] = []
    agreements: List[float] = []

    reachable = _reachable_states(mdp)
    reach_idx = torch.tensor(reachable, dtype=torch.long, device=mdp.device)
    agree_states = [s for s in reachable if s != mdp.goal_state]
    agree_idx = torch.tensor(agree_states, dtype=torch.long, device=mdp.device)

    def _log(ep_idx: int, Q):
        err = torch.max(torch.abs(Q[reach_idx] - Q_star[reach_idx])).item()
        agree = (torch.argmax(Q[agree_idx], dim=1) == pi_star[agree_idx]).float().mean().item()
        logged_episodes.append(ep_idx)
        q_errors.append(err)
        agreements.append(agree)

    # The Q-table being trained (qLearning works on its own copy of initialQ).
    live = {"Q": initialQ}
    episodes_started = [0]
    orig_update = ql.qLearningUpdate
    orig_reset = mdp.reset

    def update_hook(Q, *args, **kwargs):
        live["Q"] = Q
        return orig_update(Q, *args, **kwargs)

    def reset_hook():
        # Reset number k+1 marks the end of episode k.
        done_eps = episodes_started[0]
        if done_eps > 0 and done_eps % log_every == 0:
            _log(done_eps, live["Q"])
        episodes_started[0] += 1
        return orig_reset()

    _log(0, initialQ)
    ql.qLearningUpdate = update_hook
    mdp.reset = reset_hook
    try:
        Q, pi, episode_rewards = ql.qLearning(
            initialQ, num_episodes=num_episodes, alpha=alpha, epsilon=epsilon
        )
    finally:
        del ql.qLearningUpdate
        del mdp.reset
    _log(num_episodes, Q)

    return {
        "Q": Q,
        "pi": pi,
        "episode_rewards": episode_rewards,
        "logged_episodes": logged_episodes,
        "q_errors": q_errors,
        "agreements": agreements,
        "epsilon": epsilon,
    }


def run_qlearning_experiments(
    epsilons: Sequence[float] = DEFAULT_QL_EPSILONS,
    discount: float = DEFAULT_QL_DISCOUNT,
    num_episodes: int = DEFAULT_QL_EPISODES,
    alpha: float = DEFAULT_QL_ALPHA,
    tolerance: float = DEFAULT_VI_TOLERANCE,
    log_every: int = DEFAULT_QL_LOG_EVERY,
    seed: int = 0,
    device="cpu",
    save_dir: str = "figures",
    show: bool = False,
):
    """
    Section 6.3: train Q-learning, plot returns / Q-error / policy agreement,
    and compare across exploration rates.
    """
    os.makedirs(save_dir, exist_ok=True)
    mdp = build_default_balloon_mdp(discount=discount, device=device)
    Q_star, pi_star, V_star, n_iters, eps = _q_star_from_vi(mdp, tolerance=tolerance)
    print(f"VI reference (γ={discount}): {n_iters} iters, ε={eps:.2e}")

    results = []
    for epsilon in epsilons:
        t0 = time.perf_counter()
        # Fresh MDP sampling state per run (same dynamics).
        mdp_run = build_default_balloon_mdp(discount=discount, device=device)
        out = train_qlearning_with_metrics(
            mdp_run,
            Q_star=Q_star,
            pi_star=pi_star,
            num_episodes=num_episodes,
            alpha=alpha,
            epsilon=epsilon,
            log_every=log_every,
            seed=seed,
        )
        elapsed = time.perf_counter() - t0
        out["elapsed_sec"] = elapsed
        results.append(out)
        late = np.mean(out["episode_rewards"][-50:]) if out["episode_rewards"] else float("nan")
        print(
            f"Q-learning ε={epsilon}: {num_episodes} episodes in {elapsed:.2f}s, "
            f"final ||Q-Q*||_∞={out['q_errors'][-1]:.3f}, "
            f"agreement={out['agreements'][-1]:.0%}, "
            f"avg return (last 50)={late:.2f}"
        )

        trajectory, traj_actions, traj_rewards = greedy_trajectory(mdp_run, out["pi"])
        plot_value_and_policy(
            mdp_run,
            torch.max(out["Q"], dim=1).values,
            out["pi"],
            title=f"Q-learning ε={epsilon}",
            trajectory=trajectory,
            save_path=os.path.join(save_dir, f"ql_policy_eps{epsilon}.png"),
            show=False,
        )
        print(f"QL epsilon={epsilon}: trajectory (i,j)   = {[mdp.state_to_coord(s) for s in trajectory]}")
        print(f"QL epsilon={epsilon}: actions            = {[ACTION_NAMES[a] for a in traj_actions]}")
        print(f"QL epsilon={epsilon}: rewards            = {traj_rewards}")
        save_trajectory_gif(
            mdp_run,
            trajectory,
            save_path=os.path.join(save_dir, f"ql_traj_eps{epsilon}.gif"),
            pi=out["pi"],
            background_V=torch.max(out["Q"], dim=1).values,
            actions=traj_actions,
            rewards=traj_rewards,
            title=f"QL ε={epsilon}",
        )
        plt.close("all")

    # Learning curves (returns) for all ε.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for out in results:
        rewards = np.asarray(out["episode_rewards"], dtype=float)
        window = 50
        if len(rewards) >= window:
            smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(
                np.arange(window, len(rewards) + 1),
                smooth,
                linewidth=2.0,
                label=f"ε={out['epsilon']}",
            )
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Undiscounted return (moving avg, w={50})")
    ax.set_title("Q-learning learning curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = os.path.join(save_dir, "ql_learning_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)

    # Q-error vs episodes.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for out in results:
        ax.plot(out["logged_episodes"], out["q_errors"], linewidth=2.0, marker="o", markersize=3, label=f"ε={out['epsilon']}")
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\|Q_k - Q^*_{VI}\|_\infty$ (reachable states)")
    ax.set_title("Q-learning: distance to VI solution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = os.path.join(save_dir, "ql_q_error.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)

    # Policy agreement vs episodes.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for out in results:
        ax.plot(
            out["logged_episodes"],
            out["agreements"],
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=f"ε={out['epsilon']}",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction of reachable states agreeing with π*_VI")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Q-learning: greedy-policy agreement with VI")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = os.path.join(save_dir, "ql_policy_agreement.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)

    # Compare final learned Q against Q*_VI for the default ε=0.1 run if present.
    ref = next((r for r in results if abs(r["epsilon"] - 0.1) < 1e-12), results[0])
    plot_q_function(
        mdp,
        Q_star,
        title="Q*_VI",
        save_path=os.path.join(save_dir, "ql_qstar_vi.png"),
        show=False,
    )
    plt.close("all")
    plot_q_function(
        mdp,
        ref["Q"],
        title=f"Q learned (ε={ref['epsilon']})",
        save_path=os.path.join(save_dir, f"ql_q_learned_eps{ref['epsilon']}.png"),
        show=False,
    )
    plt.close("all")

    if show:
        plt.show()

    return {
        "Q_star": Q_star,
        "pi_star": pi_star,
        "V_star": V_star,
        "runs": results,
    }


def run_all_experiments(
    save_dir: str = "figures",
    device="cpu",
    show: bool = False,
    ql_episodes: int = DEFAULT_QL_EPISODES,
):
    """Run the full Section 6 experiment suite and report wall-clock time."""
    os.makedirs(save_dir, exist_ok=True)
    t0 = time.perf_counter()

    print("=" * 64)
    print(" Environment")
    print("=" * 64)
    mdp = build_default_balloon_mdp(discount=DEFAULT_QL_DISCOUNT, device=device)
    plot_environment(mdp, save_path=os.path.join(save_dir, "environment_wind.png"))
    plt.close("all")

    print("=" * 64)
    print(" Section 6.1 — Dynamic programming")
    print("=" * 64)
    t = time.perf_counter()
    dp_sweep = run_dp_horizon_sweep(device=device, save_dir=save_dir, show=show)
    dp_slices = run_dp_time_slices(device=device, save_dir=save_dir, show=show)
    print(f"DP experiments wall time: {time.perf_counter() - t:.2f}s")

    print("=" * 64)
    print(" Section 6.2 — Value iteration")
    print("=" * 64)
    t = time.perf_counter()
    vi_prop = run_vi_propagation(device=device, save_dir=save_dir, show=show)
    vi_sweep = run_vi_discount_sweep(device=device, save_dir=save_dir, show=show)
    print(f"VI experiments wall time: {time.perf_counter() - t:.2f}s")

    print("=" * 64)
    print(" Section 6.3 — Q-learning")
    print("=" * 64)
    t = time.perf_counter()
    ql = run_qlearning_experiments(
        device=device, save_dir=save_dir, show=show, num_episodes=ql_episodes
    )
    print(f"Q-learning experiments wall time: {time.perf_counter() - t:.2f}s")

    total = time.perf_counter() - t0
    print("=" * 64)
    print(f" All Section 6 experiments finished in {total:.2f}s")
    print(f" Figures written under: {save_dir}/")
    print("=" * 64)
    return {
        "dp_sweep": dp_sweep,
        "dp_slices": dp_slices,
        "vi_propagation": vi_prop,
        "vi_sweep": vi_sweep,
        "qlearning": ql,
        "total_seconds": total,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize balloon MDP experiments (Section 6).")
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["all", "dp", "vi", "qlearning", "env"],
        help="Which Section 6 experiment block to run",
    )
    parser.add_argument("--save-dir", default="figures")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    if args.experiment == "env":
        mdp = build_default_balloon_mdp(discount=DEFAULT_QL_DISCOUNT, device=device)
        plot_environment(mdp, save_path=os.path.join(args.save_dir, "environment_wind.png"))
        if args.show:
            plt.show()
    elif args.experiment == "dp":
        run_dp_horizon_sweep(device=device, save_dir=args.save_dir, show=args.show)
        run_dp_time_slices(device=device, save_dir=args.save_dir, show=args.show)
    elif args.experiment == "vi":
        run_vi_propagation(device=device, save_dir=args.save_dir, show=args.show)
        run_vi_discount_sweep(device=device, save_dir=args.save_dir, show=args.show)
    elif args.experiment == "qlearning":
        run_qlearning_experiments(
            seed=args.seed,
            device=device,
            save_dir=args.save_dir,
            show=args.show,
        )
    else:
        run_all_experiments(
            save_dir=args.save_dir,
            device=device,
            show=args.show,
        )


if __name__ == "__main__":
    main()
