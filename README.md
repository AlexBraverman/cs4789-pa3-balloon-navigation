# CS 4789 PA3: Balloon Navigation

A balloon navigates a 6x6 gridworld by choosing to move UP or DOWN; a
spatially varying wind field determines its horizontal displacement. See
`handout.pdf` for the full assignment description.

## Files

- `MDP.py` -- the balloon gridworld (states, actions, wind, transitions,
  rewards). Fully implemented; you should not need to modify it.
- `DP.py` -- **student code.** Finite-horizon dynamic programming (Section 3).
- `VI.py` -- **student code.** Infinite-horizon value iteration (Section 4).
- `Qlearning.py` -- **student code.** Tabular Q-learning (Section 5).
- `visualize.py` -- plotting helpers and Section 6 experiment runners.
- `colab.ipynb` -- Colab / local notebook entry point (tests + experiments).
- `test.py` -- basic sanity tests for the three files above.
- `run.py` -- CLI to run DP/VI/Q-learning once they're implemented and
  print the resulting policy/value function as an ASCII grid.

Every TODO is numbered to match the handout (e.g. `TODO 3.1.1`). Do not
change function names, argument lists, or return types.

## Setup

```bash
pip install -r requirements.txt
```

or, to install as an editable package:

```bash
pip install -e .
```

Requires Python 3.10+.

## Running the tests

```bash
python test.py
```

Passing these tests does not guarantee your implementation is correct --
also inspect the resulting value functions, policies, and learning curves
yourself with `visualize.py` / `colab.ipynb`.

## Running the Section 6 experiments

```bash
python visualize.py --experiment all --save-dir figures
python visualize.py --experiment dp
python visualize.py --experiment vi
python visualize.py --experiment qlearning --num-episodes 1000
```

Or open `colab.ipynb` and run the cells top to bottom. Figures are written
under `figures/`.

## Running a method directly

```bash
python run.py --method dp --horizon 10 [--verbose]
python run.py --method vi --discount 0.9 --tolerance 0.01
python run.py --method qlearning --discount 0.9 --num-episodes 1000 --alpha 0.5 --epsilon 0.1
```

Each prints the resulting policy (and, for `dp`/`vi`, the value function)
as an ASCII grid, oriented like Figure 1 (`S`/`G` mark the initial/goal
states, `^`/`v` mark UP/DOWN). Run `python run.py --help` for the full
list of flags.

## Notes

- All state/value/Q representations are `torch` tensors, and everything
  runs on CPU -- no GPU is needed (or used), including on Colab.
- `Qlearning.py` must not read `mdp.P` or `mdp.R` -- it should only
  interact with the environment through `mdp.reset()` / `mdp.step()`.
