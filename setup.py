from setuptools import setup

setup(
    name="cs4789-pa3",
    version="0.1.0",
    description="Balloon gridworld: finite-horizon DP, value iteration, and tabular Q-learning",
    python_requires=">=3.10",
    py_modules=[
        "MDP",
        "DP",
        "VI",
        "Qlearning",
        "visualize",
        "test",
        "run",
    ],
    install_requires=[
        "torch>=2.0.0",
        "matplotlib>=3.6",
        "Pillow>=9.0",
    ],
)
