# Heuristic Differential Evolution (DE) Experiments

This repository contains implementations and experiments with various Heuristic Algorithms, specifically focusing on Differential Evolution (DE) and its variants, tested on CEC2017 benchmark functions.

## Algorithms Implemented

The codebase includes implementations of:
*   **DE Classique**:  Standard Differential Evolution.
*   **jDE Simple**: Self-adaptive Differential Evolution.
*   **jDE Adapted**: An adapted version of jDE.
*   **PSO**: Particle Swarm Optimization.
*   **ABC**: Artificial Bee Colony.
*   **GSA**: Gravitational Search Algorithm.

## Files Structure

*   `basic_de_experiments.py`: (Formerly `AAAAA.py`) Contains basic implementations of DE and jDE algorithms with multithreaded execution for benchmarking.
*   `fonctionnel.py`: A more comprehensive script including DE, PSO, ABC, GSA, and hybrid variants, featuring optimized batch evaluations and result plotting.
*   `cec2017/`: Directory containing the CEC2017 benchmark functions.

## Running the Experiments

To run the basic experiments:
```bash
python basic_de_experiments.py
```

To run the comprehensive suite with plotting:
```bash
python fonctionnel.py
```

## Requirements

*   Python 3.x
*   NumPy
*   Matplotlib
