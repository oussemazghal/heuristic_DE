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
*   `site heuritsic/`: A web-based optimization dashboard for visualizing results.

## Running the Experiments

To run the basic experiments:
```bash
python basic_de_experiments.py
```

To run the comprehensive suite with plotting:
```bash
python fonctionnel.py
```

## Web Interface (Optimization Dashboard)

The repository includes a web dashboard to visualize and compare the algorithm performance.

### Prerequisites
*   Python 3.8+
*   `pip`

### Installation & Running

1.  Navigate to the site directory:
    ```bash
    cd "site heuritsic"
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the application:
    ```bash
    uvicorn backend.main:app --reload
    ```

4.  Open your browser and visit:
    ```
    http://127.0.0.1:8000
    ```

## Requirements for Scripts

*   Python 3.x
*   NumPy
*   Matplotlib
