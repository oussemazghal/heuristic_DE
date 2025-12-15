import os
import numpy as np
import matplotlib.pyplot as plt
from cec2017.functions import f2, f4, f12, f21  # pip install git+https://github.com/tilleyd/cec2017-py.git

# ---------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------
# 1) Configure your user algorithm here
USE_USER_ALGO = False            # Set to True to include your algorithm
USER_ALGO_NAME = "MyAlgo"        # Name used in file names and legend

# Mode for user algorithm:
#   "run"  -> run your algo on CEC functions now (and optionally save results)
#   "read" -> read precomputed results of your algo from text files (like PSO/ABC/GSA)
USER_ALGO_MODE = "run"           # "run" or "read"

# If USER_ALGO_MODE == "run", you can also save its mean curve per function to results/...
SAVE_USER_ALGO_RESULTS = True

# Directory where PSO / ABC / GSA (and optional user algo) results are stored
RESULTS_DIR = "resultss"

# ---------------------------------------------------------
# Global experiment settings (same as original script.py)
# ---------------------------------------------------------
DIM = 30
POP_SIZE = 30
ITERATIONS = 1000     # evaluations = ITERATIONS * POP_SIZE
RUNS = 30
LB, UB = -100.0, 100.0

# ---------------------------------------------------------
# TEMPLATE: USER ALGORITHM
# ---------------------------------------------------------
def run_user_algorithm(func, dim, pop_size, iterations, lb, ub):
    """
    Implement your algorithm here.
    It MUST return a 1D array 'best_history' of length 'iterations',
    where best_history[t] is the best-so-far fitness at iteration t.
    """
    raise NotImplementedError(
        "Implement run_user_algorithm() and set USE_USER_ALGO = True."
    )

# ---------------------------------------------------------
# Utility: wrap CEC2017 functions to accept (pop_size, dim)
# ---------------------------------------------------------
def cec_wrapper(cec_func):
    def f(x):
        # cec2017 expects 2D ndarray and returns 1D array of fitness values
        return cec_func(x)
    return f


CEC_FUNCTIONS = [
    ("f2 (unimodal)",       cec_wrapper(f2)),
    ("f4 (unimodal)",       cec_wrapper(f4)),
    ("f12 (hybrid)",        cec_wrapper(f12)),
    ("f21 (composition)",   cec_wrapper(f21)),
]

# ---------------------------------------------------------
# FILE HELPERS
# ---------------------------------------------------------
def safe_func_name(func_name: str) -> str:
    return (
        func_name.replace(" ", "_")
                 .replace("(", "")
                 .replace(")", "")
                 .replace(".", "")
    )


def safe_algo_name(algo_name: str) -> str:
    return algo_name.replace(" ", "_")


def curve_filename(algo_name: str, func_name: str) -> str:
    return f"{safe_algo_name(algo_name)}_{safe_func_name(func_name)}.txt"


def load_curve(file_path: str) -> np.ndarray:
    """
    Loads the curve by selecting the line with the most space-separated tokens.
    Assumes the longest line is the vector of fitness values.
    """
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"No non-empty lines in {file_path}")

    longest_line = max(lines, key=lambda l: len(l.split()))
    return np.array(list(map(float, longest_line.split())))


def read_saved_curve(algo_name: str, func_name: str):
    """
    Try to read a saved curve for (algo, function).
    Returns np.ndarray or None if file is missing.
    """
    fname = curve_filename(algo_name, func_name)
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.isfile(path):
        print(f"[WARN] File not found for {algo_name} on {func_name}: {path}")
        return None

    try:
        curve = load_curve(path)
        return curve
    except Exception as e:
        print(f"[WARN] Could not load curve from {path}: {e}")
        return None


def save_curve(algo_name: str, func_name: str, curve: np.ndarray):
    """
    Save a curve for (algo, function) in the same format as previous scripts.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = curve_filename(algo_name, func_name)
    path = os.path.join(RESULTS_DIR, fname)

    with open(path, "w") as fout:
        fout.write(f"Algorithm: {algo_name}\n")
        fout.write(f"Function: {func_name}\n")
        fout.write("Best curve per iteration:\n")
        fout.write(" ".join(map(str, curve)))
        fout.write("\n")

    print(f"[INFO] Saved user algorithm results: {path}")


# ---------------------------------------------------------
# MAIN: READ PSO/ABC/GSA FROM FILES + OPTIONAL USER ALGO
# ---------------------------------------------------------
def main():
    # results[func_name][algo_name] = curve
    results = {}

    # Base algorithms always loaded from text files
    base_algorithms = ["PSO", "ABC", "GSA"]

    eval_axis = None  # evaluations = iteration_index * POP_SIZE

    # ---- Load PSO / ABC / GSA ----
    for func_name, _ in CEC_FUNCTIONS:
        func_results = {}

        for algo_name in base_algorithms:
            curve = read_saved_curve(algo_name, func_name)
            if curve is None:
                continue

            func_results[algo_name] = curve

            # Initialize evaluation axis based on curve length
            if eval_axis is None:
                n_iter = len(curve)
                eval_axis = np.arange(1, n_iter + 1) * POP_SIZE

        if func_results:
            results[func_name] = func_results

    # ---- Handle USER ALGORITHM ----
    if USE_USER_ALGO:
        mode = USER_ALGO_MODE.lower().strip()

        if mode == "read":
            # Read user algo from text files like others
            for func_name, _ in CEC_FUNCTIONS:
                curve = read_saved_curve(USER_ALGO_NAME, func_name)
                if curve is None:
                    continue

                if func_name not in results:
                    results[func_name] = {}
                results[func_name][USER_ALGO_NAME] = curve

                if eval_axis is None:
                    n_iter = len(curve)
                    eval_axis = np.arange(1, n_iter + 1) * POP_SIZE

        elif mode == "run":
            # Run user algorithm fresh on all CEC functions
            for func_name, func in CEC_FUNCTIONS:
                print(f"[INFO] Running user algorithm '{USER_ALGO_NAME}' on {func_name} ...")
                runs_curves = np.empty((RUNS, ITERATIONS))

                for r in range(RUNS):
                    curve = run_user_algorithm(func, DIM, POP_SIZE, ITERATIONS, LB, UB)
                    if len(curve) != ITERATIONS:
                        raise ValueError(
                            f"run_user_algorithm must return length {ITERATIONS}, "
                            f"got {len(curve)}"
                        )
                    runs_curves[r, :] = curve

                mean_curve = runs_curves.mean(axis=0)

                if func_name not in results:
                    results[func_name] = {}
                results[func_name][USER_ALGO_NAME] = mean_curve

                if SAVE_USER_ALGO_RESULTS:
                    save_curve(USER_ALGO_NAME, func_name, mean_curve)

            if eval_axis is None:
                eval_axis = np.arange(1, ITERATIONS + 1) * POP_SIZE

        else:
            raise ValueError("USER_ALGO_MODE must be 'run' or 'read'.")

    # -----------------------------------------------------
    # PLOTTING: same style as original script.py (2x2)
    # -----------------------------------------------------
    if eval_axis is None:
        print("[WARN] No curves loaded. Nothing to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for idx, (func_name, _) in enumerate(CEC_FUNCTIONS):
        ax = axes[idx]
        func_results = results.get(func_name, {})

        if not func_results:
            ax.set_title(func_name + " (no data)")
            ax.axis("off")
            continue

        for algo_name, curve in func_results.items():
            # Make sure x and y sizes match
            if len(eval_axis) == len(curve):
                x_axis = eval_axis
            else:
                x_axis = np.arange(1, len(curve) + 1) * POP_SIZE

            ax.plot(x_axis, curve, label=algo_name)

        ax.set_title(func_name)
        ax.set_xlabel("Nb. evaluations")
        ax.set_ylabel("Moyenne")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
