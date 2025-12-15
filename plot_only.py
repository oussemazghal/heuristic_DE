import matplotlib.pyplot as plt
import numpy as np
import os

RESULT_DIR = "resultsss"

ALGOS = [
    "DEClassique",
    "JDESimple",
    "JDEAdapted",
    "PSO",
    "ABC",
    "GSA",
    "GA",            # <---- AJOUTER
    "PSOHybrid"      # <---- AJOUTER
]

FUNCS = {
    "f2":  "unimodal",
    "f4":  "unimodal",
    "f12": "hybrid",
    "f21": "composition"
}

def load_curve(path):
    fes, best = None, None
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    mode = None
    for line in lines:
        if line == "FES:":
            mode = "fes"
            continue
        if line == "Best:":
            mode = "best"
            continue
        if mode == "fes":
            fes = np.array(list(map(int, line.split())))
        elif mode == "best":
            best = np.array(list(map(float, line.split())))
    return fes, best


# ------------------------ PLOT ONLY ------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.ravel()

idx = 0
for fname, ftype in FUNCS.items():
    ax = axes[idx]
    idx += 1

    for algo_name in ALGOS:
        path = f"{RESULT_DIR}/{algo_name}_{fname}_{ftype}.txt"
        if not os.path.isfile(path):
            continue

        fes, best = load_curve(path)
        ax.plot(fes, best, label=algo_name)

    ax.set_title(f"{fname} ({ftype})")
    ax.set_xlabel("FES")
    ax.set_ylabel("Mean best value")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    
    ax.legend()

plt.tight_layout()
plt.show()
