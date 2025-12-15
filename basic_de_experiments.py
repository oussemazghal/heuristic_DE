import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import importlib
import time

# Charger CEC2017
cec = importlib.import_module("cec2017.functions")

# --------------------------------------------------------------------
# FONCTIONS UTILISÉES (comme ABC/GSA/PSO)
# --------------------------------------------------------------------
FUNC_MAP = {
    "f2":  ("unimodal",    cec.f2),
    "f4":  ("unimodal",    cec.f4),
    "f12": ("hybrid",      cec.f12),
    "f21": ("composition", cec.f21),
}

# --------------------------------------------------------------------
# PARAMÈTRES
# --------------------------------------------------------------------
DIM = 30
LB, UB = -100, 100
FESMAX = 30000
N_RUNS = 30

POP_DE = 60
POP_JDE = 40
POP_JDEA = 30

# --------------------------------------------------------------------
# EVALUATOR IDENTIQUE À TON CODE
# --------------------------------------------------------------------
class Evaluator:
    def __init__(self, func, FESmax):
        self.func = func
        self.FES = 0
        self.FESmax = FESmax

    def __call__(self, X):
        X = np.asarray(X)
        n = 1 if X.ndim == 1 else X.shape[0]
        if self.FES + n > self.FESmax:
            return None, True
        self.FES += n
        return self.func(X), False


# --------------------------------------------------------------------
#  DE CLASSIQUE
# --------------------------------------------------------------------
def de_classique_FES(func, D=DIM, N=POP_DE, LB=None, UB=None,
                     F=0.8, CR=0.9, FESmax=FESMAX):

    if LB is None: LB = -100*np.ones(D)
    if UB is None: UB = 100*np.ones(D)
    LB, UB = np.asarray(LB), np.asarray(UB)

    eva = Evaluator(func, FESmax)
    x = np.random.uniform(LB, UB, (N, D))
    fvals, _ = eva(x)

    best_idx = np.argmin(fvals)
    best_val = fvals[best_idx]
    best = x[best_idx].copy()

    hist_FES, hist_best = [], []

    while True:
        new_pop = np.zeros_like(x)
        for i in range(N):

            if eva.FES >= FESmax:
                return hist_FES, hist_best

            candidates = [j for j in range(N) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)

            v = x[a] + F*(x[b] - x[c])
            v = np.clip(v, LB, UB)

            u = x[i].copy()
            j_rand = np.random.randint(D)
            for j in range(D):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            fu, stop = eva(u[None, :])
            if stop: return hist_FES, hist_best
            fu = fu[0]

            if fu < fvals[i]:
                new_pop[i] = u
                fvals[i] = fu
                if fu < best_val:
                    best_val = fu
                    best = u.copy()
            else:
                new_pop[i] = x[i]

        x = new_pop

        if eva.FES % 1000 == 0:
            hist_FES.append(eva.FES)
            hist_best.append(best_val)


# --------------------------------------------------------------------
# JDE SIMPLE
# --------------------------------------------------------------------
def jde_simple_FES(func, D=DIM, N=POP_JDE, LB=None, UB=None,
                   F0=0.5, CR0=0.9, tau1=0.1, tau2=0.1,
                   FESmax=FESMAX):

    if LB is None: LB = -100*np.ones(D)
    if UB is None: UB = 100*np.ones(D)
    LB, UB = np.asarray(LB), np.asarray(UB)

    eva = Evaluator(func, FESmax)
    x = np.random.uniform(LB, UB, (N, D))
    fvals, _ = eva(x)

    F_i = np.full(N, F0)
    CR_i = np.full(N, CR0)

    best_idx = np.argmin(fvals)
    best_val = fvals[best_idx]
    best = x[best_idx].copy()

    hist_FES, hist_best = [], []

    while True:
        new_pop = np.zeros_like(x)
        for i in range(N):

            if eva.FES >= FESmax:
                return hist_FES, hist_best

            if np.random.rand() < tau1:
                F_i[i] = 0.1 + 0.9*np.random.rand()
            if np.random.rand() < tau2:
                CR_i[i] = np.random.rand()

            F = F_i[i]
            CR = CR_i[i]

            candidates = [j for j in range(N) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            v = x[a] + F*(x[b] - x[c])
            v = np.clip(v, LB, UB)

            u = x[i].copy()
            j_rand = np.random.randint(D)
            for j in range(D):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            fu, stop = eva(u[None, :])
            if stop: return hist_FES, hist_best
            fu = fu[0]

            if fu < fvals[i]:
                new_pop[i] = u
                fvals[i] = fu
                if fu < best_val:
                    best_val = fu
                    best = u.copy()
            else:
                new_pop[i] = x[i]

        x = new_pop

        if eva.FES % 1000 == 0:
            hist_FES.append(eva.FES)
            hist_best.append(best_val)


# --------------------------------------------------------------------
# JDE ADAPTÉE
# --------------------------------------------------------------------
def jde_adapted_FES(func, D=DIM, N=POP_JDEA, LB=None, UB=None,
                    F0=0.5, CR0=0.9, tau1=0.1, tau2=0.1,
                    p_current_to_best=0.1,
                    ls_interval_FES=5000, ls_max_evals=30,
                    FESmax=FESMAX):

    if LB is None: LB = -100*np.ones(D)
    if UB is None: UB = 100*np.ones(D)
    LB, UB = np.asarray(LB), np.asarray(UB)

    sigma = 0.001*(UB-LB)

    eva = Evaluator(func, FESmax)
    x = np.random.uniform(LB, UB, (N, D))
    fvals, _ = eva(x)

    F_i = np.full(N, F0)
    CR_i = np.full(N, CR0)

    best_idx = np.argmin(fvals)
    best_val = fvals[best_idx]
    best = x[best_idx].copy()

    hist_FES, hist_best = [], []
    next_ls_FES = ls_interval_FES

    while True:
        new_pop = x.copy()

        for i in range(N):

            if eva.FES >= FESmax:
                return hist_FES, hist_best

            if np.random.rand() < tau1:
                F_i[i] = 0.1 + 0.9*np.random.rand()
            if np.random.rand() < tau2:
                CR_i[i] = np.random.rand()

            F = F_i[i]
            CR = CR_i[i]

            candidates = [j for j in range(N) if j != i]

            if np.random.rand() < p_current_to_best:
                a, b = np.random.choice(candidates, 2, replace=False)
                v = x[i] + F*(best - x[i]) + F*(x[a] - x[b])
            else:
                a, b, c = np.random.choice(candidates, 3, replace=False)
                v = x[a] + F*(x[b] - x[c])

            v = np.clip(v, LB, UB)

            u = x[i].copy()
            j_rand = np.random.randint(D)
            for j in range(D):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            fu, stop = eva(u[None, :])
            if stop: return hist_FES, hist_best
            fu = fu[0]

            if fu < fvals[i]:
                new_pop[i] = u
                fvals[i] = fu
                if fu < best_val:
                    best_val = fu
                    best = u.copy()

        x = new_pop

        if eva.FES % 1000 == 0:
            hist_FES.append(eva.FES)
            hist_best.append(best_val)

        # Local search
        if eva.FES >= next_ls_FES and eva.FES < FESmax:

            used = 0
            while used < ls_max_evals and eva.FES < FESmax:
                noise = np.random.randn(D)*sigma
                cand = np.clip(best + noise, LB, UB)

                f_cand, stop = eva(cand[None, :])
                if stop: return hist_FES, hist_best

                f_cand = f_cand[0]
                used += 1

                if f_cand < best_val:
                    best_val = f_cand
                    best = cand.copy()

            worst = np.argmax(fvals)
            x[worst] = best.copy()
            fvals[worst] = best_val

            next_ls_FES += ls_interval_FES


# --------------------------------------------------------------------
# THREAD WORKER – 1 run
# --------------------------------------------------------------------
def worker(algo_name, func, jde_fn, run_id):
    print(f"   ▶ Run {run_id+1}/{N_RUNS} en cours...")
    hist_FES, hist_best = jde_fn(lambda X: func(X.reshape(X.shape[0],-1)))
    print(f"   ✔ Run {run_id+1}/{N_RUNS} terminé")
    return np.array(hist_best)


# --------------------------------------------------------------------
# SAUVEGARDE
# --------------------------------------------------------------------
def save_curve(filename, algo_name, func_name, ftype, curve):
    os.makedirs("resultss", exist_ok=True)
    with open(f"resultss/{filename}", "w") as f:
        f.write(f"Algorithm: {algo_name}\n")
        f.write(f"Function: {func_name} ({ftype})\n")
        f.write("Best curve per iteration:\n")
        f.write(" ".join(str(v) for v in curve))


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":

    ALGOS = {
        "DEClassique": de_classique_FES,
        "JDESimple": jde_simple_FES,
        "JDEAdapted": jde_adapted_FES,
    }

    print("\n🚀 Début de l'expérimentation complète...\n")

    for fname, (ftype, ffunc) in FUNC_MAP.items():

        print(f"\n==============================")
        print(f"  Fonction {fname} ({ftype})")
        print(f"==============================\n")

        for algo_name, algo_fn in ALGOS.items():

            print(f"\n🔵 Algorithme : {algo_name}")
            print(f"   30 runs en multithread...\n")

            start = time.time()

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(
                    lambda r: worker(algo_name, ffunc, algo_fn, r),
                    range(N_RUNS)
                ))

            duration = time.time() - start
            print(f"\n⏳ Temps total pour {algo_name} sur {fname} : {duration:.2f} sec")

            max_len = max(len(r) for r in results)
            curves = []
            for r in results:
                if len(r) < max_len:
                    r = np.concatenate([r, np.full(max_len-len(r), r[-1])])
                curves.append(r)

            mean_curve = np.mean(curves, axis=0)

            filename = f"{algo_name}_{fname}_{ftype}.txt"
            save_curve(filename, algo_name, fname, ftype, mean_curve.tolist())

            print(f"   ✔ Fichier sauvegardé → resultss/{filename}")

    print("\n🎉 EXPÉRIMENTATION TERMINÉE — fichiers prêts dans /resultss/ 🎉")
