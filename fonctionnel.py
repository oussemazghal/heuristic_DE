import numpy as np
from concurrent.futures import ThreadPoolExecutor
import importlib
import os
import time
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# CEC2017 FUNCTIONS
# --------------------------------------------------------------------
cec = importlib.import_module("cec2017.functions")

FUNC_MAP = {
    "f2":  ("unimodal",    cec.f2),
    "f4":  ("unimodal",    cec.f4),
    "f12": ("hybrid",      cec.f12),
    "f21": ("composition", cec.f21),
}

# --------------------------------------------------------------------
# GLOBAL PARAMS
# --------------------------------------------------------------------
DIM = 30
LB, UB = -100, 100
N_RUNS = 30
FESMAX = 30000
SAVE_EVERY = 100

# Populations
POP_DE   = 60
POP_JDE  = 40
POP_JDEA = 30
POP_PSO  = 30
POP_ABC  = 30
POP_GSA  = 30

RESULT_DIR = "resultsss"


# --------------------------------------------------------------------
# EVALUATOR OPTIMISÉ (batch evaluation)
# --------------------------------------------------------------------
class Evaluator:
    def __init__(self, func, FESmax):
        self.func = func
        self.FES = 0
        self.FESmax = FESmax

    def __call__(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n = X.shape[0]
        if self.FES + n > self.FESmax:
            return None, True

        self.FES += n
        # Évaluation batch - BEAUCOUP plus rapide
        return self.func(X), False

def ga_FES(func, D=DIM, N=30, LB=None, UB=None,
           mutation_scale=0.005, FESmax=FESMAX):

    if LB is None: LB = -100*np.ones(D)
    if UB is None: UB = 100*np.ones(D)
    LB, UB = np.asarray(LB), np.asarray(UB)

    eva = Evaluator(func, FESmax)
    split = D // 2

    mutation_amplitude = mutation_scale * (UB - LB)
    rng = np.random.default_rng()

    parents = rng.uniform(LB, UB, size=(N, D))
    fitness, stop = eva(parents)
    if stop: 
        return [], []

    best_val = np.min(fitness)

    hist_FES, hist_best = [], []
    next_save = SAVE_EVERY

    while eva.FES < FESmax:

        j = rng.integers(0, N, size=N)
        k = rng.integers(0, N, size=N)

        children = np.hstack((parents[j, :split], parents[k, split:]))

        mut_idx = rng.integers(0, D, size=N)
        children[np.arange(N), mut_idx] += rng.uniform(
            -mutation_amplitude, mutation_amplitude, size=N
        )

        children = np.clip(children, LB, UB)

        children_fit, stop = eva(children)
        if stop:
            break

        combined = np.vstack((parents, children))
        combined_fit = np.concatenate((fitness, children_fit))

        best_idx = np.argsort(combined_fit)[:N]
        parents = combined[best_idx]
        fitness = combined_fit[best_idx]

        best_val = fitness[0]

        if eva.FES >= next_save:
            hist_FES.append(eva.FES)
            hist_best.append(best_val)
            next_save += SAVE_EVERY

    return hist_FES, hist_best
def pso_hybrid_FES(func, D=DIM, N=30, LB=None, UB=None,
                   c1=2.0, c2=2.0, p_mut=0.05, sigma=25.0,
                   FESmax=FESMAX):

    if LB is None: LB = -100*np.ones(D)
    if UB is None: UB = 100*np.ones(D)
    LB, UB = np.asarray(LB), np.asarray(UB)

    eva = Evaluator(func, FESmax)
    rng = np.random.default_rng()

    x = rng.uniform(LB, UB, size=(N, D))
    v = np.zeros((N, D))

    fvals, stop = eva(x)
    if stop:
        return [], []

    pbest = x.copy()
    pbest_val = fvals.copy()

    gbest = pbest[np.argmin(pbest_val)].copy()
    best_val = np.min(pbest_val)

    hist_FES, hist_best = [], []
    next_save = SAVE_EVERY

    t = 0
    while eva.FES < FESmax:
        t += 1

        w = 0.9 - 0.5 * t / (FESmax / N)
        r1 = rng.random((N, D))
        r2 = rng.random((N, D))

        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = x + v

        mutation_mask = rng.random((N, D)) < p_mut
        if np.any(mutation_mask):
            x += mutation_mask * rng.normal(0.0, sigma, size=(N, D))

        x = np.clip(x, LB, UB)

        fvals, stop = eva(x)
        if stop:
            break

        improved = fvals < pbest_val
        pbest[improved] = x[improved]
        pbest_val[improved] = fvals[improved]

        gbest = pbest[np.argmin(pbest_val)]
        best_val = np.min(pbest_val)

        if eva.FES >= next_save:
            hist_FES.append(eva.FES)
            hist_best.append(best_val)
            next_save += SAVE_EVERY

    return hist_FES, hist_best

# --------------------------------------------------------------------
# 1) DE CLASSIQUE OPTIMISÉ
# --------------------------------------------------------------------
def de_classique_FES(func, D=DIM, N=POP_DE, LB=None, UB=None,
                     F=0.8, CR=0.9, FESmax=FESMAX):

    if LB is None: LB = -100*np.ones(D)
    if UB is None: UB = 100*np.ones(D)
    LB, UB = np.asarray(LB), np.asarray(UB)

    eva = Evaluator(func, FESmax)

    x = np.random.uniform(LB, UB, (N, D))
    fvals, stop = eva(x)
    if stop:
        return [], []

    best_idx = np.argmin(fvals)
    best_val = fvals[best_idx]

    hist_FES, hist_best = [], []
    next_save = SAVE_EVERY

    while eva.FES < FESmax:
        # Génération vectorisée de tous les mutants
        candidates = np.arange(N)
        
        # Sélection des indices a, b, c pour tous les individus
        abc_indices = np.array([
            np.random.choice([j for j in range(N) if j != i], 3, replace=False)
            for i in range(N)
        ])
        
        # Mutation vectorisée
        v = x[abc_indices[:, 0]] + F * (x[abc_indices[:, 1]] - x[abc_indices[:, 2]])
        v = np.clip(v, LB, UB)
        
        # Croisement vectorisé
        u = x.copy()
        j_rand = np.random.randint(0, D, N)
        mask = np.random.rand(N, D) < CR
        for i in range(N):
            mask[i, j_rand[i]] = True
        u[mask] = v[mask]
        
        # Évaluation en batch
        fu, stop = eva(u)
        if stop:
            break
            
        # Sélection vectorisée
        improved = fu < fvals
        x[improved] = u[improved]
        fvals[improved] = fu[improved]
        
        # Mise à jour du meilleur
        current_best_idx = np.argmin(fvals)
        if fvals[current_best_idx] < best_val:
            best_val = fvals[current_best_idx]

        # Sauvegarde
        if eva.FES >= next_save:
            hist_FES.append(eva.FES)
            hist_best.append(best_val)
            next_save += SAVE_EVERY

    return hist_FES, hist_best


# --------------------------------------------------------------------
# 2) JDE SIMPLE OPTIMISÉ
# --------------------------------------------------------------------
def jde_simple_FES(func, D=DIM, N=POP_JDE, LB=None, UB=None,
                   F0=0.5, CR0=0.9, tau1=0.1, tau2=0.1,
                   FESmax=FESMAX):

    if LB is None: LB = -100*np.ones(D)
    if UB is None: UB = 100*np.ones(D)
    LB, UB = np.asarray(LB), np.asarray(UB)

    eva = Evaluator(func, FESmax)

    x = np.random.uniform(LB, UB, (N, D))
    fvals, stop = eva(x)
    if stop:
        return [], []

    F_i = np.full(N, F0)
    CR_i = np.full(N, CR0)

    best_idx = np.argmin(fvals)
    best_val = fvals[best_idx]

    hist_FES, hist_best = [], []
    next_save = SAVE_EVERY

    while eva.FES < FESmax:
        # Adaptation des paramètres (vectorisée)
        adapt_F = np.random.rand(N) < tau1
        F_i[adapt_F] = 0.1 + 0.9 * np.random.rand(np.sum(adapt_F))
        
        adapt_CR = np.random.rand(N) < tau2
        CR_i[adapt_CR] = np.random.rand(np.sum(adapt_CR))

        # Mutation vectorisée
        abc_indices = np.array([
            np.random.choice([j for j in range(N) if j != i], 3, replace=False)
            for i in range(N)
        ])
        
        v = x[abc_indices[:, 0]] + F_i[:, None] * (x[abc_indices[:, 1]] - x[abc_indices[:, 2]])
        v = np.clip(v, LB, UB)
        
        # Croisement vectorisé
        u = x.copy()
        j_rand = np.random.randint(0, D, N)
        mask = np.random.rand(N, D) < CR_i[:, None]
        for i in range(N):
            mask[i, j_rand[i]] = True
        u[mask] = v[mask]
        
        # Évaluation en batch
        fu, stop = eva(u)
        if stop:
            break
            
        # Sélection
        improved = fu < fvals
        x[improved] = u[improved]
        fvals[improved] = fu[improved]
        
        current_best_idx = np.argmin(fvals)
        if fvals[current_best_idx] < best_val:
            best_val = fvals[current_best_idx]

        if eva.FES >= next_save:
            hist_FES.append(eva.FES)
            hist_best.append(best_val)
            next_save += SAVE_EVERY

    return hist_FES, hist_best


# --------------------------------------------------------------------
# 3) JDE ADAPTÉE OPTIMISÉE
# --------------------------------------------------------------------
def jde_adapted_FES(func, D=DIM, N=POP_JDEA, LB=None, UB=None,
                    F0=0.5, CR0=0.9, tau1=0.1, tau2=0.1,
                    p_current_to_best=0.1,
                    ls_interval_FES=5000, ls_max_evals=30,
                    FESmax=FESMAX):

    if LB is None: LB = -100*np.ones(D)
    if UB is None: UB = 100*np.ones(D)
    LB, UB = np.asarray(LB), np.asarray(UB)

    sigma = 0.001 * (UB - LB)
    eva = Evaluator(func, FESmax)

    x = np.random.uniform(LB, UB, (N, D))
    fvals, stop = eva(x)
    if stop:
        return [], []

    F_i = np.full(N, F0)
    CR_i = np.full(N, CR0)

    best_idx = np.argmin(fvals)
    best_val = fvals[best_idx]
    best = x[best_idx].copy()

    hist_FES, hist_best = [], []
    next_save = SAVE_EVERY
    next_ls_FES = ls_interval_FES

    while eva.FES < FESmax:
        # Adaptation paramètres
        adapt_F = np.random.rand(N) < tau1
        F_i[adapt_F] = 0.1 + 0.9 * np.random.rand(np.sum(adapt_F))
        
        adapt_CR = np.random.rand(N) < tau2
        CR_i[adapt_CR] = np.random.rand(np.sum(adapt_CR))

        # Mutation (mixte current-to-best et rand/1)
        use_ctb = np.random.rand(N) < p_current_to_best
        
        abc_indices = np.array([
            np.random.choice([j for j in range(N) if j != i], 3, replace=False)
            for i in range(N)
        ])
        
        v = np.zeros_like(x)
        # current-to-best
        v[use_ctb] = (x[use_ctb] + 
                      F_i[use_ctb, None] * (best - x[use_ctb]) + 
                      F_i[use_ctb, None] * (x[abc_indices[use_ctb, 0]] - x[abc_indices[use_ctb, 1]]))
        # rand/1
        v[~use_ctb] = (x[abc_indices[~use_ctb, 0]] + 
                       F_i[~use_ctb, None] * (x[abc_indices[~use_ctb, 1]] - x[abc_indices[~use_ctb, 2]]))
        
        v = np.clip(v, LB, UB)
        
        # Croisement
        u = x.copy()
        j_rand = np.random.randint(0, D, N)
        mask = np.random.rand(N, D) < CR_i[:, None]
        for i in range(N):
            mask[i, j_rand[i]] = True
        u[mask] = v[mask]
        
        # Évaluation
        fu, stop = eva(u)
        if stop:
            break
            
        improved = fu < fvals
        x[improved] = u[improved]
        fvals[improved] = fu[improved]
        
        best_idx = np.argmin(fvals)
        if fvals[best_idx] < best_val:
            best_val = fvals[best_idx]
            best = x[best_idx].copy()

        if eva.FES >= next_save:
            hist_FES.append(eva.FES)
            hist_best.append(best_val)
            next_save += SAVE_EVERY

        # Local search (batch)
        if eva.FES >= next_ls_FES and eva.FES < FESmax:
            # Générer tous les candidats d'un coup
            n_ls = min(ls_max_evals, FESmax - eva.FES)
            noise = np.random.randn(n_ls, D) * sigma
            candidates = np.clip(best + noise, LB, UB)
            
            f_cands, stop = eva(candidates)
            if not stop:
                best_ls_idx = np.argmin(f_cands)
                if f_cands[best_ls_idx] < best_val:
                    best_val = f_cands[best_ls_idx]
                    best = candidates[best_ls_idx].copy()
                    
                    worst = np.argmax(fvals)
                    x[worst] = best.copy()
                    fvals[worst] = best_val
            
            next_ls_FES += ls_interval_FES

    return hist_FES, hist_best


# --------------------------------------------------------------------
# 4) PSO OPTIMISÉ (déjà assez vectorisé)
# --------------------------------------------------------------------
def pso_FES(func, D=DIM, N=POP_PSO, LB=None, UB=None,
            w=0.7, c1=1.5, c2=1.5, FESmax=FESMAX):

    if LB is None: LB = -100*np.ones(D)
    if UB is None: UB = 100*np.ones(D)
    LB, UB = np.asarray(LB), np.asarray(UB)

    eva = Evaluator(func, FESmax)

    x = np.random.uniform(LB, UB, (N, D))
    v = np.zeros((N, D))

    fvals, stop = eva(x)
    if stop:
        return [], []

    pbest = x.copy()
    pbest_val = fvals.copy()

    gbest_idx = np.argmin(fvals)
    gbest = x[gbest_idx].copy()
    gbest_val = fvals[gbest_idx]

    hist_FES, hist_best = [], []
    next_save = SAVE_EVERY

    while eva.FES < FESmax:
        r1 = np.random.rand(N, D)
        r2 = np.random.rand(N, D)

        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x = x + v
        x = np.clip(x, LB, UB)

        fvals, stop = eva(x)
        if stop:
            break

        improved = fvals < pbest_val
        pbest[improved] = x[improved]
        pbest_val[improved] = fvals[improved]

        gbest_idx = np.argmin(pbest_val)
        if pbest_val[gbest_idx] < gbest_val:
            gbest_val = pbest_val[gbest_idx]
            gbest = pbest[gbest_idx].copy()

        if eva.FES >= next_save:
            hist_FES.append(eva.FES)
            hist_best.append(gbest_val)
            next_save += SAVE_EVERY

    return hist_FES, hist_best


# --------------------------------------------------------------------
# 5) ABC OPTIMISÉ
# --------------------------------------------------------------------
def abc_FES(func, D=DIM, N=POP_ABC, LB=None, UB=None,
            limit=50, FESmax=FESMAX):

    if LB is None: LB = -100*np.ones(D)
    if UB is None: UB = 100*np.ones(D)
    LB, UB = np.asarray(LB), np.asarray(UB)

    eva = Evaluator(func, FESmax)

    X = np.random.uniform(LB, UB, (N, D))
    fvals, stop = eva(X)
    if stop:
        return [], []

    trials = np.zeros(N, dtype=int)

    hist_FES, hist_best = [], []
    best_val = np.min(fvals)
    next_save = SAVE_EVERY

    while eva.FES < FESmax:
        # Employed bees - vectorisé
        k_indices = np.array([np.random.choice([j for j in range(N) if j != i]) 
                              for i in range(N)])
        j_indices = np.random.randint(0, D, N)
        phi = np.random.uniform(-1, 1, N)
        
        V = X.copy()
        for i in range(N):
            V[i, j_indices[i]] = X[i, j_indices[i]] + phi[i] * (X[i, j_indices[i]] - X[k_indices[i], j_indices[i]])
        V = np.clip(V, LB, UB)
        
        fv, stop = eva(V)
        if stop:
            break
            
        improved = fv < fvals
        X[improved] = V[improved]
        fvals[improved] = fv[improved]
        trials[improved] = 0
        trials[~improved] += 1

        # Onlooker bees
        fit_vals = 1 / (1 + fvals)
        probs = fit_vals / np.sum(fit_vals)
        
        selected = np.random.choice(N, size=N, p=probs)
        k_indices = np.array([np.random.choice([j for j in range(N) if j != s]) 
                              for s in selected])
        j_indices = np.random.randint(0, D, N)
        phi = np.random.uniform(-1, 1, N)
        
        V = X[selected].copy()
        for idx, i in enumerate(selected):
            V[idx, j_indices[idx]] = X[i, j_indices[idx]] + phi[idx] * (X[i, j_indices[idx]] - X[k_indices[idx], j_indices[idx]])
        V = np.clip(V, LB, UB)
        
        fv, stop = eva(V)
        if stop:
            break
            
        for idx, i in enumerate(selected):
            if fv[idx] < fvals[i]:
                X[i] = V[idx]
                fvals[i] = fv[idx]
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout bees
        scouts = trials > limit
        n_scouts = np.sum(scouts)
        if n_scouts > 0:
            X[scouts] = np.random.uniform(LB, UB, (n_scouts, D))
            fs, stop = eva(X[scouts])
            if stop:
                break
            fvals[scouts] = fs
            trials[scouts] = 0

        current_best = np.min(fvals)
        if current_best < best_val:
            best_val = current_best

        if eva.FES >= next_save:
            hist_FES.append(eva.FES)
            hist_best.append(best_val)
            next_save += SAVE_EVERY

    return hist_FES, hist_best


# --------------------------------------------------------------------
# 6) GSA OPTIMISÉ
# --------------------------------------------------------------------
def gsa_FES(func, D=DIM, N=POP_GSA, LB=None, UB=None,
            G0=100, alpha=20, FESmax=FESMAX):

    if LB is None: LB = -100*np.ones(D)
    if UB is None: UB = 100*np.ones(D)
    LB, UB = np.asarray(LB), np.asarray(UB)

    eva = Evaluator(func, FESmax)

    X = np.random.uniform(LB, UB, (N, D))
    V = np.zeros((N, D))

    fvals, stop = eva(X)
    if stop:
        return [], []

    hist_FES, hist_best = [], []
    best_val = np.min(fvals)
    next_save = SAVE_EVERY

    t = 0
    while eva.FES < FESmax:
        t += 1
        G = G0 * np.exp(-alpha * t / (FESmax / N))

        worst = np.max(fvals)
        best = np.min(fvals)
        if worst == best:
            M = np.ones(N) / N
        else:
            m_raw = (worst - fvals) / (worst - best + 1e-12)
            M = m_raw / (np.sum(m_raw) + 1e-12)

        # Forces vectorisées
        F = np.zeros((N, D))
        for i in range(N):
            diffs = X - X[i]
            dists = np.linalg.norm(diffs, axis=1) + 1e-12
            rand = np.random.rand(N, D)
            forces = rand * G * M[i] * M[:, None] * diffs / dists[:, None]
            F[i] = np.sum(forces, axis=0)

        a = F / (M[:, None] + 1e-12)
        rand2 = np.random.rand(N, D)
        V = rand2 * V + a
        X = X + V
        X = np.clip(X, LB, UB)

        fvals, stop = eva(X)
        if stop:
            break

        current_best = np.min(fvals)
        if current_best < best_val:
            best_val = current_best

        if eva.FES >= next_save:
            hist_FES.append(eva.FES)
            hist_best.append(best_val)
            next_save += SAVE_EVERY

    return hist_FES, hist_best


# Le reste du code (worker, save/load, main) reste identique
def worker(algo_name, algo_fn, func, run_id):
    print(f"   ▶ {algo_name} - Run {run_id+1}/{N_RUNS} ...")
    hist_FES, hist_best = algo_fn(lambda X: func(X.reshape(X.shape[0], -1)))
    print(f"   ✔ {algo_name} - Run {run_id+1}/{N_RUNS} terminé")
    return np.array(hist_FES), np.array(hist_best)


def save_curve_fes(filename, algo_name, func_name, ftype, fes_list, best_list):
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(f"{RESULT_DIR}/{filename}", "w") as f:
        f.write(f"Algorithm: {algo_name}\n")
        f.write(f"Function: {func_name} ({ftype})\n")
        f.write("FES:\n")
        f.write(" ".join(str(int(x)) for x in fes_list) + "\n")
        f.write("Best:\n")
        f.write(" ".join(str(x) for x in best_list) + "\n")


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


if __name__ == "__main__":

    ALGOS = {
    "DEClassique": de_classique_FES,
    "JDESimple": jde_simple_FES,
    "JDEAdapted": jde_adapted_FES,
    "PSO": pso_FES,
    "PSOHybrid": pso_hybrid_FES,     # <--- AJOUTER
    "GA": ga_FES,                    # <--- AJOUTER
    "ABC": abc_FES,
    "GSA": gsa_FES,
}


    print("\n🚀 Début de l'expérience complète FES (OPTIMISÉ)...\n")

    for fname, (ftype, ffunc) in FUNC_MAP.items():

        print("\n=================================")
        print(f"  Fonction {fname} ({ftype})")
        print("=================================\n")

        for algo_name, algo_fn in ALGOS.items():

            print(f"🔵 Algorithme : {algo_name}")
            start = time.time()

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    executor.map(
                        lambda r: worker(algo_name, algo_fn, ffunc, r),
                        range(N_RUNS)
                    )
                )

            duration = time.time() - start
            print(f"⏳ Temps total pour {algo_name} sur {fname} : {duration:.2f} sec\n")

            max_len = max(len(r[1]) for r in results)
            all_fes = []
            all_best = []

            for fes_list, best_list in results:
                if len(best_list) < max_len:
                    best_list = np.concatenate([
                        best_list,
                        np.full(max_len - len(best_list), best_list[-1])
                    ])

                if len(fes_list) < max_len:
                    last = fes_list[-1]
                    extra = np.arange(last + SAVE_EVERY,
                                      last + SAVE_EVERY*(max_len - len(fes_list)) + 1,
                                      SAVE_EVERY)
                    fes_list = np.concatenate([fes_list, extra])

                all_fes.append(fes_list)
                all_best.append(best_list)

            mean_fes = all_fes[0]
            mean_best = np.mean(all_best, axis=0)

            filename = f"{algo_name}_{fname}_{ftype}.txt"
            save_curve_fes(filename, algo_name, fname, ftype, mean_fes, mean_best)

            print(f"✔ Fichier sauvegardé : {RESULT_DIR}/{filename}\n")

    print("\n🎉 EXPÉRIMENTATION TERMINÉE — FICHIERS FES DANS resultsss/ 🎉")

    print("\n📊 Génération des graphiques...\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    idx = 0
    for fname, (ftype, ffunc) in FUNC_MAP.items():
        ax = axes[idx]
        idx += 1

        for algo_name in ALGOS.keys():
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