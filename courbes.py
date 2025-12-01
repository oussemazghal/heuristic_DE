import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================================================
#   UTILITAIRE : COMPTEUR D’ÉVALUATIONS (FES)
# ======================================================================

class Evaluator:
    def __init__(self, func, FESmax):
        self.func = func
        self.FES = 0
        self.FESmax = FESmax

    def __call__(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            n = 1
        else:
            n = X.shape[0]

        if self.FES + n > self.FESmax:
            return None, True

        self.FES += n
        return self.func(X), False


# ======================================================================
#   1) DE CLASSIQUE — N = 60
# ======================================================================

def de_classique_FES(func, D=30, N=60, LB=None, UB=None, 
                     F=0.8, CR=0.9, FESmax=30000):

    if LB is None:
        LB = -100 * np.ones(D)
    if UB is None:
        UB = 100 * np.ones(D)

    LB = np.asarray(LB)
    UB = np.asarray(UB)

    eva = Evaluator(func, FESmax)

    x = np.random.uniform(LB, UB, (N, D))
    fvals, stop = eva(x)

    best_idx = np.argmin(fvals)
    best = x[best_idx].copy()
    best_val = fvals[best_idx]

    history_FES = []
    history_best = []

    while True:
        new_pop = np.zeros_like(x)

        for i in range(N):
            if eva.FES >= FESmax:
                return history_FES, history_best, best, best_val

            candidates = [j for j in range(N) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            v = x[a] + F * (x[b] - x[c])
            v = np.clip(v, LB, UB)

            u = x[i].copy()
            j_rand = np.random.randint(D)
            for j in range(D):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            fu, stop = eva(u[None, :])
            if stop:
                return history_FES, history_best, best, best_val
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
            history_FES.append(eva.FES)
            history_best.append(best_val)


# ======================================================================
#   2) JDE SIMPLE — N = 40
# ======================================================================

def jde_simple_FES(func, D=30, N=40, LB=None, UB=None,
                   F0=0.5, CR0=0.9, tau1=0.1, tau2=0.1,
                   FESmax=30000):

    if LB is None:
        LB = -100 * np.ones(D)
    if UB is None:
        UB = 100 * np.ones(D)

    LB = np.asarray(LB)
    UB = np.asarray(UB)

    eva = Evaluator(func, FESmax)

    x = np.random.uniform(LB, UB, (N, D))
    fvals, stop = eva(x)

    F_i = np.full(N, F0)
    CR_i = np.full(N, CR0)

    best_idx = np.argmin(fvals)
    best = x[best_idx].copy()
    best_val = fvals[best_idx]

    history_FES = []
    history_best = []

    while True:
        new_pop = np.zeros_like(x)

        for i in range(N):
            if eva.FES >= FESmax:
                return history_FES, history_best, best, best_val

            if np.random.rand() < tau1:
                F_i[i] = 0.1 + 0.9 * np.random.rand()

            if np.random.rand() < tau2:
                CR_i[i] = np.random.rand()

            F = F_i[i]
            CR = CR_i[i]

            candidates = [j for j in range(N) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            v = x[a] + F * (x[b] - x[c])
            v = np.clip(v, LB, UB)

            u = x[i].copy()
            j_rand = np.random.randint(D)
            for j in range(D):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            fu, stop = eva(u[None, :])
            if stop:
                return history_FES, history_best, best, best_val
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
            history_FES.append(eva.FES)
            history_best.append(best_val)


# ======================================================================
#   3) JDE ADAPTÉE — N = 30
# ======================================================================

def jde_adapted_FES(func, D=30, N=30, LB=None, UB=None,
                    F0=0.5, CR0=0.9, tau1=0.1, tau2=0.1,
                    p_current_to_best=0.1,
                    ls_interval_FES=5000, ls_max_evals=30,
                    FESmax=30000):

    if LB is None:
        LB = -100 * np.ones(D)
    if UB is None:
        UB = 100 * np.ones(D)

    LB = np.asarray(LB)
    UB = np.asarray(UB)

    sigma = 0.001 * (UB - LB)

    eva = Evaluator(func, FESmax)

    x = np.random.uniform(LB, UB, (N, D))
    fvals, stop = eva(x)

    F_i = np.full(N, F0)
    CR_i = np.full(N, CR0)

    best_idx = np.argmin(fvals)
    best = x[best_idx].copy()
    best_val = fvals[best_idx]

    history_FES = []
    history_best = []

    next_ls_FES = ls_interval_FES

    while True:
        new_pop = x.copy()

        for i in range(N):
            if eva.FES >= FESmax:
                return history_FES, history_best, best, best_val

            if np.random.rand() < tau1:
                F_i[i] = 0.1 + 0.9 * np.random.rand()
            if np.random.rand() < tau2:
                CR_i[i] = np.random.rand()

            F = F_i[i]
            CR = CR_i[i]

            candidates = [j for j in range(N) if j != i]

            if np.random.rand() < p_current_to_best:
                a, b = np.random.choice(candidates, 2, replace=False)
                v = x[i] + F * (best - x[i]) + F * (x[a] - x[b])
            else:
                a, b, c = np.random.choice(candidates, 3, replace=False)
                v = x[a] + F * (x[b] - x[c])

            v = np.clip(v, LB, UB)

            u = x[i].copy()
            j_rand = np.random.randint(D)
            for j in range(D):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            fu, stop = eva(u[None, :])
            if stop:
                return history_FES, history_best, best, best_val
            fu = fu[0]

            if fu < fvals[i]:
                new_pop[i] = u
                fvals[i] = fu
                if fu < best_val:
                    best_val = fu
                    best = u.copy()

        x = new_pop

        if eva.FES % 1000 == 0:
            history_FES.append(eva.FES)
            history_best.append(best_val)

        if eva.FES >= next_ls_FES and eva.FES < FESmax:

            used = 0
            while used < ls_max_evals and eva.FES < FESmax:
                noise = np.random.randn(D) * sigma
                cand = np.clip(best + noise, LB, UB)

                f_cand, stop = eva(cand[None, :])
                if stop:
                    return history_FES, history_best, best, best_val

                f_cand = f_cand[0]
                used += 1

                if f_cand < best_val:
                    best_val = f_cand
                    best = cand.copy()

            worst_idx = np.argmax(fvals)
            x[worst_idx] = best.copy()
            fvals[worst_idx] = best_val

            next_ls_FES += ls_interval_FES



# ======================================================================
#   ENREGISTREMENT DANS FICHIERS TXT
# ======================================================================

def save_run_to_file(alg_name, func_name, run_id, best_val, hist_FES, hist_best):
    os.makedirs(f"results/{alg_name}", exist_ok=True)

    filename = f"results/{alg_name}/{func_name}_run{run_id}.txt"

    with open(filename, "w") as f:
        f.write(f"Algorithme : {alg_name}\n")
        f.write(f"Fonction : {func_name}\n")
        f.write(f"Run : {run_id}\n")
        f.write(f"Meilleur résultat final : {best_val}\n")
        f.write("\n--- HISTORIQUE CONVERGENCE (FES, best) ---\n")

        for fes, bv in zip(hist_FES, hist_best):
            f.write(f"{fes}\t{bv}\n")



# ======================================================================
#   EXPÉRIMENTATION COMPLÈTE (AVEC PRINTS AJOUTÉS)
# ======================================================================

def run_complete_experiment():
    try:
        from cec2017.functions import (
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
            f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
            f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
        )

        functions = [
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
            f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
            f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
        ]
        names = [f"f{i}" for i in range(1, 31)]

        D = 30
        FESmax = 30000
        N_RUNS = 30

        algorithms = [
            ("DE_Classique", de_classique_FES),
            ("JDE_Simple", jde_simple_FES),
            ("JDE_Adaptee", jde_adapted_FES),
        ]

        results = {alg_name: {"mean": [], "std": []} for alg_name, _ in algorithms}

        print("\n====================== DÉBUT EXPÉRIMENTATION ======================\n")

        for func_idx, func in enumerate(functions):
            func_name = names[func_idx]

            print(f"\n\n====================== Fonction {func_name} ======================\n")

            for alg_name, alg_func in algorithms:

                print(f"\n--- Algorithme : {alg_name} ---\n")

                best_values = []

                for run in range(N_RUNS):

                    hist_FES, hist_best, best, best_val = alg_func(
                        func, D=D, FESmax=FESmax
                    )

                    best_values.append(best_val)

                    print(f"Run {run+1:02d}/30 → best = {best_val:.4E}")

                    save_run_to_file(
                        alg_name,
                        func_name,
                        run + 1,
                        best_val,
                        hist_FES,
                        hist_best
                    )

                mean_val = np.mean(best_values)
                std_val = np.std(best_values)

                results[alg_name]["mean"].append(mean_val)
                results[alg_name]["std"].append(std_val)

                print(f"\n>>> Résultat {alg_name} sur {func_name} : "
                      f"{mean_val:.4E} ± {std_val:.4E}\n")

        print("\n====================== RÉSUMÉ FINAL ======================\n")

        for alg_name in results:
            print(f"\n### {alg_name} ###")
            for i in range(30):
                m = results[alg_name]["mean"][i]
                s = results[alg_name]["std"][i]
                print(f"f{i+1:02d} : {m:.4E} ± {s:.4E}")

        return results

    except ImportError:
        print("CEC2017 non installé: pip install cec2017")
        return None



# ======================================================================
#   SAUVEGARDE : UN FICHIER PAR ALGORITHME (MOYENNE ± STD)
# ======================================================================

def save_summary_results(results):
    os.makedirs("results_summary", exist_ok=True)

    for alg_name, stats in results.items():
        filename = f"results_summary/{alg_name}_summary.txt"

        with open(filename, "w") as f:
            f.write(f"Résumé des résultats pour : {alg_name}\n")
            f.write("=" * (30 + len(alg_name)) + "\n\n")

            mean_list = stats["mean"]
            std_list = stats["std"]

            for i in range(30):
                mean_val = mean_list[i]
                std_val = std_list[i]
                f.write(f"f{i+1:02d} : {mean_val:.6E} ± {std_val:.6E}\n")

        print(f"[OK] Fichier résumé généré → {filename}")



# ======================================================================
#   MAIN
# ======================================================================

if __name__ == "__main__":
    
    results = run_complete_experiment()

    if results:
        save_summary_results(results)

    print("Expérimentation terminée. Résumés dans ./results_summary/")
