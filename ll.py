import os
import shutil

# Dossier source et destination
source_dir = r"C:\Users\pc\Downloads\DE classique\tous\resultsss"
destination_dir = r"C:\Users\pc\Downloads\cccc"

# Créer le dossier destination s'il n'existe pas
os.makedirs(destination_dir, exist_ok=True)

# Liste des fichiers à copier
files = [
    "ABC_f2_unimodal.txt",
    "ABC_f4_unimodal.txt",
    "ABC_f12_hybrid.txt",
    "ABC_f21_composition.txt",
    "GA_f2_unimodal.txt",
    "GA_f4_unimodal.txt",
    "GA_f12_hybrid.txt",
    "GA_f21_composition.txt",
    "GSA_f2_unimodal.txt",
    "GSA_f4_unimodal.txt",
    "GSA_f12_hybrid.txt",
    "GSA_f21_composition.txt",
    "PSO_f2_unimodal.txt",
    "PSO_f4_unimodal.txt",
    "PSO_f12_hybrid.txt",
    "PSO_f21_composition.txt",
    "PSOHybrid_f2_unimodal.txt",
    "PSOHybrid_f4_unimodal.txt",
    "PSOHybrid_f12_hybrid.txt",
    "PSOHybrid_f21_composition.txt"
]

# Copier chaque fichier
for file in files:
    source_path = os.path.join(source_dir, file)
    destination_path = os.path.join(destination_dir, file)
    
    if os.path.exists(source_path):
        shutil.copy2(source_path, destination_path)
        print(f"✓ Copié: {file}")
    else:
        print(f"✗ Fichier non trouvé: {file}")

print("\nTerminé!")