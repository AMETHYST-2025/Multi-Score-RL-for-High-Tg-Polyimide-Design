import sys, os
from rdkit import Chem
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rdkit.Chem import AllChem, DataStructs, rdDepictor, QED, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from itertools import combinations
import seaborn as sns
import io
import xlsxwriter
from PIL import Image
from pathlib import Path
from matplotlib_venn import venn2
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


#import data_structs as ds
from reinvent_benchmarking.data_structs_n import canonicalize_smiles_from_file, construct_vocabulary, write_smiles_to_file
from reinvent_benchmarking.train_prior import pretrain
from reinvent_benchmarking.train_agent import train_agent


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ROOT_DIR = '..'
sys.path.append(ROOT_DIR)
sys.path.append('reinvent_benchmarking')
#from tartarus import pce
def fitness_function(smi: str):
    dipole, hl_gap, lumo, obj, pce_1, pce_2, sas = pce.get_properties(smi)
    return pce_1 - sas


#-----------------------------------------------------------

data_path = os.path.join(ROOT_DIR, 'datasets')
print(data_path)
filename = 'Polyimides_synthetic.csv'  #'hce.csv'
sep = ','
header = 'infer'
smile_name = 'SMILES' #'smiles'
# dataset load
#fname = os.path.join(data_path, filename)
fname = "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/datasets/Polyimides_synthetic.csv"
data = pd.read_csv(fname, sep=sep, header=header)
headers =  ["SMILES", "Tg"]
data.columns = headers
data = data[headers][:500000]
print(data)
smiles = data[smile_name]
print(data.sort_values(by="Tg", ascending=False).head(20))
#-----------------------------------------------------------

if not os.path.isdir('data'):
    os.mkdir('data')
# create smi file
with open(os.path.join('data', 'data.smi'), 'w') as f:
    for smi in smiles:
        f.write(smi+'\n')
smiles_file = 'data/data.smi'
print("Reading smiles...")
smiles_list = canonicalize_smiles_from_file(smiles_file)
print("Constructing vocabulary...")
voc_chars = construct_vocabulary(smiles_list)
write_smiles_to_file(smiles_list, "data/mols_filtered.smi")

#-----------------------------------------------------------
num_epochs = 3 #100
verbose = False
train_ratio = 0.8
pretrain(num_epochs=num_epochs, verbose=verbose, train_ratio=train_ratio) #, restore_from = "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/models_jupyter/data/Prior_PI_50k_epochs_8.ckpt"


#from tartarus import docking
for exp in [0]:
    train_agent(
        scoring_function = 'merge_score', #'Tg_score' , 'tanimoto', 'ProgressiveTgScorer', 'merge_score'
        batch_size = 500,
        n_steps = 100,
        experience_replay=exp
    )


#***********************************************************************************************************************
#***************************************************Plots functions*****************************************************
#***********************************************************************************************************************



# Sous-échantillon du trainset pour équilibrer la visualisation, 50 k trop grand !
data_sample = data.sample(n=2000, random_state=42)
#print(data_sample)
from reinvent_benchmarking.Pred_Tg.PI_Tg_Pred import run_PolyBERT, finetune_PolyBERT, Pred_PI_Tg
#
#rmse = np.sqrt(np.mean((data_sample['Tg_pred'] - data_sample['Tg']) ** 2))
#mae = np.mean(np.abs(data_sample['Tg_pred'] - data_sample['Tg']))
#print(rmse, mae) #47.47540132795242 30.640225479125977
#-----------------------------------------------------------

PI_generation = pd.read_csv("/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/results/run_2025-06-29_MeanTanimoto/results.csv")
#print(PI_generation)
#-----------------------------------------------------------------------------------------------------------------------
def count_sulfur_groups(smiles_list):
    """
    Compte le pourcentage de SMILES contenant S=O (sulfone, sulfoxyde)
    et ceux contenant O=S=O (groupements sulfonyle/sulfonate).
    """
    total = len(smiles_list)
    if total == 0:
        return {"S=O only (%)": 0.0, "O=S=O (%)": 0.0, "Aucun (%)": 0.0}
    # SMARTS patterns
    pattern_s_o = Chem.MolFromSmarts("[SX4](=O)")     # S=O (sulfone/sulfoxyde)
    pattern_o_s_o = Chem.MolFromSmarts("[SX4](=O)(=O)")  # O=S=O (sulfonyle)
    s_o_only = 0
    o_s_o = 0
    none = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # skip invalid SMILES

        has_s_o = mol.HasSubstructMatch(pattern_s_o)
        has_o_s_o = mol.HasSubstructMatch(pattern_o_s_o)

        if has_o_s_o:
            o_s_o += 1
        elif has_s_o:
            s_o_only += 1
        else:
            none += 1

    return {
        "S=O only (%)": 100 * s_o_only / total,
        "O=S=O (%)": 100 * o_s_o / total,
        "Aucun (%)": 100 * none / total
    }
#res = count_sulfur_groups(smiles)
#print(res)


def contains_substructure(smiles_list, pattern_smiles):
    """
    Vérifie si les SMILES de la liste contiennent un sous-motif donné.
    Retourne aussi les pourcentages de True/False.
    """
    # Conversion en liste Python si c'est une Series Pandas
    smiles_list = list(smiles_list)
    pattern = Chem.MolFromSmiles(pattern_smiles)
    if pattern is None:
        raise ValueError("Le motif fourni n'est pas un SMILES valide")
    results = {}
    count_true, count_false, count_invalid = 0, 0, 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results[smi] = None  # SMILES invalide
            count_invalid += 1
        else:
            match = mol.HasSubstructMatch(pattern)
            results[smi] = match
            if match:
                count_true += 1
            else:
                count_false += 1
    total_valid = count_true + count_false
    percentages = {
        "True (%)": 100 * count_true / total_valid if total_valid > 0 else 0,
        "False (%)": 100 * count_false / total_valid if total_valid > 0 else 0,
        "Invalid (%)": 100 * count_invalid / len(smiles_list) if len(smiles_list) > 0 else 0
    }
    print(percentages)
    return results, percentages
pattern = "O=C1NC(=O)C2=CC3=C(C=C12)C(=O)NC3=O"
#print(contains_substructure(smiles, pattern))


#****************************************************Plot multiple histogram********************************************
#***********************************************************************************************************************
# Fonction pour vérifier la validité d'un SMILES
def is_valid_smile(smile):
    return Chem.MolFromSmiles(smile) is not None
def plot_histogram_gen():
    # Liste des générations à tracer
    #generations = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 199.0]
    generations = [0.0, 40.0, 50.0, 60.0, 99.0, 0.0, 10.0, 30.0, 60.0, 90.0]

    # Préparer la figure
    fig, axes = plt.subplots(2, 5, figsize=(25, 10)) #(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    # ----------------------Histogrammes-------------------------
    for idx, gen in enumerate(generations):
        ax = axes[idx]

        # Données pour la génération en cours
        df_gen = PI_generation[PI_generation['generation'] == gen]
        df_gen['Tg_pred'] = df_gen["smiles"].apply(Pred_PI_Tg)
        tg_gen = df_gen['Tg_pred']

        # Calcul validité
        valid_count = df_gen["smiles"].apply(is_valid_smile).sum()
        total_count = len(df_gen)
        valid_pct = 100 * valid_count / total_count if total_count > 0 else 0

        # Données d'origine
        tg_train = data_sample['Tg']

        # Tracer l'histogramme
        ax.hist(tg_train, bins=30, alpha=0.5, label='Train', color='blue', density=True)
        ax.hist(tg_gen, bins=30, alpha=0.7, label=f'Gen {int(gen)}', color='orange', density=True)

        # Calcul des médianes
        median_train = np.median(tg_train)
        median_gen = np.median(tg_gen)

        # Tracer les médianes
        ax.axvline(median_gen, color='orange', linestyle='--', linewidth=2, label=f'Median {median_gen:.1f}')

        # Affichage du titre avec % de validité
        ax.set_title(f'Gen {int(gen)} - Valid: {valid_pct:.1f}%')
        ax.set_xlabel('Tg')
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    plt.show()
#plot_histogram_gen()

#****************************************************TSNE Morgan Fingerprint********************************************
#***********************************************************************************************************************
def smiles_to_morgan_fp(smile, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smile)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)  # pour les molécules invalides

def plot_TSNE_MorganFP_(data_sample,PI_generation):
    # Définir les générations à visualiser
    #generations = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 99.0]
    generations = [0.0, 40.0, 50.0, 60.0, 99.0, 0.0, 10.0, 30.0, 60.0, 90.0]

    # 1. Calcul des empreintes
    data_sample['FP'] = data_sample["SMILES"].apply(smiles_to_morgan_fp)
    X_train = np.vstack(data_sample['FP'].values)

    X_all = [X_train]
    labels_all = ['train'] * len(X_train)

    all_generations = {}

    for gen in generations:
        df_gen = PI_generation[PI_generation['generation'] == gen]
        df_gen['FP'] = df_gen["smiles"].apply(smiles_to_morgan_fp)
        X_gen = np.vstack(df_gen['FP'].values)
        X_all.append(X_gen)
        labels_all += [f'gen_{int(gen)}'] * len(X_gen)
        all_generations[gen] = X_gen

    # 2. t-SNE global
    X_all = np.vstack(X_all)
    labels_all = np.array(labels_all)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_all)

    # 3. Indices des embeddings
    tsne_points = {}
    start = 0
    tsne_points['train'] = X_tsne[start:start+len(X_train)]
    start += len(X_train)

    for gen in generations:
        gen_len = len(all_generations[gen])
        tsne_points[gen] = X_tsne[start:start+gen_len]
        start += gen_len

    # 4. Plot
    n_gen = len(generations)
    fig, axes = plt.subplots(nrows=(n_gen+1)//2, ncols=5, figsize=(15, (n_gen+1)//2 * 3))
    axes = axes.flatten()

    for idx, gen in enumerate(generations):
        ax = axes[idx]
        ax.scatter(tsne_points['train'][:, 0], tsne_points['train'][:, 1],
                   label='Train', alpha=0.3, s=10, color='gray')
        ax.scatter(tsne_points[gen][:, 0], tsne_points[gen][:, 1],
                   label=f'Gen {int(gen)}', alpha=0.7, s=10, color='royalblue')
        ax.set_title(f"t-SNE: Génération {int(gen)}")
        ax.legend()

    # Cacher les axes inutilisés si nécessaire
    for ax in axes[len(generations):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
#plot_TSNE_MorganFP_(data_sample,PI_generation)


def plot_TSNE_MorganFP(data_sample, PI_generation, target_smiles_list):
    # Définir les générations à visualiser
    generations = [0.0, 40.0, 50.0, 60.0, 99.0, 0.0, 10.0, 30.0, 60.0, 90.0]

    # 1. Calcul des empreintes
    data_sample['FP'] = data_sample["SMILES"].apply(smiles_to_morgan_fp)
    X_train = np.vstack(data_sample['FP'].values)

    X_all = [X_train]
    labels_all = ['train'] * len(X_train)

    all_generations = {}

    for gen in generations:
        df_gen = PI_generation[PI_generation['generation'] == gen].copy()
        df_gen['FP'] = df_gen["smiles"].apply(smiles_to_morgan_fp)
        X_gen = np.vstack(df_gen['FP'].values)
        X_all.append(X_gen)
        labels_all += [f'gen_{int(gen)}'] * len(X_gen)
        all_generations[gen] = X_gen

    # 1.b Empreintes des SMILES cibles
    target_fps = []
    for smi in target_smiles_list:
        fp = smiles_to_morgan_fp(smi).reshape(1, -1)
        X_all.append(fp)
        labels_all.append("target")
        target_fps.append(fp)

    # 2. t-SNE global
    X_all = np.vstack(X_all)
    labels_all = np.array(labels_all)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_all)

    # 3. Indices des embeddings
    tsne_points = {}
    start = 0
    tsne_points['train'] = X_tsne[start:start+len(X_train)]
    start += len(X_train)

    for gen in generations:
        gen_len = len(all_generations[gen])
        tsne_points[gen] = X_tsne[start:start+gen_len]
        start += gen_len

    # Récupération des embeddings des cibles
    tsne_targets = X_tsne[start:]

    # 4. Plot
    n_gen = len(generations)
    fig, axes = plt.subplots(nrows=(n_gen+1)//2, ncols=5, figsize=(15, (n_gen+1)//2 * 3))
    axes = axes.flatten()

    for idx, gen in enumerate(generations):
        ax = axes[idx]
        ax.scatter(tsne_points['train'][:, 0], tsne_points['train'][:, 1],
                   label='Train', alpha=0.3, s=10, color='gray')
        ax.scatter(tsne_points[gen][:, 0], tsne_points[gen][:, 1],
                   label=f'Gen {int(gen)}', alpha=0.7, s=10, color='royalblue')
        for i, point in enumerate(tsne_targets):
            ax.scatter(point[0], point[1],
                       label='Target' if i == 0 else None,  # Un seul label
                       color='red', s=30, edgecolors='black', linewidths=0.5)
        ax.set_title(f"t-SNE: Gen {int(gen)}")
        ax.legend()

    # Cacher les axes inutilisés si nécessaire
    for ax in axes[len(generations):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
target_smiles = ["Ic1cc2c(cc1C)c1c(S2(=O)=O)cc(c(c1)C)n1c(=O)c2c(c1=O)cc1c(c2)c(=O)n(c1=O)I", "Ic1ccc(cn1)n1c(=O)c2c(c1=O)cc1c(c2)c(=O)n(c1=O)I", "Ic1ccc(c(c1Cl)Cl)n1c(=O)c2c(c1=O)cc1c(c2)c(=O)n(c1=O)I"]
#plot_TSNE_MorganFP(data_sample,PI_generation, target_smiles)

#************************************************Plot metric unicity, divercity*****************************************
#***********************************************************************************************************************

def canonicalize_smiles(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
    return None

def get_morgan_fp(smile, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smile)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return None

def compute_intdiv(fps):
    if len(fps) < 2:
        return 0.0
    sims = [DataStructs.TanimotoSimilarity(fp1, fp2) for fp1, fp2 in combinations(fps, 2)]
    return 1 - np.mean(sims)

def compute_snn(generated_fps, train_fps):
    snn_values = []
    for gen_fp in generated_fps:
        sims = DataStructs.BulkTanimotoSimilarity(gen_fp, train_fps)
        if sims:
            snn_values.append(max(sims))  # Similarité avec le plus proche voisin
    return np.mean(snn_values) if snn_values else 0.0

def plot_all_metrics(data_sample, PI_generation):
    # Canonical training set
    generations = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 99.0]
    #generations = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 199.0]
    train_canon = data_sample["SMILES"].apply(canonicalize_smiles).dropna().unique()
    train_fps = [get_morgan_fp(s) for s in train_canon if get_morgan_fp(s)]

    metrics = {
        'Unicité externe': [],
        'Unicité interne': [],
        'IntDiv': [],
        'Fraction valide': [],
        'SNN': [],
    }

    for gen in generations:
        df_gen = PI_generation[PI_generation['generation'] == gen].copy()
        df_gen['can_smiles'] = df_gen['smiles'].apply(canonicalize_smiles)
        smiles_all = df_gen['can_smiles'].tolist()
        smiles_valid = [s for s in smiles_all if s is not None]
        unique_smiles = list(set(smiles_valid))

        # Validité
        f_valid = len(smiles_valid) / len(smiles_all) if smiles_all else 0
        metrics['Fraction valide'].append(f_valid)

        # Unicité
        uniq_ext = len([s for s in unique_smiles if s not in train_canon]) / len(unique_smiles) if unique_smiles else 0
        uniq_int = len(unique_smiles) / len(smiles_valid) if smiles_valid else 0
        metrics['Unicité externe'].append(uniq_ext)
        metrics['Unicité interne'].append(uniq_int)

        # Fingerprints
        gen_fps = [get_morgan_fp(s) for s in unique_smiles if get_morgan_fp(s)]
        intdiv = compute_intdiv(gen_fps) if gen_fps else 0
        metrics['IntDiv'].append(intdiv)

        # SNN
        snn = compute_snn(gen_fps, train_fps) if gen_fps else 0
        metrics['SNN'].append(snn)

    # Plot
    plt.figure(figsize=(14, 7))
    for name, values in metrics.items():
        plt.plot(generations, values, label=name, marker='o')

    plt.xlabel("Génération")
    plt.ylabel("Valeur métrique")
    plt.title("Évolution des métriques de qualité des polymères générés")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()
#plot_all_metrics(data, PI_generation)

#****************************************************Plot histogram vertical********************************************
#***********************************************************************************************************************

def plot_tg_distribution_by_generation(data_sample, PI_generation, generations, metric_func):
    """
    Affiche l'évolution des distributions Tg avec histogrammes inversés en horizontal.
    """
    # Préparer les données d'entraînement
    data_sample = data_sample.copy()
    data_sample['Tg_pred'] = data_sample['SMILES'].apply(metric_func)
    tg_train = data_sample['Tg_pred'].dropna().values

    # Création de la figure
    fig, ax = plt.subplots(figsize=(16, 6))
    #spacing = 200  # espace horizontal entre les histogrammes
    scale = 4000     # facteur pour rendre visible les barres (à ajuster selon besoin)
    colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))

    # Histogramme du jeu d'entraînement
    counts, bins = np.histogram(tg_train, bins=30, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.barh(bin_centers, counts * scale, height=np.diff(bins),
            left=0, color='gray', alpha=0.6, label='Train')
    median_points = []
    # Histogrammes des générations
    for i, gen in enumerate(generations):
        df_gen = PI_generation[PI_generation['generation'] == gen].copy()
        if df_gen.empty:
            continue

        df_gen['Tg_pred'] = df_gen['smiles'].apply(metric_func)
        tg_gen = df_gen['Tg_pred'].dropna().values
        if len(tg_gen) == 0:
            continue

        #x_offset = spacing * (i + 1)
        counts, bins = np.histogram(tg_gen, bins=30, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax.barh(bin_centers, counts * scale, height=np.diff(bins),
                left=0, color=colors[i], alpha=0.6, #left=x_offset
                label=f'Gen {int(gen)}')

        # Ligne médiane
        median = np.median(tg_gen)
        median_points.append((0, median)) #append((x_offset
        ax.plot([0]*2, [median - 5, median + 5], color='black', lw=2) #[x_offset]*2

        # Texte au-dessus
        ax.text(0, ax.get_ylim()[1] + 10, f'Gen {int(gen)}', #ax.text(x_offset
                ha='center', va='bottom', fontsize=9)
    if median_points:
        x_vals, y_vals = zip(*median_points)
        ax.plot(x_vals, y_vals, color='blue', linestyle='--', linewidth=2, marker='o', label='Median Tg')

    # Configuration de la figure
    ax.set_ylabel("Predicted Tg (K)")
    ax.set_xlabel("")
    ax.set_title("")
    ax.legend()
    plt.tight_layout()
    plt.show()

generations = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 99.0]
#plot_tg_distribution_by_generation(data_sample, PI_generation, generations, metric_func=Pred_PI_Tg)

#***********************************************************************************************************************
#*****************************Plot de l'evolution des bits qui influence la Tg avec les steps***************************

def plot_bit_frequencies_over_generations(data_sample, PI_generation, B):
    from collections import Counter

    generations = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 99.0]
    bit_frequencies = {bit: [] for bit in B}

    for gen in generations:
        df_gen = PI_generation[PI_generation['generation'] == gen].copy()
        df_gen['can_smiles'] = df_gen['smiles'].apply(canonicalize_smiles)
        smiles_all = df_gen['can_smiles'].dropna().unique()

        # Compte les bits présents dans les fingerprints
        bit_counts = Counter()
        total_mols = 0

        for smi in smiles_all:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                on_bits = list(fp.GetOnBits())
                bit_counts.update(on_bits)
                total_mols += 1

        # Calcule la fréquence pour chaque bit de B
        for bit in B:
            freq = bit_counts[bit] / total_mols if total_mols > 0 else 0
            bit_frequencies[bit].append(freq)

    # Plot
    plt.figure(figsize=(16, 8))
    for bit, freqs in bit_frequencies.items():
        plt.plot(generations, freqs, label=f'Bit {bit}', marker='o')

    plt.xlabel("Génération")
    plt.ylabel("Fréquence d’occurrence")
    plt.title("Évolution de la fréquence des bits Morgan (r=2) dans les polymères générés")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#B = [552, 1357, 350, 1571, 1039, 1411, 59, 863, 1691, 1283, 1391, 785, 1011, 28, 695, 408, 1160, 1617, 1370, 9] #score
B = [350, 1571, 1691, 552, 1411, 1074, 9, 863, 216, 59, 1453, 680, 1391, 252, 1076, 1039, 330, 1839, 719, 879] # Tg
#plot_bit_frequencies_over_generations(data_sample, PI_generation, B)


#***********************************************************************************************************************
def analyze_multiple_Metrics_generations(list_csv_paths, data_sample, doc_save, label):
    os.makedirs(doc_save, exist_ok=True)
    Tg_threshold = 750
    # Canonical training set
    train_canon = data_sample["SMILES"].apply(canonicalize_smiles).dropna().unique()
    train_fps = [get_morgan_fp(s) for s in train_canon if get_morgan_fp(s)]

    summary_rows = []
    summary_path = os.path.join(doc_save, "best_steps_summary_e.csv")
    #df_summary = pd.read_csv(summary_path)

    for file_path, l in zip(list_csv_paths, label):
        df_gen_all = pd.read_csv(file_path)
        generations = sorted(df_gen_all['generation'].unique())

        metrics_all = []
        zero = 0
        save_gen = pd.read_csv(file_path)

        for gen in generations:
            df_gen = df_gen_all[df_gen_all['generation'] == gen].copy()
            df_gen['can_smiles'] = df_gen['smiles'].apply(canonicalize_smiles)

            df_gen['duplicate'] = df_gen['can_smiles'].apply(lambda s: 1 if s in train_canon else 0)

            df_gen["Tg"] = df_gen["smiles"].apply(Pred_PI_Tg)
            df_gen = df_gen[df_gen["Tg"] >= Tg_threshold]

            smiles_all = df_gen['can_smiles'].tolist()
            smiles_valid = [s for s in smiles_all if s is not None]
            unique_smiles = list(set(smiles_valid))

            f_valid = len(smiles_valid) / len(smiles_all) if smiles_all else 0
            uniq_ext = len([s for s in unique_smiles if s not in train_canon]) / len(
                unique_smiles) if unique_smiles else 0
            uniq_int = len(unique_smiles) / len(smiles_valid) if smiles_valid else 0
            gen_fps = [get_morgan_fp(s) for s in unique_smiles if get_morgan_fp(s)]
            intdiv = compute_intdiv(gen_fps) if gen_fps else 0
            snn = compute_snn(gen_fps, train_fps) if gen_fps else 0
            n_smiles = len(smiles_valid)

            #rate = (uniq_int + intdiv + len(smiles_valid) + (1 - snn)) / 4
            rate = (
                    0.3 * uniq_int +
                    0.2 * intdiv +
                    0.3 * f_valid +
                    0.1 * (1 - snn) +
                    0.1 * n_smiles * uniq_ext
            )

            metrics_all.append({
                "generation": gen,
                "Unicité externe": uniq_ext,
                "Unicité interne": uniq_int,
                "IntDiv": intdiv,
                "Fraction valide": f_valid,
                "SNN": snn,
                "Nbr_Smiles": len(smiles_all),
                "rate": rate
            })

            if zero <= rate:
                zero = rate
                save_gen = df_gen


        df_metrics = pd.DataFrame(metrics_all).sort_values(by="rate", ascending=False)

        # Save metrics per file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        metrics_path = os.path.join(doc_save, f"{l}.csv") #f"{l}_e.csv
        df_metrics.to_csv(metrics_path, index=False)
        metrics_path_data = os.path.join(doc_save, f"{l}_data.csv") #data_e.csv
        save_gen.to_csv(metrics_path_data, index=False)

        # Best step summary
        best_row = df_metrics.iloc[0].copy()
        best_row['file'] = l
        best_row['best_step'] = best_row['generation']
        summary_rows.append(best_row)


    # Save best step summary
    df_summary_new = pd.DataFrame(summary_rows)
    #df_summary_fusion = pd.concat([df_summary, df_summary_new], ignore_index=True)
    summary_path = os.path.join(doc_save, "best_steps_summary_I_2.csv")
    #df_summary_fusion.to_csv(summary_path, index=False)
    df_summary_new.to_csv(summary_path, index=False)

    print(f"Résultats sauvegardés dans : {doc_save}")

list_csv_paths = []
list_csv_files =["run_2025-07-05_SI", "run_2025-07-05_SI_O", "run_2025-07-05_SI_2",
                 "run_2025-07-05_SII", "run_2025-07-05_SII_O",
                 "run_2025-07-05_SIII", "run_2025-07-05_SIII_O"]

label =         ["SI", "SI°","SI_2",
                 "SII", "SII°",
                 "SIII", "SIII°"]


"""
list_csv_files =[#"run_2025-07-03_TgHighTg_Mean_0.1", "run_2025-07-03_TgHighTg_Mean_0.2",
                 "run_2025-07-04_TgHighTg_Mean_0.3", "run_2025-07-04_TgHighTg_Mean_0.4",
                 "run_2025-07-04_TgHighTg_Mean_0.6", "run_2025-07-04_TgHighTg_Mean_0.7",
                 "run_2025-07-04_TgHighTg_Mean_0.8", "run_2025-07-04_TgHighTg_Mean_0.9"]

label =         [#"Mean_0.1_STg_Sh", "Mean_0.2_STg_Sh",
                 "Mean_0.3_STg_Sh", "Mean_0.4_STg_Sh",
                 "Mean_0.6_STg_Sh", "Mean_0.7_STg_Sh",
                 "Mean_0.8_STg_Sh", "Mean_0.9_STg_Sh"]
"""


"""
list_csv_files =["run_2025-06-29_Tg", "run_2025-06-29_Tanimoto",
                 "run_2025-06-29_RandTanimoto", "run_2025-06-29_MeanTanimoto",
                 "run_2025-06-29_TgTanimoto_Mean", "run_2025-06-29_TgTanimoto_GeoMean", "run_2025-06-29_TgTanimoto_HarmMean", "run_2025-06-29_TgTanimoto_ExpMean",
                 "run_2025-06-29_TgRanTanimoto_Mean", "run_2025-06-29_TgRanTanimoto_GeoMean", "run_2025-06-29_TgRanTanimoto_HarmMean", "run_2025-06-29_TgRanTanimoto_ExpMean",
                 "run_2025-06-29_TgMeanTanimoto_Mean", "run_2025-06-29_TgMeanTanimoto_GeoMean", "run_2025-06-29_TgMeanTanimoto_HarmMean", "run_2025-06-29_TgMeanTanimoto_ExpMean",
                 "run_2025-06-30_HighTg", "run_2025-06-30_TgHighTg_Mean",
                 "run_2025-06-30_TgHighTg_GeoMean", "run_2025-06-30_TgHighTg_HarmMean", "run_2025-06-30_TgHighTg_ExpMean"]
label = ["Tg_score", "Tanimoto",
         "RandTanimoto", "MeanTanimoto",
         "TgTanimoto_Mean", "TgTanimoto_GeoMean", "TgTanimoto_HarmMean", "TgTanimoto_ExpMean",
         "TgRanTanimoto_Mean", "TgRanTanimoto_GeoMean", "TgRanTanimoto_HarmMean", "TgRanTanimoto_ExpMean",
         "TgMeanTanimoto_Mean", "TgMeanTanimoto_GeoMean", "TgMeanTanimoto_HarmMean", "TgMeanTanimoto_ExpMean",
         "HighTg", "TgHighTg_Mean", "TgHighTg_GeoMean", "TgHighTg_HarmMean", "TgHighTg_ExpMean"]
"""

dir_files = "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/results/"
for i in list_csv_files:
    list_csv_paths.append(dir_files+i+"/results.csv")
doc_save = "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics"
data_sample_2 = data.sort_values(by="Tg", ascending=False).head(2000)
#print(data_sample_2)
#analyze_multiple_Metrics_generations(list_csv_paths, data_sample_2, doc_save, label)





def plot_tsne_and_histogram(df, smiles_col='SMILES', target_col='Tg', generation_col=None):
    # 1. Compute Morgan fingerprints
    print("Computing Morgan fingerprints...")
    df['FP'] = df[smiles_col].apply(smiles_to_morgan_fp)
    X = np.vstack(df['FP'].values)

    # 2. Apply t-SNE
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_embedded = tsne.fit_transform(X)

    # 3. Prepare colors / labels
    if generation_col and generation_col in df.columns:
        groups = df[generation_col].unique()
        colors = plt.cm.get_cmap('tab10', len(groups))
        color_map = {g: colors(i) for i, g in enumerate(groups)}
        point_colors = df[generation_col].map(color_map)
    else:
        point_colors = 'blue'

    # 4. Plot t-SNE
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=point_colors, s=15, alpha=0.7)
    axes[0].set_title('t-SNE Projection of Morgan Fingerprints')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    if generation_col and generation_col in df.columns:
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(g),
                              markerfacecolor=color_map[g], markersize=8)
                   for g in groups]
        axes[0].legend(handles=handles, title=generation_col)

    # 5. Plot histogram of target property
    axes[1].hist(df[target_col].dropna(), bins=30, color='green', alpha=0.7)
    axes[1].set_title(f'Histogram of {target_col} °K')
    axes[1].set_xlabel(target_col)
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()
#plot_tsne_and_histogram(data)

#***********************************************************************************************************************

top_mols_data = []

def mol_to_high_quality_image(mol, size=(300, 300)):
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])  # Cairo backend = high quality PNG
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png_data))
def save_top_tg_molecules_to_excel(top_mols_data, filename='top_tg_molecules.xlsx'):
    # Create a new Excel file with xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet("Top Tg Molecules")

    # Set column widths
    worksheet.set_column('A:A', 18)  # Image
    worksheet.set_column('B:B', 10)  # Tg
    worksheet.set_column('C:C', 45)  # SMILES
    worksheet.set_column('D:D', 25)  # Label

    # Headers
    headers = ['Molecule', 'Tg (K)', 'SMILES', 'Label']
    for col_num, header in enumerate(headers):
        worksheet.write(0, col_num, header)

    row = 1
    for entry in top_mols_data:
        mol = entry['mol']
        tg = entry['Tg']
        smiles = entry['smiles']
        label = entry['label']

        # Draw image and save to a byte stream
        img = Draw.MolToImage(mol, size=(300, 300))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Write image to cell
        worksheet.insert_image(row, 0, 'mol.png', {'image_data': img_byte_arr, 'x_scale': 1, 'y_scale': 1})

        # Write text data
        worksheet.write(row, 1, tg)
        worksheet.write(row, 2, smiles)
        worksheet.write(row, 3, label)

        row += 1

    workbook.close()


def plot_tsne_replay_vs_no_replay(csvs_replay, csvs_no_replay, score_labels=None):
    if score_labels is None:
        score_labels =[
            "Mean(S_Tg, S_s)","GeoMean(S_Tg, S_s)","HarmMean(S_Tg, S_s)","ExpMean(S_Tg, S_s)","Mean(S_Tg, S_r)","GeoMean(S_Tg, S_r)","HarmMean(S_Tg, S_r)","ExpMean(S_Tg, S_r)","Mean(S_Tg, S_m)","GeoMean(S_Tg, S_m)", "HarmMean(S_Tg, S_m)","ExpMean(S_Tg, S_m)","Mean(S_Tg, S_h)","GeoMean(S_Tg, S_h)","HarmMean(S_Tg, S_h)","ExpMean(S_Tg, S_h)"
            #"TgTanimoto_Mean","TgTanimoto_GeoMean","TgTanimoto_HarmMean","TgTanimoto_ExpMean","TgRanTanimoto_Mean","TgRanTanimoto_GeoMean","TgRanTanimoto_HarmMean","TgRanTanimoto_ExpMean","TgMeanTanimoto_Mean","TgMeanTanimoto_GeoMean", "TgMeanTanimoto_HarmMean","TgMeanTanimoto_ExpMean","TgHighTg_Mean","TgHighTg_GeoMean","TgHighTg_HarmMean","TgHighTg_ExpMean"
        ]
        #score_labels = ['TgTanimoto', 'Tg-score', 'Single-T', 'Random-T', 'Mean-T']
        #score_labels = ['High-score', 'Tg-score', 'Single-T', 'Random-T', 'Mean-T']

    assert len(csvs_replay) == len(csvs_no_replay) == len(score_labels), "Each group must contain 5 CSVs."

    all_fps = []
    all_labels = []
    all_colors = []
    all_markers = []
    highlight_indices = []

    """
    'High-score': 'tab:blue',
        'Tg-score': 'tab:orange',
        'Single-T': 'tab:green',
        'Random-T': 'tab:red',
        'Mean-T': 'tab:purple'
    """
    color_map = {

        'Mean(S_Tg, S_s)': 'tab:blue',
        'GeoMean(S_Tg, S_s)': 'tab:blue',
        'HarmMean(S_Tg, S_s)': 'tab:blue',
        "ExpMean(S_Tg, S_s)": 'tab:blue',

        "Mean(S_Tg, S_r)" : 'tab:orange',
        "GeoMean(S_Tg, S_r)": 'tab:orange',
        "HarmMean(S_Tg, S_r)": 'tab:orange',
        "ExpMean(S_Tg, S_r)": 'tab:orange',

        "Mean(S_Tg, S_m)": 'tab:green',
        "GeoMean(S_Tg, S_m)": 'tab:green',
        "HarmMean(S_Tg, S_m)": 'tab:green',
        "ExpMean(S_Tg, S_m)": 'tab:green',

        "Mean(S_Tg, S_h)": 'tab:red',
        "GeoMean(S_Tg, S_h)": 'tab:red',
        "HarmMean(S_Tg, S_h)": 'tab:red',
        "ExpMean(S_Tg, S_h)": 'tab:red'

    }

    index_counter = 0

    # Replay (ronds)
    for csv_file, label in zip(csvs_replay, score_labels):
        print(csv_file)
        df = pd.read_csv(csv_file)
        df = df[(df['duplicate'] == 0) & df['smiles'].notna()]

        if 'Tg' in df.columns:
            top_tg = df.sort_values('Tg', ascending=False).head(1)
            print(f"\nTop Tg values in '{csv_file}' (duplicates == 0):")
            print(top_tg[['smiles', 'Tg']])
            # -------------------------------------------------------------------------------------------------
            if not top_tg.empty:
                mols = []
                legends = []
                for _, row in top_tg.iterrows():
                    mol = Chem.MolFromSmiles(row['smiles'])
                    if mol:
                        mols.append(mol)
                        legends.append(f"{label}, Tg={row['Tg']:.1f}K, ER")
                        top_mols_data.append({
                            'mol': mol,
                            'Tg': row['Tg'],
                            'smiles': row['smiles'],
                            'label': f"{label} (ER)"  # ou "No ER" dans le second bloc
                        })

                """if mols:
                    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 500), legends=legends)
                    img.show()  # Affiche dans une fenêtre interactive (ou `display(img)` si Jupyter)"""

            # -------------------------------------------------------------------------------------------------
        else:
            top_tg = pd.DataFrame()
            print(f"\nWarning: 'Tg' column not found in '{csv_file}'. Skipping Tg ranking.")

        fps = np.array([smiles_to_morgan_fp(smi) for smi in df['smiles']])
        if fps.size == 0:
            print(f"Skipping empty or invalid dataset: {csv_file}")
            continue

        all_fps.append(fps)
        all_labels.extend([label] * len(fps))
        all_colors.extend([color_map[label]] * len(fps))
        all_markers.extend(['o'] * len(fps))

        # Ajout des indices des meilleurs Tg
        highlight_indices.extend(index_counter + df.index.get_indexer(top_tg.index))
        index_counter += len(fps)
        # -------------------------------------------------------------------------------------------------

    # No replay (carrés)
    for csv_file, label in zip(csvs_no_replay, score_labels):
        df = pd.read_csv(csv_file)
        df = df[(df['duplicate'] == 0) & df['smiles'].notna()]

        if 'Tg' in df.columns:
            top_tg = df.sort_values('Tg', ascending=False).head(1)
            print(f"\nTop Tg values in '{csv_file}' (duplicates == 0):")
            print(top_tg[['smiles', 'Tg']])
            #-------------------------------------------------------------------------------------------------
            if not top_tg.empty:
                mols = []
                legends = []
                for _, row in top_tg.iterrows():
                    mol = Chem.MolFromSmiles(row['smiles'])
                    if mol:
                        mols.append(mol)
                        legends.append(f"{label}, Tg={row['Tg']:.1f}K, No ER")
                        top_mols_data.append({
                            'mol': mol,
                            'Tg': row['Tg'],
                            'smiles': row['smiles'],
                            'label': f"{label} (NO ER)"  # ou "No ER" dans le second bloc
                        })

                """f mols:
                    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(500, 500), legends=legends)
                    img.show()  # Affiche dans une fenêtre interactive (ou `display(img)` si Jupyter)"""
            # -------------------------------------------------------------------------------------------------

        else:
            top_tg = pd.DataFrame()
            print(f"\nWarning: 'Tg' column not found in '{csv_file}'. Skipping Tg ranking.")

        fps = np.array([smiles_to_morgan_fp(smi) for smi in df['smiles']])
        if fps.size == 0:
            print(f"Skipping empty or invalid dataset: {csv_file}")
            continue

        all_fps.append(fps)
        all_labels.extend([label] * len(fps))
        all_colors.extend([color_map[label]] * len(fps))
        all_markers.extend(['s'] * len(fps))

        # Ajout des indices des meilleurs Tg
        highlight_indices.extend(index_counter + df.index.get_indexer(top_tg.index))
        index_counter += len(fps)

    # t-SNE
    X = np.vstack(all_fps)
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    X_embedded = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(X_embedded):
        plt.scatter(x, y, c=all_colors[i], marker=all_markers[i], s=85, edgecolors='k', linewidths=0.3, alpha=0.8)

    # Entourer les top Tg
    for i in highlight_indices:
        x, y = X_embedded[i]
        plt.scatter(x, y, s=200, facecolors='none', edgecolors='black', linewidths=1.2, label='_nolegend_')

    # Légende
    legend_elements = [
        Patch(facecolor=color_map[l], label=l) for l in score_labels
    ] + [
        Line2D([0], [0], marker='o', color='w', label='With Replay',
               markerfacecolor='gray', markeredgecolor='k', markersize=8),
        Line2D([0], [0], marker='s', color='w', label='No Replay',
               markerfacecolor='gray', markeredgecolor='k', markersize=8),
        Line2D([0], [0], marker='o', color='black', label='Highest Tg',
               markerfacecolor='none', markersize=10, linewidth=1)
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=9)
    plt.title("t-SNE of Generated Polymers: Experience Replay vs No Replay")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    save_top_tg_molecules_to_excel(top_mols_data, filename='top_tg_molecules.xlsx')


csvs_replay = [
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgTanimoto_Mean_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgTanimoto_GeoMean_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgTanimoto_HarmMean_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgTanimoto_ExpMean_data_e.csv",

    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgRanTanimoto_Mean_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgRanTanimoto_GeoMean_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgRanTanimoto_HarmMean_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgRanTanimoto_ExpMean_data_e.csv",

    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgMeanTanimoto_Mean_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgMeanTanimoto_GeoMean_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgMeanTanimoto_HarmMean_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgMeanTanimoto_ExpMean_data_e.csv",

    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgHighTg_Mean_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgHighTg_GeoMean_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgHighTg_HarmMean_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgHighTg_ExpMean_data_e.csv"
]
csvs_no_replay = [
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgTanimoto_Mean_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgTanimoto_GeoMean_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgTanimoto_HarmMean_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgTanimoto_ExpMean_data.csv",

    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgRanTanimoto_Mean_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgRanTanimoto_GeoMean_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgRanTanimoto_HarmMean_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgRanTanimoto_ExpMean_data.csv",

    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgMeanTanimoto_Mean_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgMeanTanimoto_GeoMean_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgMeanTanimoto_HarmMean_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgMeanTanimoto_ExpMean_data.csv",

    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgHighTg_Mean_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgHighTg_GeoMean_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgHighTg_HarmMean_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/TgHighTg_ExpMean_data.csv"
]

#plot_tsne_replay_vs_no_replay(csvs_replay, csvs_no_replay)



"""
# Première analyse sans combinaison.
csvs_replay = [
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/HighTg_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/Tg_score_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/Tanimoto_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/RandTanimoto_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/MeanTanimoto_data_e.csv"
]
csvs_no_replay = [
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/HighTg_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/Tg_score_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/Tanimoto_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/RandTanimoto_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/MeanTanimoto_data.csv"
]

"""


#------------------------------Comparons les molécules generées avec les scores individuels et collectifs------------------------
CSV_solo_score_e = [
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/HighTg_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/Tg_score_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/Tanimoto_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/RandTanimoto_data_e.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/MeanTanimoto_data_e.csv"
]
CSV_solo_score = [
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/HighTg_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/Tg_score_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/Tanimoto_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/RandTanimoto_data.csv",
    "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/MeanTanimoto_data.csv"
]
csv_list_1 = CSV_solo_score + CSV_solo_score_e
csv_list_2 = csvs_no_replay + csvs_replay



def canonicalize_smiles(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
    return None

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * 5
    try:
        qed_score = QED.qed(mol)
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        return [qed_score, logp, mw, tpsa, rot_bonds]
    except:
        return [np.nan] * 5

def compare_smiles_with_properties(csv_list_1, csv_list_2, output_excel_path='smiles_comparison_enhanced.xlsx'):
    def load_and_process(csv_paths):
        smiles_set = set()
        full_data = []
        total_count = 0
        for path in csv_paths:
            df = pd.read_csv(path)
            total_count += len(df)
            df = df[df['duplicate'] == 0]
            df['original_smiles'] = df['smiles']
            df['canonical_smiles'] = df['smiles'].apply(canonicalize_smiles)
            df = df.dropna(subset=['canonical_smiles'])
            df['source'] = Path(path).stem
            full_data.append(df[['canonical_smiles', 'original_smiles', 'Tg', 'source']])
            smiles_set.update(df['canonical_smiles'].tolist())
        return set(smiles_set), total_count, pd.concat(full_data, ignore_index=True)

    set1, total1, df1 = load_and_process(csv_list_1)
    set2, total2, df2 = load_and_process(csv_list_2)

    common_smiles = set1 & set2
    unique_to_1 = set1 - set2
    unique_to_2 = set2 - set1

    df1['group'] = df1['canonical_smiles'].apply(lambda s: 'common' if s in common_smiles else 'group1_only')
    df2['group'] = df2['canonical_smiles'].apply(lambda s: 'common' if s in common_smiles else 'group2_only')

    final_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=['canonical_smiles'])

    # Compute molecular descriptors
    descriptors = final_df['canonical_smiles'].apply(compute_descriptors)
    final_df[['QED', 'LogP', 'MW', 'TPSA', 'RotBonds']] = pd.DataFrame(descriptors.tolist(), index=final_df.index)

    # Save Excel
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        final_df.to_excel(writer, sheet_name='Molecules', index=False)
        summary = {
            'Metric': [
                'Total molecules (group 1)',
                'Total molecules (group 2)',
                'Unique canonical (group 1)',
                'Unique canonical (group 2)',
                'Common canonical SMILES',
                'Unique to group 1',
                'Unique to group 2',
                'Total unique SMILES'
            ],
            'Value': [
                total1, total2, len(set1), len(set2),
                len(common_smiles), len(unique_to_1), len(unique_to_2),
                len(set1.union(set2))
            ]
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)

    # Save venn diagram
    venn_path = output_excel_path.replace('.xlsx', '_venn.png')
    plt.figure(figsize=(5, 5))
    venn2([set1, set2], set_labels=('Group 1', 'Group 2'))
    plt.title('Canonical SMILES Overlap')
    plt.tight_layout()
    plt.savefig(venn_path)
    plt.close()

    # Compute average properties for radar chart
    radar_df = final_df.groupby('group')[['QED', 'LogP', 'MW', 'TPSA', 'RotBonds']].mean().dropna()
    fig = go.Figure()
    for group in radar_df.index:
        fig.add_trace(go.Scatterpolar(
            r=radar_df.loc[group].values,
            theta=radar_df.columns.tolist(),
            fill='toself',
            name=group
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title='Average Molecular Properties by Group'
    )
    radar_fig_path = output_excel_path.replace('.xlsx', '_radar.html')
    fig.write_html(radar_fig_path)
    print(f"\n Excel saved to: {output_excel_path}")
    print(f" Venn diagram saved to: {venn_path}")
    print(f" Common molecules: {len(common_smiles)}")
    print(f" Unique to group 1: {len(unique_to_1)}")
    print(f" Unique to group 2: {len(unique_to_2)}")

    return output_excel_path, venn_path, radar_fig_path
#compare_smiles_with_properties(csv_list_1, csv_list_2)


def Image_diagram_kiviat(file_path):
    df = pd.read_excel(file_path, sheet_name="Molecules")
    descriptor_cols = ['QED', 'LogP', 'MW', 'TPSA', 'RotBonds']
    group_means = df.groupby('group')[descriptor_cols].mean()
    norm_means = group_means.copy()

    # Normalisation
    for col in descriptor_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        norm_means[col] = (group_means[col] - col_min) / (col_max - col_min + 1e-9)

    labels = descriptor_cols
    groups = norm_means.index.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, group in enumerate(groups):
        values_norm = norm_means.loc[group].tolist()
        values_norm += values_norm[:1]
        ax.plot(angles, values_norm, color=colors[i], linewidth=2, label=group)
        ax.fill(angles, values_norm, color=colors[i], alpha=0.25)

    # Construire les étiquettes des axes avec les moyennes originales
    axis_labels = []
    for col in descriptor_cols:
        label = col
        values = group_means[col]
        mean_vals = " / ".join([f"{v:.2f}" for v in values])
        axis_labels.append(f"{label} ({mean_vals})")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), axis_labels)

    ax.set_ylim(0, 0.7) #Max
    ax.set_title("Radar plot (normalized)\nRaw mean values in axis labels", y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    image_path = file_path.replace('.xlsx', '_kiviat_labeled_axes_07.png')
    plt.savefig(image_path, dpi=300)
    plt.close()

    print(f"Diagramme avec moyennes dans les axes sauvegardé : {image_path}")
    print("\n=== Statistiques des groupes ===")
    for group in groups:
        count = df[df['group'] == group].shape[0]
        tg_mean = df[df['group'] == group]['Tg'].mean()
        print(f"Groupe {group} : {count} molécules, Tg moyen = {tg_mean:.2f}")

file_path_xlsx = "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/smiles_comparison_enhanced.xlsx"
#Image_diagram_kiviat(file_path_xlsx)