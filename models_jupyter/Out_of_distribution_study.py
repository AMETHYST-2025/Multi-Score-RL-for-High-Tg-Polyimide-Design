import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from itertools import combinations
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import shap
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import display
#from reinvent_benchmarking.scoring_functions import Tg_score_pred
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from sklearn.model_selection import train_test_split
import csv
from collections import defaultdict
from reinvent_benchmarking.high_Tg_Score import compute_normalized_high_Tg_score_TFIDF
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer
import os


dir_out= "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/models_jupyter/Out_of_distribution"
filename = 'Polyimides_synthetic.csv'  #'hce.csv'
sep = ','
header = 'infer'
smile_name = 'smiles'

fname = "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/datasets/Polyimides_synthetic.csv"
data = pd.read_csv(fname, sep=sep, header=header)
headers =  ["smiles", "Tg"]
data.columns = headers
data = data[headers][:50000]
print(data)
smiles = data[smile_name]
#-----------------------------------------------------------
Polybert_reload = SentenceTransformer('kuelumbus/polyBERT')

def run_PolyBERT(smiles):
    embedding = Polybert_reload.encode(smiles)
    return embedding

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [
            Descriptors.MolWt(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.TPSA(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol)
        ]
    else:
        return [0]*7

def morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr



#features = data['smiles'].apply(featurize)
#X = pd.DataFrame(features.tolist(), columns=['MolWt', 'AromaticRings', 'AliphaticRings', 'TPSA', 'LogP', 'HAcceptors', 'HDonors' ])
#X = np.array(data['smiles'].apply(morgan_fp).tolist())
#data['scores']=data['smiles'].apply(Tg_score_pred)
#y = data['scores']
data['PolyBERT'] = data['smiles'].apply(run_PolyBERT)
y = data['Tg']

def create_intervals(data, y_column='Tg'):
    intervals = {
        'A': data[(data[y_column] >= 400) & (data[y_column] < 500)],
        'B': data[(data[y_column] >= 500) & (data[y_column] < 600)],
        'C': data[(data[y_column] >= 600) & (data[y_column] < 700)],
        'D': data[(data[y_column] >= 700) & (data[y_column] < 800)],
    }
    return intervals

def train_and_evaluate_model_intervals_cumulative(intervals, save_plot_dir = None):
    """
    Entraîne sur les intervalles cumulés et teste sur le suivant.
    Exemple :
    - Train: A → Test: B
    - Train: A+B → Test: C
    - Train: A+B+C → Test: D
    """
    results = {}
    keys = list(intervals.keys())

    cumulative_train = None  # servira à empiler les intervalles

    for i in range(len(keys) - 1):
        train_key = keys[i]
        test_key = keys[i + 1]

        # Mise à jour du jeu d'entraînement cumulatif
        if cumulative_train is None:
            cumulative_train = intervals[train_key]
        else:
            cumulative_train = pd.concat([cumulative_train, intervals[train_key]])

        train_data = cumulative_train
        test_data = intervals[test_key]

        print(f"\n[INFO] Training on intervals up to {train_key}, testing on {test_key}")

        X_train = np.array(train_data['smiles'].apply(morgan_fp).tolist())
        y_train = train_data['Tg']

        X_test = np.array(test_data['smiles'].apply(morgan_fp).tolist())
        y_test = test_data['Tg']

        # 2. Entraînement du modèle
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 3. Prédictions
        y_pred = model.predict(X_test)

        # 4. Évaluation
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"[Train ≤ {train_key} → Test {test_key}] RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")

        # 5. Affichage du plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y_train, model.predict(X_train), color='royalblue', alpha=0.5, label='Train set')
        plt.scatter(y_test, y_pred, color='crimson', alpha=0.8, label='Test set')
        plt.plot([300, 800], [300, 800], 'k--', lw=1.5, label='Perfect prediction')
        plt.xlim(300, 800)
        plt.ylim(300, 800)
        plt.xlabel('Tg True Values')
        plt.ylabel('Tg Predicted Values')
        plt.title(f'Train ≤ {train_key} → Test {test_key}\nRMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_plot_dir is not None:
            fname = f"polybert_RandF_train_le_{train_key}_test_{test_key}.png"
            fpath = os.path.join(save_plot_dir, fname)
            plt.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f"[INFO] Figure saved to {fpath}")

        results[f"≤{train_key}->{test_key}"] = {'rmse': rmse, 'mae': mae, 'r2': r2}

    return results


intervals = create_intervals(data)
#results_rf = train_and_evaluate_model_intervals_cumulative(intervals, save_plot_dir=dir_out)
#print(results_rf)


class FNNRegressor(nn.Module):
    def __init__(self, input_dim):
        super(FNNRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)


def train_and_evaluate_polybert_fnn_cumulative(intervals, input_dim, n_epochs=1000, lr=1e-3, batch_plot_size=None, save_plot_dir=None):
    """
    Même logique cumulative que la version Random Forest,
    mais avec un réseau de neurones (FNN) pour embeddings PolyBERT.
    - intervals : dict like {'A': df_A, 'B': df_B, ...}
    - input_dim : dimension des embeddings
    - n_epochs, lr : hyperparamètres d'entraînement
    - save_plot_dir : si non None, répertoire où sauvegarder les figures (PNG)
    """
    best_loss = float('inf')
    patience = 10  # nombre d'epochs sans amélioration avant arrêt
    counter = 0
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keys = list(intervals.keys())
    cumulative_train = None

    if save_plot_dir is not None:
        os.makedirs(save_plot_dir, exist_ok=True)

    for i in range(len(keys) - 1):
        train_key = keys[i]
        test_key = keys[i + 1]

        # Accumulation progressive des intervalles d'entraînement
        if cumulative_train is None:
            cumulative_train = intervals[train_key].copy()
        else:
            cumulative_train = pd.concat([cumulative_train, intervals[train_key]], ignore_index=True)

        train_data = cumulative_train
        test_data = intervals[test_key]

        print(f"\n[INFO] Training on intervals up to {train_key}, testing on {test_key}")

        from sklearn.model_selection import train_test_split



        # On suppose que la colonne contenant les embeddings s'appelle 'PolyBERT' et contient des vecteurs (list/np.array)
        X_train = np.array(train_data['PolyBERT'].tolist())
        y_train = np.array(train_data['Tg']).reshape(-1, 1)

        # Split train_data en train/validation (ex : 90%/10%)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split( X_train, y_train, test_size=0.1, random_state=42)
        X_train_tensor = torch.FloatTensor(X_train_split).to(device)
        y_train_tensor = torch.FloatTensor(y_train_split).to(device)
        X_val_tensor = torch.FloatTensor(X_val_split).to(device)
        y_val_tensor = torch.FloatTensor(y_val_split).to(device)

        # Conversion en tenseurs
        X_test = np.array(test_data['PolyBERT'].tolist())
        y_test = np.array(test_data['Tg']).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)

        # Modèle FNN
        model = FNNRegressor(input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Entraînement (entraînement "full batch" pour rester proche de ta version)
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # évaluation sur validation set (ici on peut prendre test comme proxy)
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)

            # early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), "best_model.pt")  # sauvegarde du meilleur modèle
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    model.load_state_dict(torch.load("best_model.pt"))
                    break

        # Évaluation
        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.detach().cpu().numpy().reshape(-1)
        y_test_flat = y_test_tensor.detach().cpu().numpy().reshape(-1)

        rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred))
        mae = mean_absolute_error(y_test_flat, y_pred)
        r2 = r2_score(y_test_flat, y_pred)

        print(f"[Train ≤ {train_key} → Test {test_key}] RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")

        results[f"≤{train_key}->{test_key}"] = {'rmse': rmse, 'mae': mae, 'r2': r2}

        # -----------------------
        # PLOT (même style que pour la RF)
        # -----------------------
        plt.figure(figsize=(7, 7))

        # Prédictions sur l'ensemble d'entraînement
        model.eval()
        with torch.no_grad():
            y_train_pred = model(X_train_tensor).detach().cpu().numpy().reshape(-1)

        # Scatter pour les données d'entraînement
        plt.scatter(y_train_tensor.detach().cpu().numpy().reshape(-1), y_train_pred, color='royalblue', alpha=0.5, label='Train set')
        # Scatter pour les données de test
        plt.scatter(y_test_tensor.detach().cpu().numpy().reshape(-1), y_pred, color='crimson', alpha=0.8, label='Test set')
        # Ligne de prédiction parfaite
        plt.plot([300, 800], [300, 800], 'k--', lw=1.5, label='Perfect prediction')

        # Mise en forme
        plt.xlim(300, 800)
        plt.ylim(300, 800)
        plt.xlabel('True $T_g$ (K)')
        plt.ylabel('Predicted $T_g$ (K)')
        plt.title(
            f'PolyBERT-FNN Evaluation\nTrain ≤ {train_key} → Test {test_key}\nRMSE = {rmse:.2f}, R² = {r2:.2f}, MAE = {mae:.2f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_plot_dir is not None:
            fname = f"polybert_fnn_train_le_{train_key}_test_{test_key}.png"
            fpath = os.path.join(save_plot_dir, fname)
            plt.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f"[INFO] Figure saved to {fpath}")

    return results

results_polybert = train_and_evaluate_polybert_fnn_cumulative(intervals, input_dim=600, save_plot_dir=dir_out)

