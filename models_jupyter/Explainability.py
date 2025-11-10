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



features = data['smiles'].apply(featurize)
#X = pd.DataFrame(features.tolist(), columns=['MolWt', 'AromaticRings', 'AliphaticRings', 'TPSA', 'LogP', 'HAcceptors', 'HDonors' ])
X = np.array(data['smiles'].apply(morgan_fp).tolist())
#data['scores']=data['smiles'].apply(Tg_score_pred)
#y = data['scores']
y = data['Tg']


def train_and_evaluate_model(X, y, model_path='/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/models_jupyter/reinvent_benchmarking/Pred_Tg/Random_forest/random_forest_model.pkl'):
    # 1. Split des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Entraînement du modèle
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)

    # 3. Prédictions
    y_pred = model.predict(X_test)

    # 4. Évaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test,y_pred)
    print(mae)
    r2 = r2_score(y_test, y_pred)

    # 5. Affichage du plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue', label='Predicted vs True')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect prediction')
    plt.xlabel('Tg True Values')
    plt.ylabel('Tg Predicted Values')
    plt.title(f'Model Evaluation\nRMSE = {rmse:.2f}, R² = {r2:.2f}, MAE = {mae:.2f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 6. Sauvegarde du modèle
    #joblib.dump(model, model_path)
    #print(f"[INFO] Modèle sauvegardé sous : {model_path}")

    return model, rmse, r2

#model, rmse, r2 = train_and_evaluate_model(X, y)


#-----------------------------------------------------------------------------------------------------------------------
def find_molecule_with_bit(bit, smiles_list):
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bit_info)
        if bit in bit_info:
            return mol, bit_info
    return None, None

def Explainer_():
    # 1. Calcul des SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # 2. Statistiques sur les SHAP
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)       # Importance
    mean_signed_shap = shap_values.values.mean(axis=0)            # Sens de l'effet

    top_bits_all = np.argsort(mean_abs_shap)[:][::-1]
    top_bits = np.argsort(mean_abs_shap)[-12:][::-1]  # Top 10 bits les plus influents

    # 3. Création du dictionnaire avec les 2 types d'impact
    important_bits_all = {
        int(bit): {
            "mean_abs": float(mean_abs_shap[bit]),
            "mean_signed": float(mean_signed_shap[bit])
        }
        for bit in top_bits_all
    }

    # 4. Sauvegarde dans un fichier CSV
    with open("/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/important_bits.csv", mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["bit", "mean_abs", "mean_signed"])  # En-têtes
        for bit, values in important_bits_all.items():
            writer.writerow([bit, values["mean_abs"], values["mean_signed"]])

    # 5. Sous-structures associées aux bits (si la fonction `find_molecule_with_bit` est définie)
    submols = []
    titles = []

    for bit in top_bits:
        mol, bit_info = find_molecule_with_bit(bit, data['smiles'])  # -> doit exister
        if mol and bit_info and bit in bit_info:
            center, radius = bit_info[bit][0]
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
            if not env:
                continue
            submol = Chem.PathToSubmol(mol, env)
            submols.append(submol)

            direction = "+" if mean_signed_shap[bit] >= 0 else "-"
            titles.append(f"Bit {bit} ({direction}{mean_signed_shap[bit]:.3f})")

    if submols:
        img = Draw.MolsToGridImage(
            submols,
            molsPerRow=5,
            subImgSize=(200, 200),
            legends=titles
        )
        img.save("/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/IMG_sub_structures/important_bits.png", dpi=(300, 300))
        plt.figure(figsize=(15, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print("Aucune sous-structure trouvée.")

    # 6. SHAP summary plot limité aux bits les plus influents
    shap.summary_plot(
        shap_values[:, top_bits],
        features=X[:, top_bits],
        feature_names=[f'bit_{i}' for i in top_bits]
    )

#Explainer_()

#-----------------------------------------------------------------------------------------------------------------------

bit_csv_path = "/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/important_bits.csv"
important_df = pd.read_csv(bit_csv_path)
bit_idf = dict(zip(important_df['bit'], important_df['idf']))
bit_strength = dict(zip(
    important_df['bit'],
    important_df['mean_abs'] * important_df['mean_signed']
))

def custom_normalization(x, inf=-15, sup=5, sensibility=100):
    if inf <= x <= sup:
        return 1.0
    elif x < inf:
        return max(0.0, 1 - abs(x - inf) / sensibility)
    else:  # x > sup
        return max(0.0, 1 - abs(x - sup) / sensibility)

def compute_high_Tg_susceptibility_tfidf_with_tf(bit_csv_path, df_smiles_tg, smile_col="smiles", tg_col="Tg", radius=2, n_bits=2048):
    # 1. Charger les poids
    important_df = pd.read_csv(bit_csv_path)
    important_df["impact"] = important_df["mean_abs"] * important_df["mean_signed"]
    #important_df["impact"] = np.sign(important_df["mean_signed"]) * np.sqrt(np.abs(important_df["mean_abs"] * important_df["mean_signed"]))

    bit_impact = dict(zip(important_df["bit"], important_df["impact"]))

    # 2. Compter DF : combien de molécules contiennent chaque bit
    df_counts = defaultdict(int)
    N = len(df_smiles_tg)

    fingerprints = []
    for smiles in df_smiles_tg[smile_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fingerprints.append(None)
            continue
        fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=n_bits)
        fingerprints.append(fp)
        for b in fp.GetNonzeroElements().keys():
            df_counts[b] += 1

    # 3. Calcul du score TF-IDF pondéré
    scores = []
    for fp in fingerprints:
        if fp is None:
            scores.append(0.0)
            continue
        score = 0.0
        for b, tf in fp.GetNonzeroElements().items():  # tf = nombre d’occurrences du bit
            impact = bit_impact.get(b, 0.0)
            df_b = df_counts.get(b, 0)
            idf = np.log(N / (1 + df_b))
            score += impact * tf * idf
        scores.append(score)

    # 4. Ajouter au DataFrame et trier
    df = df_smiles_tg.copy()
    df["susceptibility_score"] = scores

    # Normalisation min-max du score
    min_score = df["susceptibility_score"].min()
    max_score = df["susceptibility_score"].max()
    print(min_score, max_score)
    #df["norm_score"] = (df["susceptibility_score"] - min_score) / (max_score - min_score)
    born_inf = -15
    born_sup = 5
    df = df.sample(n=100, random_state=42)
    df["norm_score"] = df["susceptibility_score"].apply(lambda x: custom_normalization(x, born_inf, born_sup))
    #df["norm_score"] = df["smiles"].apply(lambda s: compute_normalized_high_Tg_score_TFIDF(s, bit_idf, bit_strength)) # Vérification de la fonction de scoring
    df_sorted = df.sort_values(by=tg_col).reset_index(drop=True)

    # 5. Plot
    plt.figure(figsize=(10, 6))

    """colors = ['red' if tg > 700 else '#1f77b4' for tg in df_sorted[tg_col]]
    plt.scatter(df_sorted[tg_col], df_sorted["susceptibility_score"], c=colors, alpha=0.6)
    # Lignes horizontales
    plt.axhline(y=-15, color='gray', linestyle='--', linewidth=1.2)
    plt.axhline(y=5, color='gray', linestyle='--', linewidth=1.2)"""

    #plt.plot(df_sorted[tg_col], df_sorted["norm_score"], 'o', alpha=0.6) #
    colors = ['red' if tg > 700 else '#1f77b4' for tg in df_sorted[tg_col]]
    plt.scatter(df_sorted[tg_col], df_sorted["norm_score"], c=colors, alpha=0.6)

    plt.xlabel("Tg (K)")
    plt.ylabel("Score susceptibility")
    plt.title("Actual Tg vs score susceptibility to High Tg")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 6. Ajouter colonne IDF au CSV
    #important_df["idf"] = important_df["bit"].apply(lambda b: np.log(N / (1 + df_counts.get(b, 0))))
    #important_df.to_csv(bit_csv_path, index=False)

    return df_sorted


#compute_high_Tg_susceptibility_tfidf_with_tf("/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/important_bits.csv", data)