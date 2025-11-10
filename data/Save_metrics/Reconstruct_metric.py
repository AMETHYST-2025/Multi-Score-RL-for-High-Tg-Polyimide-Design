import pandas as pd

# Chargement du fichier CSV
#df = pd.read_csv("/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/best_steps_summary.csv")
df = pd.read_csv("/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/best_steps_summary_e.csv")
# Correction des fautes de frappe dans les noms de fichier
df["file"] = df["file"].str.replace("RanTanimoto", "RandTanimoto", regex=False)

# Configuration
scores = ["Tg_score", "Tanimoto", "RandTanimoto", "MeanTanimoto", "HighTg"]
columns = ["X", "Mean", "GeoMean", "HarmMean", "ExpMean"]
metrics_labels = {
    "rate": "Rate",
    "Unicité externe": "ExtUn",
    "Unicité interne": "IntUn",
    "IntDiv": "IntDiv",
    "SNN":"SNN",
    "Fraction valide": "Val",
    "Nbr_Smiles": "N",
    "best_step": "Step"

}

# Stockage des lignes finales
final_rows = []

# Traitement principal
for score in scores:
    row = {"Scores": score, "X": score}

    for col in columns:
        if col == "X":
            target_file = score
        else:
            target_file = f"Tg{score}_{col}" if score != "Tg_score" else score

        matched = df[df["file"] == target_file]

        if not matched.empty:
            entry = matched.iloc[0]
            cell_lines = []
            for orig, label in metrics_labels.items():
                val = entry[orig]
                if isinstance(val, float):
                    if label in ["N", "Step"]:
                        line = f"{label}: {int(round(val))}"
                    else:
                        line = f"{label}: {val:.3f}"
                else:
                    line = f"{label}: {val}"
                cell_lines.append(line)
            cell_text = "\n".join(cell_lines)
        else:
            cell_text = ""

        row[col] = cell_text

    final_rows.append(row)

# Création du DataFrame final
df_final = pd.DataFrame(final_rows)

# ➕ Ajout de la ligne somme des rates
sum_row = {"Scores": "SUM_RATES", "X": ""}
for col in columns[1:]:
    sum_rate = 0
    for row in df_final[col]:
        if row:
            for line in row.split("\n"):
                if "Rate:" in line:
                    try:
                        rate_value = float(line.split(":")[1].strip())
                        sum_rate += rate_value
                    except:
                        continue
    sum_row[col] = f"{sum_rate:.3f}"

df_final = pd.concat([df_final, pd.DataFrame([sum_row])], ignore_index=True)


df_final.to_csv("/Users/tchagoue/Documents/AMETHYST/Code_git/AMETHYST/Generative_AI/Tartarus/data/Save_metrics/tableau_scores_aggreges_e.csv", index=False)
print(" Fichier exporté sous : tableau_scores_aggreges.csv")




