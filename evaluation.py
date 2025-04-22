import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

print("-----------------------------Start--------------------------")

def classify(row, method):
    """Classifies a prediction as TP, FP, TN, or FN."""
    if row[method] == 1 and row['LIGYSIS'] == 1:
        return 'TP'  # True Positive
    elif row[method] == 0 and row['LIGYSIS'] == 0:
        return 'TN'  # True Negative
    elif row[method] == 1 and row['LIGYSIS'] == 0:
        return 'FP'  # False Positive
    elif row[method] == 0 and row['LIGYSIS'] == 1:
        return 'FN'  # False Negative
    return None  # Handle unexpected cases


# **Load the predictions**
df = pd.read_parquet("ligysis_predictions_90.parquet")  
print(df.head(10))

# **Load ion-binding information**
ion_info_path = "/home/podlesny/esm2-generator/ligand_binding_info.csv"
ion_df = pd.read_csv(ion_info_path)

# **Ensure 'resnum' is correctly extracted as an integer**
def clean_resnum(res):
    return int(''.join(filter(str.isdigit, str(res)))) if pd.notna(res) else np.nan

ion_df['residue'] = ion_df['residue'].apply(clean_resnum)
df['position'] = df['resnum'].apply(clean_resnum)
df['rep_chain'] = df['pdb_id'] + "_" + df['chain']  
true_binding_sites = df[df['LIGYSIS'] == 1]

df["resnum"] = df["resnum"].astype(str).str.extract('(\d+)').astype(float).astype('Int64')
ion_df["residue"] = ion_df["residue"].astype(str).str.extract('(\d+)').astype(float).astype('Int64')

# **Extract relevant columns**
y_true = df["LIGYSIS"].values  
y_pred_probabilities = df["predicted_binding"].values  

# **Calculate Closest True Binding Distance**
def calculate_closest_distance(row):
    if row['type'] == 'FP':  
        relevant_sites = true_binding_sites[
            (true_binding_sites['rep_chain'] == row['rep_chain'])
        ]
        if not relevant_sites.empty:
            return np.abs(relevant_sites['position'] - row['position']).min()
    return np.nan  

df["type"] = df.apply(lambda row: classify(row, "predicted_binding"), axis=1)

print("-----------------------------Calculating distance--------------------------")
df['closest_true_binding_distance'] = df.apply(calculate_closest_distance, axis=1)
print("-----------------------------Calculating weight--------------------------")

# **Weight function**
def weight_function_1(x, x_t=5.0, k=1.0):
    if pd.isna(x):
        return 0
    x_ratio = x / x_t
    weight = np.exp(- (x_ratio) ** k) * (1 - x_ratio)
    return np.clip(weight, 0, 1)

df["weight_f1"] = df["closest_true_binding_distance"].apply(weight_function_1)

# **Merge Ion Information**
if not {'rep_chain', 'residue', 'ion_prop', 'is_ion'}.issubset(ion_df.columns):
    raise ValueError("❌ Ion binding file is missing required columns!")

print("-----------------------------Merging ion info--------------------------")
ion_df['residue'] = ion_df['residue'].astype(int)

df = df.merge(ion_df[['rep_chain', 'residue', 'is_ion']], 
              left_on=['rep_chain', 'resnum'], 
              right_on=['rep_chain', 'residue'], 
              how='left')

df["is_ion"] = df["is_ion"].fillna(0).astype(bool)

# ⬆ Keep everything before this unchanged ⬆

# Filter out ion-binding residues before evaluation
df_no_ions = df[df["is_ion"] == False].copy()
true_binding_sites_no_ions = df_no_ions[df_no_ions["LIGYSIS"] == 1]

# Reclassify types for filtered dataset
df_no_ions["type"] = df_no_ions.apply(lambda row: classify(row, "predicted_binding"), axis=1)

# Recalculate distances for FP
def calculate_closest_distance_filtered(row):
    if row['type'] == 'FP':
        relevant_sites = true_binding_sites_no_ions[
            (true_binding_sites_no_ions['rep_chain'] == row['rep_chain'])
        ]
        if not relevant_sites.empty:
            return np.abs(relevant_sites['position'] - row['position']).min()
    return np.nan

def plot_histogram(data, title, filename, color):
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 100, 50)
    plt.hist(data, bins=bins, weights=np.ones(len(data)) / len(data) * 100,
             alpha=0.6, edgecolor='black', color=color)
    plt.xlabel("Sequence Distance to Nearest True Binding Site")
    plt.ylabel("Percentage (%)")
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
 
print("-----------------------------Recalculating distances without ions--------------------------")
df_no_ions['closest_true_binding_distance'] = df_no_ions.apply(calculate_closest_distance_filtered, axis=1)
df_no_ions["weight_f1"] = df_no_ions["closest_true_binding_distance"].apply(weight_function_1)

# Evaluate only non-ion data
y_true = df_no_ions["LIGYSIS"].values
y_pred_probabilities = df_no_ions["predicted_binding"].values

false_positives = df_no_ions[df_no_ions["type"] == "FP"]
true_positives = df_no_ions[df_no_ions["type"] == "TP"]

# Metrics
TP = len(true_positives)
FP = len(false_positives)
FN = len(df_no_ions[df_no_ions["type"] == "FN"])
TN = len(df_no_ions[df_no_ions["type"] == "TN"])

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
weighted_fp = df_no_ions["weight_f1"].sum()
adjusted_precision = (TP + weighted_fp) / (TP + FP) if FP > 0 else 1.0
adjusted_f1_score = 2 * (adjusted_precision * recall) / (adjusted_precision + recall) if (adjusted_precision + recall) > 0 else 0.0

roc_auc = roc_auc_score(y_true, y_pred_probabilities)
fpr, tpr, _ = roc_curve(y_true, y_pred_probabilities)

# Save ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (No Ion Residues)')
plt.legend()
plt.grid()
plt.savefig("roc_curve_no_ions.png")

# Save Metrics
metrics_df = pd.DataFrame([{
    "ROC AUC": roc_auc,
    "TP": TP,
    "FP": FP,
    "Weighted FP": weighted_fp,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1_score_val,
    "Adjusted Precision": adjusted_precision,
    "Adjusted F1 Score": adjusted_f1_score
}])
metrics_df.to_csv("classification_metrics_no_ions_90.csv", index=False)

# Plot histogram
#plot_histogram(false_positives['closest_true_binding_distance'].dropna(),
#               "False Positive Distance Histogram (No Ion Residues)",
#               "fp_distance_no_ions_percentage.png",
#               "green")

print("✅ Recalculated metrics without ion-binding residues.")
