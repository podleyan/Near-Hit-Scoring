# This script trains and evaluates a binary classifier to predict ligand-binding residues using ESM-derived protein embeddings.
# 
# Training is performed on the scPDB dataset or its subsets, selected based on sequence similarity thresholds (90%, 30%, or 10%).
# The input .parquet file (specified via --train_file) contains residue-level embeddings and binding labels.
#
# Evaluation is conducted on the LIGYSIS dataset using precomputed ESM embeddings.
# The script outputs per-residue predictions, classification metrics, and an ROC curve.
# Model checkpoints and predictions are automatically named based on the training file suffix.



import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp  # Mixed Precision Training
import time
import gc
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pickle


# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description="Train binding site predictor with specified input file")
parser.add_argument("--train_file", type=str, required=True, help="Path to training .parquet file")
args = parser.parse_args()

# Extract suffix from filename (e.g., 'all' from 'protein_residue_embeddings_all.parquet')
model_suffix = os.path.splitext(os.path.basename(args.train_file))[0].split("_")[-1]
model_filename = f"binding_predictor_best_{model_suffix}.pth"
predictions_filename = f"ligysis_predictions_{model_suffix}.parquet"

class BindingDataset(Dataset):
    def __init__(self, df):
        self.embeddings = np.stack(df["embedding"].values)
        self.labels = df["binding"].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

class BindingPredictor(nn.Module):
    def __init__(self, input_dim=1280):
        super(BindingPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, epochs=200):
    best_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_preds = []
        all_labels = []

        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(embeddings).squeeze()
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_accuracy = accuracy_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds, zero_division=0)
        train_recall = recall_score(all_labels, all_preds, zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)
        train_mcc = matthews_corrcoef(all_labels, all_preds)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Accuracy: {train_accuracy:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f} | MCC: {train_mcc:.4f} | Time: {epoch_time:.2f}s")

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), model_filename)
            print(" Model improved, saving checkpoint.")

        gc.collect()
        torch.cuda.empty_cache()

    print(" Training Complete!")

# ---------------------- Main Flow ----------------------

train_flag = True

if train_flag:
    train_parquet = args.train_file
    df_train = pd.read_parquet(train_parquet)
    print(df_train.head(10))
    df_train["embedding"] = df_train["embedding"].apply(lambda x: np.array(x, dtype=np.float32))

    batch_size = 512
    train_dataset = BindingDataset(df_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Data Loaded: Training Samples = {len(df_train)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BindingPredictor().to(device)

    binding_counts = df_train["binding"].value_counts()
    num_non_binding = binding_counts[0]
    num_binding = binding_counts[1]
    pos_weight = torch.tensor([num_non_binding / num_binding], dtype=torch.float32).to(device)

    print(f"Class Weights: Non-Binding={num_non_binding}, Binding={num_binding}, Pos Weight={pos_weight.item():.4f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    train_model(model, train_loader, epochs=2)
    model.eval()

else:
    print("Loading model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BindingPredictor().to(device)
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
    print("Model is successfully loaded")

# ---------------------- Prediction & Evaluation ----------------------

print(" Started LIGYSIS Predictions...")

ligysis_pkl = "/home/podlesny/esm2-generator/LIGYSIS_ress_binary_labels.pkl"
with open(ligysis_pkl, "rb") as file:
    ligysis_df = pickle.load(file)

esm_embedding_dir = "/home/podlesny/esm2-generator/data/output_ligysis"

ligysis_df["pdb_id"] = ligysis_df["rep_chain"].apply(lambda x: x.split("_")[0])
ligysis_df["file_name"] = ligysis_df["pdb_id"] + "_" + ligysis_df["chain"]

merged_data = []

for pdb_chain in ligysis_df['file_name'].unique():
    npy_file = os.path.join(esm_embedding_dir, f"{pdb_chain}.npy")
    if os.path.exists(npy_file):
        embeddings = np.load(npy_file, mmap_mode='r')
        chain_df = ligysis_df[ligysis_df["file_name"] == pdb_chain].copy()
        if len(chain_df) == embeddings.shape[0]:
            chain_df["embedding"] = list(embeddings)
            merged_data.append(chain_df)
        else:
            print(f"⚠️ Mismatch: {pdb_chain} (Residues: {len(chain_df)}, Embeddings: {embeddings.shape[0]})")

ligysis_df = pd.concat(merged_data, ignore_index=True)
ligysis_df["embedding"] = ligysis_df["embedding"].apply(lambda x: np.array(x, dtype=np.float32))

print(f" LIGYSIS dataset processed with {len(ligysis_df)} residues.")

embeddings_array = np.stack(ligysis_df["embedding"].values)
embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32).to(device)

with torch.no_grad():
    outputs = model(embeddings_tensor).squeeze()
    probabilities = torch.sigmoid(outputs).cpu().numpy()
    predictions = (probabilities > 0.5).astype(int)

ligysis_df["predicted_binding"] = predictions
ligysis_df.to_parquet(predictions_filename)
print(f" Predictions saved to '{predictions_filename}'!")

y_true = ligysis_df["LIGYSIS"].values
y_pred = ligysis_df["predicted_binding"].values
roc_auc = roc_auc_score(y_true, probabilities)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)

print("🔍 Classification Report:")
print(classification_report(y_true, y_pred))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

fpr, tpr, _ = roc_curve(y_true, probabilities)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid()
plt.show()
