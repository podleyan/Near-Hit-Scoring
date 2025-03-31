# Near-Hit Scoring

This repository contains code and data for training and evaluating binary classifiers that predict ligand-binding residues using ESM-derived protein embeddings.

## 📥 Input Files

All required input files (e.g., training embeddings, LIGYSIS data) can be downloaded from this shared Google Drive folder:

🔗 [Input Files on Google Drive](https://drive.google.com/drive/folders/15rIGV7OB60f1sirQV26__LJUbQIM1aW7?hl=cs)

Please download and place them in the appropriate `data/input` directory before running the scripts.
There you can find:
- `protein_residue_embeddings_SI_10.parquet,  protein_residue_embeddings_SI_30.parquet,  protein_residue_embeddings_SI_90.parquet,  protein_residue_embeddings_all.parquet` - scPDB dataset with ESM emneddings
- `protein_SI_10.csv, protein_SI_30.csv` - List of the representative embeddings for sequence identity 10%, 30%
- `LIGYSIS.csv` - LIGYSIS dataset per protein chain
- `ligysis_df.parquet` - LIGYSIS dataset per residue
- `LIGYSIS_predictions.pkl` - Predictions of all models from https://github.com/bartongroup/LBS-comparison/tree/master on LIGYSIS dataset
- Folder ems_vectors_ligysis - Contain .npy vectors for each protein chain in LIGYSIS dataset


## 📂 Project Structure

- `generate_esm_vectors_ligysis.py` – Extracts ESM-2 residue embeddings from protein sequences (LIGYSIS set).
- `train_model.py` – Trains an MLP classifier on scPDB-based datasets and evaluates on LIGYSIS.
- `models/` – Saved PyTorch model checkpoints.
- `data/` – Contains `.parquet` embeddings and prediction outputs.
- `LIGYSIS_predictions.pkl` – Preprocessed LIGYSIS dataset with residue-level annotations.

## Usage

### Train Model
```bash
python train_model.py --train_file data/protein_residue_embeddings_all.parquet

