# Near-Hit Scoring

This repository contains code and data for training and evaluating binary classifiers that predict ligand-binding residues using ESM-derived protein embeddings.

## 📥 Input Files

All required input files (e.g., training embeddings, LIGYSIS data) can be downloaded from this shared Google Drive folder:

🔗 [Input Files on Google Drive](https://drive.google.com/drive/folders/15rIGV7OB60f1sirQV26__LJUbQIM1aW7?hl=cs)

Please download and place them in the appropriate `data/input` directory before running the scripts.


## 📂 Project Structure

- `generate_esm_vectors_ligysis.py` – Extracts ESM-2 residue embeddings from protein sequences (LIGYSIS set).
- `train_model.py` – Trains an MLP classifier on scPDB-based datasets and evaluates on LIGYSIS.
- `models/` – Saved PyTorch model checkpoints.
- `data/` – Contains `.parquet` embeddings and prediction outputs.
- `LIGYSIS_predictions.pkl` – Preprocessed LIGYSIS dataset with residue-level annotations.

## Usage

### Generate ESM Vectors
```bash
python generate_esm_vectors_ligysis.py
