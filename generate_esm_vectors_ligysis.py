import os
import torch
import esm
import numpy as np
import pandas as pd
import gc

# This script generates per-residue embeddings using the ESM-2 (33-layer, 650M) model.
    # Input: CSV file with protein sequences (pdb_id, chain, sequence)
    # Output: .npy files with embeddings saved to a specified directory
    # In case prediction is already done - skips protein sequence


# Define paths
input_csv = "/Users/yanapodlesna/main/Near-Hit Scoring Documentation/Near-Hit-Scoring/data/input/LIGYSIS.csv"
output_dir = "/Users/yanapodlesna/main/Near-Hit Scoring Documentation/Near-Hit-Scoring/data/esm_vectors_ligysis"
chunk_size = 200  # Adjust based on memory availability
repr_layer = 33

# Load ESM model
print("Loading ESM model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model loaded!")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get existing .npy files
existing_files = {f.split(".npy")[0] for f in os.listdir(output_dir) if f.endswith(".npy")}

# Read CSV and identify missing entries
df = pd.read_csv(input_csv, sep=';')
df["file_name"] = df["pdb_id"] + "_" + df["chain"]

missing_df = df[~df["file_name"].isin(existing_files)]
print(f"🔍 Found {len(missing_df)} missing .npy files out of {len(df)} total.")

if len(missing_df) == 0:
    print("✅ All .npy files are already generated. Exiting.")
    exit()

# Function to process a chunk
def process_chunk(chunk):
    for index, row in chunk.iterrows():
        pdb_id, chain, sequence = row["pdb_id"], row["chain"], row["sequence"]
        file_name = f"{pdb_id}_{chain}"
        output_file = os.path.join(output_dir, f"{file_name}.npy")

        if os.path.exists(output_file):
            continue  # Skip if already exists

        print(f"Processing {file_name} ...")
        threshold = 1022
        vectors = []

        while len(sequence) > 0:
            sequence_chunk = sequence[:threshold]
            sequence = sequence[threshold:]
            data = [(file_name, sequence_chunk)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            # Extract per-residue representations
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            token_representations = results["representations"][repr_layer]

            vectors_chunk = token_representations.detach().cpu().numpy()[0][1:-1]
            vectors = np.concatenate((vectors, vectors_chunk)) if len(vectors) > 0 else vectors_chunk

            # Free memory
            del results, token_representations, batch_tokens
            torch.cuda.empty_cache()
            gc.collect()

        # Save vectors to a .npy file
        np.save(output_file, vectors)

        # Free memory for the current sequence
        del vectors
        torch.cuda.empty_cache()
        gc.collect()

# Process missing sequences in chunks
for chunk_idx, chunk in enumerate(np.array_split(missing_df, len(missing_df) // chunk_size + 1)):
    print(f"Processing chunk {chunk_idx + 1}/{len(missing_df) // chunk_size + 1} ...")
    process_chunk(chunk)

print("All missing .npy files generated!")
