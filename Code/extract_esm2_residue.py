import os
import sys
import argparse
import torch
import esm
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_esm_embeddings_per_residue(sequences, batch_size=16, repr_layer=33):
    embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[i:i + batch_size]
        data = [("protein{}".format(j), seq) for j, seq in enumerate(batch_seqs)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)

        token_reps = results["representations"][repr_layer]

        for rep, seq in zip(token_reps, batch_seqs):
            emb = rep[1:len(seq)+1]
            embeddings.append(emb.cpu().numpy())

        del batch_tokens, results, token_reps
        torch.cuda.empty_cache()
        gc.collect()

    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--seq_col", type=str, default="Sequence")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    input_dir = args.input_dir.replace("\\", "/")
    if not os.path.isdir(input_dir):
        sys.exit(1)

    output_dir = input_dir + "/output_residue"
    os.makedirs(output_dir, exist_ok=True)

    train_path = input_dir + "/Train_combined.csv"
    test_path = input_dir + "/Test_combined.csv"

    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        seqs = df[args.seq_col].astype(str).tolist()
        y = df["Target"].values
        X = extract_esm_embeddings_per_residue(seqs, batch_size=args.batch_size)
        np.save(output_dir + "/X_train_residue.npy", np.array(X, dtype=object))
        np.save(output_dir + "/y_train.npy", y)

    if os.path.exists(test_path):
        df = pd.read_csv(test_path)
        seqs = df[args.seq_col].astype(str).tolist()
        y = df["Target"].values
        X = extract_esm_embeddings_per_residue(seqs, batch_size=args.batch_size)
        np.save(output_dir + "/X_test_residue.npy", np.array(X, dtype=object))
        np.save(output_dir + "/y_test.npy", y)

if __name__ == "__main__":
    main()
