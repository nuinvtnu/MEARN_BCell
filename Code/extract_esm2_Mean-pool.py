import os
import sys
import argparse
import torch
import esm
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc

def load_model():
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, batch_converter, device

def extract_embeddings_batch(model, batch_converter, device, df, batch_size=16):
    X, Y = [], []
    embed_dim = model.embed_dim if hasattr(model, "embed_dim") else 1280
    for i in tqdm(range(0, len(df), batch_size), desc="Processing"):
        batch_df = df.iloc[i:i+batch_size]
        batch_labels = [(str(row['ID']), str(row['Sequence']).upper()) for _, row in batch_df.iterrows()]
        try:
            batch_tokens = batch_converter(batch_labels)[2].to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            for j, (_, seq) in enumerate(batch_labels):
                try:
                    emb = token_representations[j, 1:len(seq)+1].mean(0).cpu().numpy()
                    X.append(emb)
                    Y.append(batch_df.iloc[j].get("Target", None))
                except:
                    X.append(np.zeros(embed_dim))
                    Y.append(batch_df.iloc[j].get("Target", None))
        except:
            for _ in range(len(batch_df)):
                X.append(np.zeros(embed_dim))
                Y.append(None)
        torch.cuda.empty_cache()
        gc.collect()
    return np.array(X), np.array(Y, dtype=object)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings for Train_combined.csv and Test_combined.csv.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    input_dir = args.input_dir.replace("\\", "/")
    if not os.path.isdir(input_dir):
        sys.exit(1)
    output_dir = args.output_dir or f"{input_dir}/output"
    ensure_dir(output_dir)
    train_path = os.path.join(input_dir, "Train_combined.csv")
    test_path = os.path.join(input_dir, "Test_combined.csv")
    model, batch_converter, device = load_model()
    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
        X_train, y_train = extract_embeddings_batch(model, batch_converter, device, df_train, batch_size=args.batch_size)
        np.save(f"{output_dir}/Train_combined_X.npy", X_train)
        np.save(f"{output_dir}/Train_combined_y.npy", y_train)
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        X_test, y_test = extract_embeddings_batch(model, batch_converter, device, df_test, batch_size=args.batch_size)
        np.save(f"{output_dir}/Test_combined_X.npy", X_test)
        np.save(f"{output_dir}/Test_combined_y.npy", y_test)

if __name__ == "__main__":
    main()
