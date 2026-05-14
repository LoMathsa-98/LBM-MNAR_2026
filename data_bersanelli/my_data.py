import pandas as pd
import numpy as np

def load_my_data(path):
    df = pd.read_csv(path)
    
    # estrai solo le colonne "result" (una per gene)
    result_cols = [col for col in df.columns if col.strip().endswith('result')]
    print(f"Colonne result trovate: {len(result_cols)}")
    print(result_cols)
    
    X = df[result_cols].copy()
    
    # i NaN restano NaN (il modello di Frisch li gestisce così)
    # verifica che i valori non-missing siano solo 0 e 1
    valori_unici = set(X.stack().unique())  # ignora NaN
    assert valori_unici.issubset({0, 1, 0.0, 1.0}), \
        f"Valori inattesi nelle colonne result: {valori_unici}"
    
    X = X.to_numpy(dtype=float)  # NaN restano NaN, 0/1 diventano float
    
    print(f"Shape matrice: {X.shape}")
    print(f"Frazione missing: {np.isnan(X).mean():.2%}")
    print(f"Frazione 1 (mutati, su osservati): {np.nanmean(X):.2%}")
    
    return X, result_cols  # restituisce anche i nomi dei geni


if __name__ == "__main__":
    X, geni = load_my_data("data/my_matrix.csv")