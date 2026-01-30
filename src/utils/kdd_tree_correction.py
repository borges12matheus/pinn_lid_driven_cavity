import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

df_c = pd.read_parquet("data/prepared/cavity20_full_norm.parquet")
df_f = pd.read_parquet("data/prepared/cavity80_full_norm.parquet")

Pc = df_c[["xN","yN"]].to_numpy()
Pf = df_f[["xN","yN"]].to_numpy()

tree = cKDTree(Pc)
dist, idx = tree.query(Pf, k=4)  # 4 vizinhos

# pesos inverso da distância (IDW)
w = 1.0 / (dist + 1e-12)
w = w / w.sum(axis=1, keepdims=True)

def interp(col):
    vals = df_c[col].to_numpy()[idx]  # (Nfine, k)
    return (w * vals).sum(axis=1)

u_c2f = interp("uN")
v_c2f = interp("vN")
p_c2f = interp("p")

out = pd.DataFrame({
    "xN": df_f["xN"].to_numpy(),
    "yN": df_f["yN"].to_numpy(),
    "u_c": u_c2f, "v_c": v_c2f, "p_c": p_c2f,
    "du": df_f["uN"].to_numpy() - u_c2f,
    "dv": df_f["vN"].to_numpy() - v_c2f,
    "dp": df_f["p"].to_numpy()  - p_c2f,
})

out.to_parquet("data/prepared/cavity20_to_80_delta.parquet", index=False)
print("OK:", out.shape)
