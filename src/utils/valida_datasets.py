import pandas as pd
import numpy as np

FULL = "data/prepared/cavity80_full_norm.parquet"
BC   = "data/prepared/cavity80_bc.parquet"
INT  = "data/prepared/cavity80_interior.parquet"

df_full = pd.read_parquet(FULL)
df_bc   = pd.read_parquet(BC)
df_int  = pd.read_parquet(INT)

print("Shapes:")
print(" full:", df_full.shape)
print(" bc  :", df_bc.shape)
print(" int :", df_int.shape)

print("\nColunas:", df_full.columns.tolist())

# Checagens básicas de faixa (normalizado)
for c in ["xN","yN"]:
    print(f"{c}: min={df_full[c].min():.4f} max={df_full[c].max():.4f}")

# Separar tampa e paredes
lid = df_bc[df_bc["region"] == "moving_lid"]
walls = df_bc[df_bc["region"] == "walls"]

print("\nBC counts:")
print(" lid  :", len(lid))
print(" walls:", len(walls))

# Sanidade da tampa: uN ~ 1 e vN ~ 0
if len(lid) > 0:
    print("\nLid check (means):")
    print(" uN mean:", lid["uN"].mean(), " | uN std:", lid["uN"].std())
    print(" vN mean:", lid["vN"].mean(), " | vN std:", lid["vN"].std())

# Montar tensores (inputs e targets)
X_bc = df_bc[["xN","yN"]].to_numpy(dtype=np.float32)
Y_bc = df_bc[["uN","vN"]].to_numpy(dtype=np.float32)

X_int = df_int[["xN","yN"]].to_numpy(dtype=np.float32)

print("\nTensores prontos:")
print(" X_bc :", X_bc.shape, " Y_bc:", Y_bc.shape)
print(" X_int:", X_int.shape)

# Salvar npz (opcional, bem prático pra treino)
np.savez("data/prepared/cavity20_pinn_pack.npz", X_bc=X_bc, Y_bc=Y_bc, X_int=X_int)
print("\nSalvo: data/prepared/cavity20_pinn_pack.npz")