import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------
# Config
# -----------------------
#CSV_PATH = Path("data/cavity_20/cavity_20_full.csv")  # path para CSV consolidado
CSV_PATH = Path("data/cavity_80/cavity_80_full.csv")  # path para CSV referência 80x80
OUT_DIR = Path("data/prepared")
OUT_DIR.mkdir(parents=True, exist_ok=True)

L = 0.1       # tamanho do domínio (0.1m no cavity tutorial)
U_lid = 1.0   # velocidade da tampa
EPS = 1e-10   # tolerância para identificar contornos

# -----------------------
# Helpers
# -----------------------
def require_cols(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV está sem colunas esperadas: {missing}\nColunas encontradas: {list(df.columns)}")

def classify_regions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifica pontos em:
    - moving_lid: y=1 (tampa)
    - walls: y=0 ou x=0 ou x=1 (paredes)
    - interior: resto
    """
    x = df["xN"].to_numpy()
    y = df["yN"].to_numpy()
    y_top = y.max()
    y_bot = y.min()
    x_left = x.min()
    x_right = x.max()
    
    eps_edge = 1e-12

    is_top = np.isclose(y, y_top, atol=eps_edge)
    is_bottom = np.isclose(y, y_bot, atol=eps_edge)
    is_left = np.isclose(x, x_left, atol=eps_edge)
    is_right = np.isclose(x, x_right, atol=eps_edge)

    # opcional: remover cantos da tampa (ajuda a média do u ficar ~1)
    is_corner = (is_top & is_left) | (is_top & is_right)

    region = np.full(len(df), "interior", dtype=object)

    # 1) paredes (sem incluir topo)
    region[is_bottom | is_left | is_right] = "walls"

    # 2) tampa por último, sobrescrevendo (exceto cantos)
    region[is_top & (~is_corner)] = "moving_lid"

    df["region"] = region
    return df

# -----------------------
# Main
# -----------------------
def main():
    df = pd.read_csv(CSV_PATH)

    # Colunas esperadas (formato “full” consolidado)
    require_cols(df, ["x", "y", "u", "v", "p"])

    # Normalização adimensional
    df["xN"] = df["x"] / L
    df["yN"] = df["y"] / L
    df["uN"] = df["u"] / U_lid
    df["vN"] = df["v"] / U_lid

    # Classificar regiões (BCs vs interior)
    df = classify_regions(df)

    # Salvar datasets separados
    df_full = df[["xN", "yN", "uN", "vN", "p", "region"]].copy()
    df_bc = df_full[df_full["region"].isin(["walls", "moving_lid"])].copy()
    df_int = df_full[df_full["region"] == "interior"].copy()

    # (Opcional) reduzir interior para collocation points (se o dataset for grande)
    # df_int = df_int.sample(n=min(20000, len(df_int)), random_state=42)

    out_full = OUT_DIR / "cavity80_full_norm.parquet"
    out_bc   = OUT_DIR / "cavity80_bc.parquet"
    out_int  = OUT_DIR / "cavity80_interior.parquet"

    df_full.to_parquet(out_full, index=False)
    df_bc.to_parquet(out_bc, index=False)
    df_int.to_parquet(out_int, index=False)

    print("OK! Arquivos gerados:")
    print("-", out_full)
    print("-", out_bc, f"(BC: {len(df_bc)})")
    print("-", out_int, f"(Interior: {len(df_int)})")

    # Checagem rápida das BCs (sanidade)
    # Tampa: u=1, v=0 (aprox)
    lid = df_bc[df_bc["region"] == "moving_lid"]
    if len(lid) > 0:
        print("\nChecagem tampa (média): uN=", lid["uN"].mean(), " vN=", lid["vN"].mean())

if __name__ == "__main__":
    main()
