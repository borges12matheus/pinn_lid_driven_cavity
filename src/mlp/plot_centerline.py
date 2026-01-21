import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from mlp.common import MLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_80 = Path("data/prepared/cavity80_full_norm.parquet")
MODEL_PATH = Path("outputs/models/mlp_uv_cavity20.pt")
OUT_FIG = Path("outputs/figures/mlp_centerline_vs_80.pdf")
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

def main():
    df80 = pd.read_parquet(DATA_80)

    # Pega o centerline x/L ~ 0.5 (tolerância baseada no grid)
    x_vals = np.sort(df80["xN"].unique())
    dx = np.min(np.diff(x_vals)) if len(x_vals) > 1 else 1e-3
    x0 = 0.5
    tol = dx / 2 + 1e-12

    line = df80[np.abs(df80["xN"] - x0) <= tol].copy()
    line = line.sort_values("yN")

    X = torch.tensor(line[["xN", "yN"]].to_numpy(), dtype=torch.float32, device=DEVICE)
    y = line["yN"].to_numpy()
    u_ref = line["uN"].to_numpy()

    model = MLP(in_dim=2, out_dim=2, hidden=128, depth=4, act="tanh").to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        pred = model(X).cpu().numpy()
    u_pred = pred[:, 0]

    plt.figure(figsize=(6.2, 4.8))
    plt.plot(u_ref, y, linewidth=2.2, label="CFD 80×80 (ref)")
    plt.plot(u_pred, y, linewidth=2.2, linestyle="--", label="MLP (treinado 20×20)")
    plt.xlabel(r"$u/U_{\mathrm{lid}}$")
    plt.ylabel(r"$y/L$")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_FIG, bbox_inches="tight")
    plt.show()

    print("Figura salva em:", OUT_FIG)

if __name__ == "__main__":
    main()
