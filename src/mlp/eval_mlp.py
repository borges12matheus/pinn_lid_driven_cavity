import pandas as pd
import torch
from pathlib import Path

from common import MLP, rmse, mae

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_80 = Path("data/prepared/cavity80_full_norm.parquet")
MODEL_PATH = Path("outputs/models/mlp_uv_cavity20.pt")

# Ajuste se quiser acelerar:
N_EVAL = None  # None = usa tudo; ou um int tipo 20000/30000

def main():
    df = pd.read_parquet(DATA_80)

    if N_EVAL is not None and len(df) > N_EVAL:
        df = df.sample(n=N_EVAL, random_state=42).reset_index(drop=True)

    # --- Tensores para erro (sem grad) ---
    X0 = torch.tensor(df[["xN", "yN"]].to_numpy(), dtype=torch.float32, device=DEVICE)
    Y = torch.tensor(df[["uN", "vN"]].to_numpy(), dtype=torch.float32, device=DEVICE)

    model = MLP(in_dim=2, out_dim=2, hidden=128, depth=4, act="tanh").to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        pred = model(X0)
        rmse_uv = rmse(pred, Y).item()
        mae_uv = mae(pred, Y).item()

    # --- Divergência via autograd (precisa de grad) ---
    X = X0.detach().clone().requires_grad_(True)
    pred_g = model(X)
    u = pred_g[:, 0:1]
    v = pred_g[:, 1:2]

    grads_u = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u),
        retain_graph=True, create_graph=False
    )[0]
    grads_v = torch.autograd.grad(
        v, X, grad_outputs=torch.ones_like(v),
        retain_graph=False, create_graph=False
    )[0]

    du_dx = grads_u[:, 0:1]
    dv_dy = grads_v[:, 1:2]
    div = du_dx + dv_dy

    div_l2 = torch.sqrt(torch.mean(div**2)).item()

    print("=== Avaliação MLP (treinado no 20×20, avaliado no 80×80) ===")
    print(f"N pontos avaliados: {len(df)}")
    print(f"RMSE(u,v): {rmse_uv:.6e}")
    print(f" MAE(u,v): {mae_uv:.6e}")
    print(f"L2(div):   {div_l2:.6e}   (quanto menor, mais incompressível)")

if __name__ == "__main__":
    main()
