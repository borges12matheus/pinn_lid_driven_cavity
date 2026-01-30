import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# Definição do MLP (importado de src/mlp/common.py)
class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, hidden=128, depth=4, act="tanh"):
        super().__init__()
        acts = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        a = acts.get(act.lower(), nn.Tanh())

        layers = [nn.Linear(in_dim, hidden), a]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), a]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((a - b) ** 2))

def mae(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(a - b))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------- Paths ----------
NPZ_PACK = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/data/prepared/cavity20_pinn_pack.npz")               # BC + interior (20)
DELTA_DATA = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/data/prepared/cavity20_to_80_delta.parquet")       # pontos no grid do 80 + coarse projetado
DELTA_MODEL = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/outputs/models/mlp_uv_refine_cavity20.pt")            # seu MLP de refino (delta)
OUT_PINN = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/outputs/models/pinn_uv_hybrid.pt")
OUT_PINN.parent.mkdir(parents=True, exist_ok=True)

# --------- Hiperparâmetros ----------
EPOCHS = 5000
LR = 1e-3

# amostragem por época
N_INT = 2000        # collocation (interior) por época
N_BC = None         # None = usa todas as BCs sempre
N_DATA = 8000       # pontos com pseudo-label do MLP-refino por época (no grid do 80)

# pesos da loss (curriculum simples)
LAMBDA_BC = 10.0
LAMBDA_CONT = 0.5
LAMBDA_DATA_START = 20.0
LAMBDA_DATA_END = 5.0
WARMUP_FRAC = 0.4   # fração das épocas pra decair lambda_data

# --------- PINN (rede para u,v) ----------
class PINN_UV(nn.Module):
    def __init__(self, hidden=128, depth=4, act="tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}
        a = acts.get(act.lower(), nn.Tanh())
        layers = [nn.Linear(2, hidden), a]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), a]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (N,2) -> (u,v)
        return self.net(x)

def continuity_residual(model, X):
    Xg = X.detach().clone().requires_grad_(True)
    uv = model(Xg)
    u = uv[:, 0:1]
    v = uv[:, 1:2]

    grads_u = torch.autograd.grad(
        u, Xg, torch.ones_like(u),
        retain_graph=True, create_graph=True
    )[0]

    grads_v = torch.autograd.grad(
        v, Xg, torch.ones_like(v),
        retain_graph=True, create_graph=True   # <-- mudou aqui
    )[0]

    du_dx = grads_u[:, 0:1]
    dv_dy = grads_v[:, 1:2]
    return du_dx + dv_dy


def lambda_data(epoch):
    # decai linearmente durante warmup
    t = min(epoch / max(1, int(EPOCHS * WARMUP_FRAC)), 1.0)
    return (1 - t) * LAMBDA_DATA_START + t * LAMBDA_DATA_END

@torch.no_grad()
def get_pseudolabel_uv(delta_model, df_batch):
    """Retorna u_hat, v_hat = u_c + du_pred, v_c + dv_pred para um batch do delta parquet."""
    X = torch.tensor(df_batch[["xN","yN","u_c","v_c","p_c"]].to_numpy(), dtype=torch.float32, device=DEVICE)
    u_c = torch.tensor(df_batch["u_c"].to_numpy(), dtype=torch.float32, device=DEVICE).unsqueeze(1)
    v_c = torch.tensor(df_batch["v_c"].to_numpy(), dtype=torch.float32, device=DEVICE).unsqueeze(1)

    d = delta_model(X)
    du_p = d[:, 0:1]
    dv_p = d[:, 1:2]
    u_hat = u_c + du_p
    v_hat = v_c + dv_p
    return u_hat, v_hat

def main():
    # ---- carrega pack 20 (BC + interior) ----
    pack = np.load(NPZ_PACK)
    X_bc = torch.tensor(pack["X_bc"], dtype=torch.float32, device=DEVICE)
    Y_bc = torch.tensor(pack["Y_bc"], dtype=torch.float32, device=DEVICE)
    X_int_all = torch.tensor(pack["X_int"], dtype=torch.float32, device=DEVICE)

    # ---- carrega delta-data (grid do 80) ----
    df_delta = pd.read_parquet(DELTA_DATA)

    # ---- carrega MLP delta (refino) ----
    delta_model = MLP(in_dim=5, out_dim=3, hidden=128, depth=4, act="tanh").to(DEVICE)
    ckpt = torch.load(DELTA_MODEL, map_location=DEVICE)
    delta_model.load_state_dict(ckpt["state_dict"])
    delta_model.eval()

    # ---- cria PINN ----
    pinn = PINN_UV(hidden=128, depth=4, act="tanh").to(DEVICE)
    opt = torch.optim.Adam(pinn.parameters(), lr=LR)

    # indices fixos
    bc_idx_all = torch.arange(len(X_bc), device=DEVICE)
    int_idx_all = torch.arange(len(X_int_all), device=DEVICE)

    for epoch in range(1, EPOCHS + 1):
        pinn.train()

        # --- sample interior ---
        if N_INT is None or N_INT >= len(int_idx_all):
            X_int = X_int_all
        else:
            sel = torch.randint(0, len(int_idx_all), (N_INT,), device=DEVICE)
            X_int = X_int_all[sel]

        # --- sample BC (opcional) ---
        if N_BC is None or N_BC >= len(bc_idx_all):
            Xb = X_bc
            Yb = Y_bc
        else:
            sel = torch.randint(0, len(bc_idx_all), (N_BC,), device=DEVICE)
            Xb = X_bc[sel]
            Yb = Y_bc[sel]

        # --- sample data (pseudo-label do MLP refino) ---
        if N_DATA is None or N_DATA >= len(df_delta):
            df_batch = df_delta
        else:
            df_batch = df_delta.sample(n=N_DATA, random_state=epoch)

        X_data = torch.tensor(df_batch[["xN","yN"]].to_numpy(), dtype=torch.float32, device=DEVICE)
        u_hat, v_hat = get_pseudolabel_uv(delta_model, df_batch)
        Y_data = torch.cat([u_hat, v_hat], dim=1)

        # ---- losses ----
        # BC loss
        pred_bc = pinn(Xb)
        loss_bc = torch.mean((pred_bc - Yb) ** 2)

        # Data loss (PINN ≈ MLP-refino)
        pred_data = pinn(X_data)
        loss_data = torch.mean((pred_data - Y_data) ** 2)

        # Continuity loss (physics)
        r_cont = continuity_residual(pinn, X_int)
        loss_cont = torch.mean(r_cont ** 2)

        lam_data = lambda_data(epoch)
        loss = LAMBDA_BC * loss_bc + lam_data * loss_data + LAMBDA_CONT * loss_cont

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if epoch % 200 == 0 or epoch == 1:
            print(
                f"epoch={epoch:4d} "
                f"loss={loss.item():.3e} "
                f"bc={loss_bc.item():.3e} "
                f"data={loss_data.item():.3e} "
                f"cont={loss_cont.item():.3e} "
                f"lam_data={lam_data:.2f}"
            )

    torch.save({"state_dict": pinn.state_dict()}, OUT_PINN)
    print("\nOK! PINN salva em:", OUT_PINN)

if __name__ == "__main__":
    main()
