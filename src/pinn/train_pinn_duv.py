#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Paths ----------------
NPZ_PACK = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/data/prepared/cavity_20/cavity_20_train_pack.npz")
DELTA_DATA = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/data/prepared/cavity20_base_to_40_delta.npz")
DELTA_MODEL = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/outputs/models/mlp_uv_refiner.pt")
OUT_PINN = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/outputs/models/pinn_uv_hybrid.pt")
OUT_PINN.parent.mkdir(parents=True, exist_ok=True)

# ---------------- Hiperparâmetros ----------------
EPOCHS = 5000
LR = 1e-3

N_BC = None
N_DATA = 3000  # 2000~8000

# pesos fixos
LAMBDA_BC = 10.0
LAMBDA_REG = 1e-6

# suavidade do delta (muito útil p/ MAE)
LAMBDA_SMOOTH = 1e-2  # teste 1e-3, 1e-2, 5e-2

# currículo (duas fases)
PHASE_FRAC = 0.4
E_PHASE = int(EPOCHS * PHASE_FRAC)

# Fase A: cola no ref, física leve
REF_A = 5.0
CONT_A = 5.0

# Fase B: aperta física, mantém ref menor
REF_B = 1.0
CONT_B = 20.0

def lambdas(epoch: int):
    if epoch <= E_PHASE:
        return CONT_A, REF_A
    return CONT_B, REF_B

# ---------------- Utils ----------------
def to_tensor(a):
    return torch.tensor(a, dtype=torch.float32, device=DEVICE)

def get_npz_key(z, *candidates):
    for k in candidates:
        if k in z:
            return k
    raise KeyError(f"Nenhuma das chaves {candidates} encontrada. Chaves disponíveis: {list(z.keys())}")

# ---------------- Modelos ----------------
class MLP(nn.Module):
    def __init__(self, in_dim=4, out_dim=2, hidden=128, depth=4, act="tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}
        a = acts.get(act.lower(), nn.Tanh())
        layers = [nn.Linear(in_dim, hidden), a]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), a]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PINN_DUV(nn.Module):
    def __init__(self, hidden=128, depth=4, act="tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}
        a = acts.get(act.lower(), nn.Tanh())
        layers = [nn.Linear(2, hidden), a]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), a]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def continuity_residual_hybrid(delta_model: nn.Module, pinn: nn.Module, X: torch.Tensor, U_base: torch.Tensor):
    """
    div(U_final) = d(u_mlp + du_pinn)/dx + d(v_mlp + dv_pinn)/dy
    U_mlp(X) = U_base + delta_model([X, U_base])
    """
    if not X.requires_grad:
        X = X.detach().clone().requires_grad_(True)

    U_mlp = U_base + delta_model(torch.cat([X, U_base], dim=1))  # (N,2)
    duv = pinn(X)                                                # (N,2)

    u = U_mlp[:, 0:1] + duv[:, 0:1]
    v = U_mlp[:, 1:2] + duv[:, 1:2]

    gu = torch.autograd.grad(u, X, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    gv = torch.autograd.grad(v, X, torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    return gu[:, 0:1] + gv[:, 1:2]

def smooth_loss(pinn: nn.Module, X: torch.Tensor):
    """
    Penaliza gradiente de Δu e Δv (suaviza correção e reduz MAE).
    """
    Xg = X.detach().clone().requires_grad_(True)
    duv = pinn(Xg)
    du = duv[:, 0:1]
    dv = duv[:, 1:2]

    gdu = torch.autograd.grad(du, Xg, torch.ones_like(du), retain_graph=True, create_graph=True)[0]
    gdv = torch.autograd.grad(dv, Xg, torch.ones_like(dv), retain_graph=True, create_graph=True)[0]
    return torch.mean(gdu**2) + torch.mean(gdv**2)

# ---------------- Main ----------------
def main():
    # ---- carrega pack 20 (BC) ----
    pack = np.load(NPZ_PACK)
    kXbc = get_npz_key(pack, "X_bc", "xbc", "Xb")
    X_bc = to_tensor(pack[kXbc])  # (Nbc,2)

    # ---- carrega delta-data (grid 40) ----
    dnpz = np.load(DELTA_DATA)
    X_all = to_tensor(dnpz["X"])            # (N,2)
    U_base_all = to_tensor(dnpz["U_base"])  # (N,2)
    Delta_all = to_tensor(dnpz["Delta"])    # (N,2) ou (N,>=2)
    if Delta_all.shape[1] > 2:
        Delta_all = Delta_all[:, :2]

    U_ref_all = (U_base_all + Delta_all).detach()
    Nd = X_all.shape[0]

    # ---- carrega MLP refiner (congelado) ----
    delta_model = MLP(in_dim=4, out_dim=2, hidden=128, depth=4, act="tanh").to(DEVICE)
    ckpt = torch.load(DELTA_MODEL, map_location=DEVICE)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    delta_model.load_state_dict(sd)
    delta_model.eval()
    for p in delta_model.parameters():
        p.requires_grad_(False)

    # ---- cria PINN corretora ----
    pinn = PINN_DUV(hidden=128, depth=4, act="tanh").to(DEVICE)
    opt = torch.optim.Adam(pinn.parameters(), lr=LR)

    bc_idx_all = torch.arange(len(X_bc), device=DEVICE)
    data_idx_all = torch.arange(Nd, device=DEVICE)

    for epoch in range(1, EPOCHS + 1):
        pinn.train()
        L_CONT, L_REF = lambdas(epoch)

        # --- sample BC ---
        if N_BC is None or N_BC >= len(bc_idx_all):
            sel_bc = bc_idx_all
        else:
            sel_bc = torch.randint(0, len(bc_idx_all), (N_BC,), device=DEVICE)
        Xb = X_bc[sel_bc]

        # --- sample data (grid 40) ---
        if N_DATA is None or N_DATA >= Nd:
            sel_d = data_idx_all
        else:
            sel_d = torch.randint(0, Nd, (N_DATA,), device=DEVICE)

        Xd = X_all[sel_d]
        Ubase = U_base_all[sel_d].detach()
        Uref = U_ref_all[sel_d].detach()

        # X com grad (para continuidade e smooth)
        Xd_g = Xd.detach().clone().requires_grad_(True)

        # --- forward ---
        duv_bc = pinn(Xb)
        duv_d = pinn(Xd_g)

        U_mlp = Ubase + delta_model(torch.cat([Xd_g, Ubase], dim=1))
        U_final = U_mlp + duv_d

        # --- losses ---
        loss_bc = torch.mean(duv_bc ** 2)
        loss_reg = torch.mean(duv_d ** 2)

        r_cont = continuity_residual_hybrid(delta_model, pinn, Xd_g, Ubase)
        loss_cont = torch.mean(r_cont ** 2)

        loss_ref = torch.mean((U_final - Uref) ** 2)

        loss_smooth = smooth_loss(pinn, Xd_g)

        loss = (
            LAMBDA_BC * loss_bc
            + L_CONT * loss_cont
            + L_REF * loss_ref
            + LAMBDA_REG * loss_reg
            + LAMBDA_SMOOTH * loss_smooth
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if epoch % 200 == 0 or epoch == 1:
            print(
                f"epoch={epoch:4d} "
                f"loss={loss.item():.3e} "
                f"bc={loss_bc.item():.3e} "
                f"cont={loss_cont.item():.3e} "
                f"ref={loss_ref.item():.3e} "
                f"reg={loss_reg.item():.3e} "
                f"smooth={loss_smooth.item():.3e} "
                f"L_CONT={L_CONT:.2f} L_REF={L_REF:.2f}"
            )

    torch.save({"state_dict": pinn.state_dict()}, OUT_PINN)
    print("\nOK! PINN(delta) salva em:", OUT_PINN)

if __name__ == "__main__":
    main()
