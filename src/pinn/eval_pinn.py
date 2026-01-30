#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- paths (ajuste se precisar) ----------
DELTA_DATA = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/data/prepared/cavity20_base_to_40_delta.npz")
MLP_PATH   = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/outputs/models/mlp_uv_refiner.pt")
PINN_PATH  = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/outputs/models/pinn_uv_hybrid.pt")
OUT_DIR    = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/outputs/figs_hybrid_40")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- modelos (mesmas classes do teu pipeline) ----------
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

# -------- métricas ----------
def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2)).item()

def mae(a, b):
    return torch.mean(torch.abs(a - b)).item()

def div_field(U, X, retain_graph=False):
    """
    U: (N,2) tensor; X: (N,2) requires_grad=True
    Retorna div: (N,1)
    """
    u = U[:, 0:1]
    v = U[:, 1:2]
    gu = torch.autograd.grad(u, X, torch.ones_like(u), retain_graph=True,  create_graph=False)[0]
    gv = torch.autograd.grad(v, X, torch.ones_like(v), retain_graph=retain_graph, create_graph=False)[0]
    div = gu[:, 0:1] + gv[:, 1:2]
    return div

def div_rms(U, X, retain_graph=False):
    d = div_field(U, X, retain_graph=retain_graph)
    return torch.sqrt(torch.mean(d**2)).item(), d

# -------- plot helpers ----------
def savefig(name):
    p = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()
    print("salvo:", p)

def tricontour_map(x, y, z, title, fname, nlevels=40):
    plt.figure(figsize=(6,5))
    tcf = plt.tricontourf(x, y, z, levels=nlevels)
    plt.colorbar(tcf)
    plt.title(title)
    plt.xlabel("xN"); plt.ylabel("yN")
    savefig(fname)

def tricontour_diff(x, y, z1, z2, title, fname, nlevels=40):
    dz = z1 - z2
    tricontour_map(x, y, dz, title, fname, nlevels=nlevels)

def line_profile_nearest(x, y, val, x_target=None, y_target=None, tol=0.02):
    """
    Extrai perfil por 'banda' usando pontos próximos de x=x_target (vertical) ou y=y_target (horizontal).
    - Se x_target definido: seleciona |x-x_target|<=tol e ordena por y.
    - Se y_target definido: seleciona |y-y_target|<=tol e ordena por x.
    Retorna (s, v) onde s é y ou x.
    """
    if x_target is not None:
        m = np.abs(x - x_target) <= tol
        xs, ys, vs = x[m], y[m], val[m]
        idx = np.argsort(ys)
        return ys[idx], vs[idx]
    if y_target is not None:
        m = np.abs(y - y_target) <= tol
        xs, ys, vs = x[m], y[m], val[m]
        idx = np.argsort(xs)
        return xs[idx], vs[idx]
    raise ValueError("Defina x_target ou y_target")

def plot_profiles(x, y, U_ref, U_mlp, U_hyb, tol=0.02):
    # u(x=0.5, y)
    y1, uref = line_profile_nearest(x, y, U_ref[:,0], x_target=0.5, tol=tol)
    y2, umlp = line_profile_nearest(x, y, U_mlp[:,0], x_target=0.5, tol=tol)
    y3, uhyb = line_profile_nearest(x, y, U_hyb[:,0], x_target=0.5, tol=tol)

    plt.figure(figsize=(6,5))
    plt.plot(uref, y1, label="ref40")
    plt.plot(umlp, y2, label="MLP")
    plt.plot(uhyb, y3, label="Híbrido")
    plt.xlabel("u"); plt.ylabel("yN")
    plt.title(f"Perfil u em x=0.5 (tol={tol})")
    plt.legend()
    savefig("perfil_u_x05.png")

    # v(x, y=0.5)
    x1, vref = line_profile_nearest(x, y, U_ref[:,1], y_target=0.5, tol=tol)
    x2, vmlp = line_profile_nearest(x, y, U_mlp[:,1], y_target=0.5, tol=tol)
    x3, vhyb = line_profile_nearest(x, y, U_hyb[:,1], y_target=0.5, tol=tol)

    plt.figure(figsize=(6,5))
    plt.plot(x1, vref, label="ref40")
    plt.plot(x2, vmlp, label="MLP")
    plt.plot(x3, vhyb, label="Híbrido")
    plt.xlabel("xN"); plt.ylabel("v")
    plt.title(f"Perfil v em y=0.5 (tol={tol})")
    plt.legend()
    savefig("perfil_v_y05.png")

def plot_hist_cdf(abs_div_mlp, abs_div_hyb):
    # Hist
    plt.figure(figsize=(6,5))
    plt.hist(abs_div_mlp, bins=60, alpha=0.6, label="MLP", density=True)
    plt.hist(abs_div_hyb, bins=60, alpha=0.6, label="Híbrido", density=True)
    plt.yscale("log")
    plt.xlabel("|div|"); plt.ylabel("densidade (log)")
    plt.title("Histograma de |div|")
    plt.legend()
    savefig("hist_abs_div.png")

    # CDF
    def cdf(a):
        a = np.sort(a)
        p = np.linspace(0, 1, len(a))
        return a, p

    x1, p1 = cdf(abs_div_mlp)
    x2, p2 = cdf(abs_div_hyb)

    plt.figure(figsize=(6,5))
    plt.plot(x1, p1, label="MLP")
    plt.plot(x2, p2, label="Híbrido")
    plt.xscale("log")
    plt.xlabel("|div| (log)"); plt.ylabel("CDF")
    plt.title("CDF de |div| (quanto menor, melhor)")
    plt.legend()
    savefig("cdf_abs_div.png")

def main():
    # --- load data ---
    d = np.load(DELTA_DATA)
    X = torch.tensor(d["X"], dtype=torch.float32, device=DEVICE)
    U_base = torch.tensor(d["U_base"], dtype=torch.float32, device=DEVICE)
    Delta = torch.tensor(d["Delta"], dtype=torch.float32, device=DEVICE)
    if Delta.shape[1] > 2:
        Delta = Delta[:, :2]
    U_ref = U_base + Delta

    # numpy coords for plotting
    x = X[:,0].detach().cpu().numpy()
    y = X[:,1].detach().cpu().numpy()

    # --- load models ---
    mlp = MLP(in_dim=4, out_dim=2, hidden=128, depth=4, act="tanh").to(DEVICE)
    ckpt = torch.load(MLP_PATH, map_location=DEVICE)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    mlp.load_state_dict(sd)
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)

    pinn = PINN_DUV(hidden=128, depth=4, act="tanh").to(DEVICE)
    ckpt2 = torch.load(PINN_PATH, map_location=DEVICE)
    sd2 = ckpt2["state_dict"] if isinstance(ckpt2, dict) and "state_dict" in ckpt2 else ckpt2
    pinn.load_state_dict(sd2)
    pinn.eval()

    # --- infer fields ---
    with torch.no_grad():
        U_base_g = U_base.detach()
        U_mlp = U_base_g + mlp(torch.cat([X, U_base_g], dim=1))
        U_hyb = U_mlp + pinn(X)

    # --- errors (global) ---
    rmse_mlp = rmse(U_mlp, U_ref)
    rmse_hyb = rmse(U_hyb, U_ref)
    mae_mlp  = mae(U_mlp, U_ref)
    mae_hyb  = mae(U_hyb, U_ref)

    # --- divergence (global) ---
    Xg = X.detach().clone().requires_grad_(True)
    # recomputa U_mlp e U_hyb com esse Xg pra garantir grafo
    U_mlp_g = U_base.detach() + mlp(torch.cat([Xg, U_base.detach()], dim=1))
    duv_g = pinn(Xg)
    U_hyb_g = U_mlp_g + duv_g

    div_mlp_rms, div_mlp = div_rms(U_mlp_g, Xg, retain_graph=True)
    div_hyb_rms, div_hyb = div_rms(U_hyb_g, Xg, retain_graph=False)

    # report
    print("\n=== Métricas globais (ref40 = U_base + Delta) ===")
    print(f"RMSE MLP:      {rmse_mlp:.6e}")
    print(f"RMSE Híbrido:  {rmse_hyb:.6e}")
    print(f" MAE MLP:      {mae_mlp:.6e}")
    print(f" MAE Híbrido:  {mae_hyb:.6e}")
    print("\n=== Incompressibilidade (div) ===")
    print(f"L2(div) RMS (MLP):      {div_mlp_rms:.6e}")
    print(f"L2(div) RMS (Híbrido):  {div_hyb_rms:.6e}")
    print(f"Melhora div (x):        {div_mlp_rms / max(div_hyb_rms, 1e-30):.2f}x")

    # ---- maps (u,v) ----
    Uref_np = U_ref.detach().cpu().numpy()
    Umlp_np = U_mlp.detach().cpu().numpy()
    Uhyb_np = U_hyb.detach().cpu().numpy()

    # erro magnitude
    err_mlp = np.sqrt((Umlp_np[:,0]-Uref_np[:,0])**2 + (Umlp_np[:,1]-Uref_np[:,1])**2)
    err_hyb = np.sqrt((Uhyb_np[:,0]-Uref_np[:,0])**2 + (Uhyb_np[:,1]-Uref_np[:,1])**2)

    tricontour_map(x, y, err_mlp, "Erro |U_MLP - U_ref|", "map_err_mlp.png")
    tricontour_map(x, y, err_hyb, "Erro |U_Híbrido - U_ref|", "map_err_hybrid.png")

    # maps de u e v (opcional, mas útil)
    tricontour_map(x, y, Umlp_np[:,0], "u (MLP)", "map_u_mlp.png")
    tricontour_map(x, y, Uhyb_np[:,0], "u (Híbrido)", "map_u_hybrid.png")
    tricontour_map(x, y, Uref_np[:,0], "u (ref40)", "map_u_ref.png")

    tricontour_map(x, y, Umlp_np[:,1], "v (MLP)", "map_v_mlp.png")
    tricontour_map(x, y, Uhyb_np[:,1], "v (Híbrido)", "map_v_hybrid.png")
    tricontour_map(x, y, Uref_np[:,1], "v (ref40)", "map_v_ref.png")

    # ---- div maps ----
    div_mlp_np = div_mlp.detach().cpu().numpy().reshape(-1)
    div_hyb_np = div_hyb.detach().cpu().numpy().reshape(-1)
    tricontour_map(x, y, np.abs(div_mlp_np), "|div| (MLP)", "map_absdiv_mlp.png")
    tricontour_map(x, y, np.abs(div_hyb_np), "|div| (Híbrido)", "map_absdiv_hybrid.png")
    tricontour_diff(x, y, np.abs(div_mlp_np), np.abs(div_hyb_np), "|div| MLP - |div| Híbrido", "map_absdiv_diff.png")

    # ---- profiles ----
    plot_profiles(x, y, Uref_np, Umlp_np, Uhyb_np, tol=0.02)

    # ---- hist/cdf of |div| ----
    plot_hist_cdf(np.abs(div_mlp_np), np.abs(div_hyb_np))

    # ---- resumo em txt ----
    summary = OUT_DIR / "summary_metrics.txt"
    with open(summary, "w") as f:
        f.write("=== Métricas globais (ref40 = U_base + Delta) ===\n")
        f.write(f"RMSE MLP:     {rmse_mlp:.6e}\n")
        f.write(f"RMSE Híbrido: {rmse_hyb:.6e}\n")
        f.write(f"MAE  MLP:     {mae_mlp:.6e}\n")
        f.write(f"MAE  Híbrido: {mae_hyb:.6e}\n\n")
        f.write("=== Incompressibilidade (div) ===\n")
        f.write(f"L2(div) RMS (MLP):     {div_mlp_rms:.6e}\n")
        f.write(f"L2(div) RMS (Híbrido): {div_hyb_rms:.6e}\n")
        f.write(f"Melhora div (x):       {div_mlp_rms / max(div_hyb_rms, 1e-30):.2f}x\n")
    print("salvo:", summary)

if __name__ == "__main__":
    main()
