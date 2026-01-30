import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== Ajuste caminhos ======
DATA_80 = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/data/prepared/cavity80_full_norm.parquet")

PINN_CKPT = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/outputs/models/pinn_uv_hybrid.pt")

# (opcional) para plotar junto:
MLP_UV_CKPT = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/outputs/models/mlp_uv_cavity20.pt")             # MLP direto (x,y)->(u,v)
DELTA_DATA = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/data/prepared/cavity20_to_80_delta.parquet")      # para MLP refino
MLP_DELTA_CKPT = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/outputs/models/mlp_delta_20_to_80.pt")        # MLP delta (x,y,u_c,v_c,p_c)->(du,dv,dp)

OUTDIR = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/outputs/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ====== Modelos ======
class MLP(torch.nn.Module):
    def __init__(self, in_dim=2, out_dim=2, hidden=128, depth=4, act="tanh"):
        super().__init__()
        acts = {"tanh": torch.nn.Tanh(), "relu": torch.nn.ReLU(), "gelu": torch.nn.GELU(), "silu": torch.nn.SiLU()}
        a = acts.get(act.lower(), torch.nn.Tanh())
        layers = [torch.nn.Linear(in_dim, hidden), a]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(hidden, hidden), a]
        layers += [torch.nn.Linear(hidden, out_dim)]
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class PINN_UV(torch.nn.Module):
    def __init__(self, hidden=128, depth=4, act="tanh"):
        super().__init__()
        acts = {"tanh": torch.nn.Tanh(), "relu": torch.nn.ReLU(), "gelu": torch.nn.GELU(), "silu": torch.nn.SiLU()}
        a = acts.get(act.lower(), torch.nn.Tanh())
        layers = [torch.nn.Linear(2, hidden), a]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(hidden, hidden), a]
        layers += [torch.nn.Linear(hidden, 2)]
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def predict_pinn(df80: pd.DataFrame) -> np.ndarray:
    X = torch.tensor(df80[["xN","yN"]].to_numpy(), dtype=torch.float32, device=DEVICE)
    pinn = PINN_UV(hidden=128, depth=4, act="tanh").to(DEVICE)
    ckpt = torch.load(PINN_CKPT, map_location=DEVICE)
    pinn.load_state_dict(ckpt["state_dict"])
    pinn.eval()
    uv = pinn(X).cpu().numpy()
    return uv  # (N,2)

@torch.no_grad()
def predict_mlp_uv(df80: pd.DataFrame) -> np.ndarray:
    X = torch.tensor(df80[["xN","yN"]].to_numpy(), dtype=torch.float32, device=DEVICE)
    mlp = MLP(in_dim=2, out_dim=2, hidden=128, depth=4, act="tanh").to(DEVICE)
    ckpt = torch.load(MLP_UV_CKPT, map_location=DEVICE)
    mlp.load_state_dict(ckpt["state_dict"])
    mlp.eval()
    uv = mlp(X).cpu().numpy()
    return uv

@torch.no_grad()
def predict_mlp_refine(df80: pd.DataFrame) -> np.ndarray:
    """
    Reconstrói u_hat,v_hat nos pontos do DELTA_DATA (que já estão no grid do 80).
    IMPORTANTE: aqui assumimos que DELTA_DATA tem as mesmas (xN,yN) do df80.
    """
    df = pd.read_parquet(DELTA_DATA)
    # garante mesma ordem do df80: merge por (xN,yN)
    dfm = df80[["xN","yN"]].merge(df, on=["xN","yN"], how="left")
    if dfm[["u_c","v_c","p_c"]].isna().any().any():
        raise RuntimeError("DELTA_DATA não casou 100% com DATA_80 em (xN,yN).")

    X = torch.tensor(dfm[["xN","yN","u_c","v_c","p_c"]].to_numpy(), dtype=torch.float32, device=DEVICE)
    u_c = torch.tensor(dfm["u_c"].to_numpy(), dtype=torch.float32, device=DEVICE).unsqueeze(1)
    v_c = torch.tensor(dfm["v_c"].to_numpy(), dtype=torch.float32, device=DEVICE).unsqueeze(1)

    delta = MLP(in_dim=5, out_dim=3, hidden=128, depth=4, act="tanh").to(DEVICE)
    ckpt = torch.load(MLP_DELTA_CKPT, map_location=DEVICE)
    delta.load_state_dict(ckpt["state_dict"])
    delta.eval()
    d = delta(X)
    u_hat = (u_c + d[:,0:1]).cpu().numpy()
    v_hat = (v_c + d[:,1:2]).cpu().numpy()
    uv = np.concatenate([u_hat, v_hat], axis=1)
    return uv

def centerline_x(df: pd.DataFrame, x0=0.5) -> pd.DataFrame:
    x_vals = np.sort(df["xN"].unique())
    dx = np.min(np.diff(x_vals)) if len(x_vals) > 1 else 1e-3
    tol = dx/2 + 1e-12
    line = df[np.abs(df["xN"] - x0) <= tol].copy().sort_values("yN")
    return line, tol

def centerline_y(df: pd.DataFrame, y0=0.5) -> pd.DataFrame:
    y_vals = np.sort(df["yN"].unique())
    dy = np.min(np.diff(y_vals)) if len(y_vals) > 1 else 1e-3
    tol = dy/2 + 1e-12
    line = df[np.abs(df["yN"] - y0) <= tol].copy().sort_values("xN")
    return line, tol

def plot_centerlines(df80, uv_pinn, uv_mlp=None, uv_refine=None):
    # anexa previsões ao df
    dfp = df80.copy()
    dfp["u_pinn"] = uv_pinn[:,0]
    dfp["v_pinn"] = uv_pinn[:,1]
    if uv_mlp is not None:
        dfp["u_mlp"] = uv_mlp[:,0]
        dfp["v_mlp"] = uv_mlp[:,1]
    if uv_refine is not None:
        dfp["u_refine"] = uv_refine[:,0]
        dfp["v_refine"] = uv_refine[:,1]

    # --- u vs y em x=0.5 ---
    line, tol = centerline_x(dfp, x0=0.5)
    y = line["yN"].to_numpy()
    u_ref = line["uN"].to_numpy()

    plt.figure(figsize=(6.2, 4.8))
    plt.plot(u_ref, y, linewidth=2.2, label="CFD 80×80 (ref)")
    plt.plot(line["u_pinn"].to_numpy(), y, linewidth=2.2, linestyle="--", label="PINN híbrida")
    if "u_mlp" in line.columns:
        plt.plot(line["u_mlp"].to_numpy(), y, linewidth=2.0, linestyle=":", label="MLP direto")
    if "u_refine" in line.columns:
        plt.plot(line["u_refine"].to_numpy(), y, linewidth=2.0, linestyle="-.", label="MLP refino")

    plt.xlabel(r"$u/U_{\mathrm{lid}}$")
    plt.ylabel(r"$y/L$")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    out = OUTDIR / "centerline_u_x0p5.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Salvo:", out, "| tol usado:", tol)

    # --- v vs x em y=0.5 ---
    line, tol = centerline_y(dfp, y0=0.5)
    x = line["xN"].to_numpy()
    v_ref = line["vN"].to_numpy()

    plt.figure(figsize=(6.2, 4.8))
    plt.plot(x, v_ref, linewidth=2.2, label="CFD 80×80 (ref)")
    plt.plot(x, line["v_pinn"].to_numpy(), linewidth=2.2, linestyle="--", label="PINN híbrida")
    if "v_mlp" in line.columns:
        plt.plot(x, line["v_mlp"].to_numpy(), linewidth=2.0, linestyle=":", label="MLP direto")
    if "v_refine" in line.columns:
        plt.plot(x, line["v_refine"].to_numpy(), linewidth=2.0, linestyle="-.", label="MLP refino")

    plt.xlabel(r"$x/L$")
    plt.ylabel(r"$v/U_{\mathrm{lid}}$")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    out = OUTDIR / "centerline_v_y0p5.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Salvo:", out, "| tol usado:", tol)


def plot_error_maps(df80, uv_pred, tag="pinn"):
    x = df80["xN"].to_numpy()
    y = df80["yN"].to_numpy()

    u_ref = df80["uN"].to_numpy()
    v_ref = df80["vN"].to_numpy()

    u_p = uv_pred[:, 0]
    v_p = uv_pred[:, 1]

    eu = np.abs(u_p - u_ref)
    ev = np.abs(v_p - v_ref)

    tri = mtri.Triangulation(x, y)

    # |u - u_ref|
    plt.figure(figsize=(6.2, 4.8))
    im = plt.tricontourf(tri, eu, levels=40)
    plt.colorbar(im, label=r"$|u - u_{\mathrm{ref}}|$")
    plt.xlabel(r"$x/L$")
    plt.ylabel(r"$y/L$")
    plt.tight_layout()
    out = OUTDIR / f"error_map_u_{tag}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Salvo:", out)

    # |v - v_ref|
    plt.figure(figsize=(6.2, 4.8))
    im = plt.tricontourf(tri, ev, levels=40)
    plt.colorbar(im, label=r"$|v - v_{\mathrm{ref}}|$")
    plt.xlabel(r"$x/L$")
    plt.ylabel(r"$y/L$")
    plt.tight_layout()
    out = OUTDIR / f"error_map_v_{tag}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Salvo:", out)

def main():
    df80 = pd.read_parquet(DATA_80)

    # ref
    uv_pinn = predict_pinn(df80)

    # opcionais (se você quiser comparar tudo)
    uv_mlp = None
    if MLP_UV_CKPT.exists():
        uv_mlp = predict_mlp_uv(df80)

    uv_refine = None
    if DELTA_DATA.exists() and MLP_DELTA_CKPT.exists():
        uv_refine = predict_mlp_refine(df80)

    plot_centerlines(df80, uv_pinn, uv_mlp=uv_mlp, uv_refine=uv_refine)

    # mapas de erro (Pinn)
    plot_error_maps(df80, uv_pinn, tag="pinn")

if __name__ == "__main__":
    main()
