import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configurações ---
L = 0.1        # tamanho do domínio
U_lid = 1.0    # velocidade da tampa

files = {
    "20x20": "data/lid_driven_cavity_20.csv",
    "40x40": "data/lid_driven_cavity_40.csv",
    "80x80": "data/lid_driven_cavity_80.csv",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
})

fig, ax = plt.subplots(figsize=(6.2, 4.8))

# ------------------
# Plot
# ------------------
for label, path in files.items():
    df = pd.read_csv(path)

    y = df["Points:1"].to_numpy()
    u = df["U:0"].to_numpy()

    y_star = y / L
    u_star = u / U_lid

    idx = np.argsort(y_star)
    y_star = y_star[idx]
    u_star = u_star[idx]

    # Destaque visual: coarse com marcador leve (opcional)
    if label == "20×20":
        ax.plot(u_star, y_star, linewidth=2.2, marker="o", markersize=3, markevery=12, label=label)
    else:
        ax.plot(u_star, y_star, linewidth=2.2, label=label)

# ------------------
# Estilo e rótulos
# ------------------
ax.set_xlabel(r"$u/U_{\mathrm{lid}}$")
ax.set_ylabel(r"$y/L$")

ax.set_xlim(-0.30, 1.02)
ax.set_ylim(0.0, 1.0)

ax.grid(True, alpha=0.25)
ax.legend(frameon=False, loc="upper left")

# Título curto (opcional)
ax.set_title("Perfil de velocidade na linha central (x/L = 0.5)")

fig.tight_layout()

# ------------------
# Salvar (recomendado)
# ------------------
fig.savefig("figure/centerline_convergence.pdf", bbox_inches="tight")   # ideal para dissertação
fig.savefig("figure/centerline_convergence.svg", bbox_inches="tight")   # ótimo para editar no Inkscape
fig.savefig("figure/centerline_convergence.png", dpi=300, bbox_inches="tight")

plt.show()