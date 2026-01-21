import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from common import MLP, rmse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_20 = Path("data/prepared/cavity20_full_norm.parquet")
OUT_MODEL = Path("outputs/models/mlp_uv_cavity20.pt")
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_parquet(DATA_20)

    # Treino supervisionado: usa todos os pontos (inclui BC e interior)
    X = torch.tensor(df[["xN", "yN"]].to_numpy(), dtype=torch.float32)
    Y = torch.tensor(df[["uN", "vN"]].to_numpy(), dtype=torch.float32)

    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    model = MLP(in_dim=2, out_dim=2, hidden=128, depth=4, act="tanh").to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    best = float("inf")
    for epoch in range(1, 2001):
        model.train()
        loss_acc = 0.0

        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_acc += loss.item() * xb.size(0)

        loss_epoch = loss_acc / len(ds)

        # “early-ish” print
        if epoch % 200 == 0 or epoch == 1:
            print(f"epoch={epoch:4d} mse={loss_epoch:.6e}")

        # salva melhor
        if loss_epoch < best:
            best = loss_epoch
            torch.save({"state_dict": model.state_dict()}, OUT_MODEL)

    print("\nOK! Modelo salvo em:", OUT_MODEL)

if __name__ == "__main__":
    main()
