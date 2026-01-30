import torch, numpy as np
from pathlib import Path

DELTA_DATA = Path("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/data/prepared/cavity20_base_to_40_delta.npz")
d = np.load(DELTA_DATA)
print("keys:", d.files)
print("X:", d["X"].shape)
print("U_base:", d["U_base"].shape)
print("Delta:", d["Delta"].shape)

# se você quiser: checar o checkpoint do MLP
ckpt = torch.load("/home/matheus/Documentos/Mestrado/Pesquisa/Lid_driven_cavity/outputs/models/mlp_uv_refiner.pt", map_location="cpu")
sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
print("some weights:", list(sd.keys())[:5])

print("first layer:", sd["net.0.weight"].shape)  # (hidden, in_dim)
# última camada (depende do depth), então pega a última weight:
last_w_key = [k for k in sd.keys() if k.endswith("weight")][-1]
print("last layer:", last_w_key, sd[last_w_key].shape)  # (out_dim, hidden)

