# Physics-Informed ML for Lid-Driven Cavity (OpenFOAM + ML)

This repository presents a **Machine Learning pipeline applied to Computational Fluid Dynamics (CFD)**, using the canonical **lid-driven cavity** case as a study problem. The main goal is to evaluate **purely data-driven models** and **hybrid physics-informed models** for correcting coarse-grid CFD solutions toward a more refined reference.

---

## 📌 Problem Description

The analyzed flow is the **incompressible, laminar lid-driven cavity**, initially solved using OpenFOAM. The *coarse* simulation is used as input for ML models that learn to correct the velocity field, aiming for higher accuracy and improved physical consistency.

**Main physical setup:**

- Incompressible, Newtonian fluid  
- Laminar regime  
- Solver: `icoFoam`  
- Reynolds number:
  
  \[
  Re = 10
  \]

---

## 🧱 Domain and Mesh

- Geometry: 2D square cavity  
- Characteristic length: \(L = 0.1\)  
- *Coarse* mesh: **20 × 20**  
- Reference for evaluation: **40 × 40**

Two-dimensionality is enforced via the `empty` condition on the front and back faces.

---

## ⚙️ Boundary Conditions

- **Moving lid (`movingWall`):**
  - Prescribed velocity: `U = (1 0 0)`
- **Remaining walls:** no-slip condition  
- **Front and back:** `empty`

---

## 🧠 Machine Learning Pipeline

The pipeline implemented in the Jupyter notebook follows these steps:

1. Extraction of velocity fields from CFD simulations.  
2. Data organization preserving the spatial structure \((x,y)\).  
3. Preprocessing and normalization.  
4. Model training:
   - **MLP (baseline data-driven)**
   - **Hybrid model (MLP + physics correction / PINN-delta)**
5. Quantitative and physics-based evaluation of the results.

---

## 📊 Partial Results

### Global Metrics

Reference definition:

$$
U_{\text{ref}} = U_{\text{base}} + \Delta \quad (\text{mesh } 40 \times 40)
$$

| Metric | MLP | Hybrid |
|------|-----|--------|
| RMSE | 2.34e-02 | 3.26e-02 |
| MAE  | 9.64e-03 | 1.96e-02 |

➡️ The **MLP achieves lower global error** in both L2 and L1 norms compared to the hybrid model.

---

### Incompressibility (Physical Consistency)

Evaluation based on the L2 RMS norm of the velocity divergence:

| Model | L2(div) RMS |
|------|-------------|
| MLP  | 3.85e-01 |
| Hybrid | 4.53e-02 |

**Physical gain obtained:**

- 🔽 Divergence reduction of approximately **8.5×**  
- ✔️ The hybrid model significantly improves **global incompressibility**

---

## 🔍 Key Observations

- Purely data-driven models may show **good global accuracy**, but often violate physical constraints.
- Incorporating physical information during training:
  - significantly reduces divergence,
  - improves global physical consistency,
  - may introduce local *trade-offs* in L2/MAE error.
- Physics-based metrics are essential when evaluating ML models for CFD.

---

## 🚀 Next Steps

- Evaluation on more refined meshes (e.g., 80 × 80).  
- Explicit inclusion of the pressure field.  
- Local error vs. physical improvement analysis.  
- Extension of the pipeline to more complex regimes (higher Reynolds numbers).

---

📌 **Note:**  
This repository is part of an academic study focused on *Physics-Informed Machine Learning* for accelerating and correcting CFD simulations.

---
## **📜 License**
- **Source code:** MIT License  
- **Data, figures, and text:** Creative Commons Attribution 4.0 (CC BY 4.0)

© 2016 Matheus da Silva Borges

---
## **💬 Contact**  
**Author:** `Matheus Borges`

📧 **Email:** borgesmatheus1201@email.com  
🎓 **Lattes Curriculum (CNPq):** https://lattes.cnpq.br/6344448246000027  
🔗 **LinkedIn:** https://www.linkedin.com/in/matheusborges12/  
🐍 **GitHub:** https://github.com/borges12matheus
