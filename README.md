# Physics-Informed ML for Lid-Driven Cavity (OpenFOAM + ML)

Este repositório apresenta um pipeline de **Machine Learning aplicado à Dinâmica dos Fluidos Computacional (CFD)**, utilizando o caso canônico **lid-driven cavity** como problema de estudo. O foco é avaliar modelos **puramente data-driven** e **modelos híbridos com informação física** na tarefa de correção de soluções CFD de malha grosseira (*coarse*) para uma referência mais refinada.

---

## 📌 Descrição do Problema

O escoamento analisado é o **lid-driven cavity incompressível e laminar**, resolvido inicialmente via OpenFOAM. A simulação *coarse* é utilizada como entrada para modelos de ML que aprendem a corrigir o campo de velocidade, buscando maior acurácia e melhor consistência física.

**Configuração física principal:**

- Fluido incompressível e Newtoniano  
- Regime laminar  
- Solver: `icoFoam`  
- Número de Reynolds:
  
  \[
  Re = 10
  \]

---

## 🧱 Domínio e Malha

- Geometria: cavidade quadrada 2D  
- Comprimento característico: \(L = 0.1\)  
- Malha *coarse*: **20 × 20**  
- Referência para avaliação: **40 × 40**

A bidimensionalidade é garantida via condição `empty` nas faces frontal e traseira.

---

## ⚙️ Condições de Contorno

- **Tampa móvel (`movingWall`):**
  - Velocidade imposta: `U = (1 0 0)`
- **Demais paredes:** condição de não deslizamento (*no-slip*)
- **Front and back:** `empty`

---

## 🧠 Pipeline de Machine Learning

O pipeline implementado no notebook Jupyter segue as seguintes etapas:

1. Extração dos campos de velocidade das simulações CFD.
2. Organização dos dados preservando a estrutura espacial \((x,y)\).
3. Pré-processamento e normalização.
4. Treinamento de modelos:
   - **MLP (baseline data-driven)**
   - **Modelo híbrido (MLP + correção física / PINN-delta)**
5. Avaliação quantitativa e física dos resultados.

---

## 📊 Resultados Parciais

### Métricas Globais

Referência utilizada:

$$
U_{\text{ref}} = U_{\text{base}} + \Delta \quad (\text{malha } 40 \times 40)
$$

| Métrica | MLP | Híbrido |
|-------|-----|---------|
| RMSE  | 2.34e-02 | 3.26e-02 |
| MAE   | 9.64e-03 | 1.96e-02 |

➡️ O **MLP apresenta menor erro global** em norma L2 e L1 quando comparado ao modelo híbrido.

---

### Incompressibilidade (Consistência Física)

Avaliação baseada na norma L2 RMS da divergência do campo de velocidade:

| Modelo | L2(div) RMS |
|-------|-------------|
| MLP   | 3.85e-01 |
| Híbrido | 4.53e-02 |

**Ganho físico obtido:**

- 🔽 Redução da divergência em aproximadamente **8.5×**
- ✔️ O modelo híbrido melhora significativamente a **incompressibilidade global** do escoamento

---

## 🔍 Principais Observações

- Modelos puramente data-driven podem apresentar **boa acurácia global**, mas tendem a violar restrições físicas.
- A inclusão de informação física no treinamento:
  - reduz significativamente a divergência do campo,
  - melhora a consistência física global,
  - pode introduzir *trade-offs* locais em termos de erro L2/MAE.
- Métricas físicas são essenciais para avaliar modelos ML aplicados a CFD.

---

## 🚀 Próximos Passos

- Avaliação em malhas mais refinadas (ex.: 80 × 80).
- Inclusão explícita do campo de pressão.
- Análise local de erro vs. melhoria física.
- Extensão do pipeline para regimes mais complexos (Re mais elevados).

---

📌 **Nota:**  
Este repositório faz parte de um estudo acadêmico focado em *Physics-Informed Machine Learning* aplicado à aceleração e correção de simulações CFD.
