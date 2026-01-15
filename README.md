---
title: Regresión Logística sin Caja Negra
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 📊 Regresión Logística sin Caja Negra: Estadística, Optimización y Algoritmos sobre un Caso Real

**[Read in English](#english-version)**

---
## En Español

### 📘 Descripción
Proyecto para estudiar **regresión logística** desde la estadística y la optimización, sin depender de cajas negras. La API expone un modelo entrenado sobre el dataset numérico de crédito alemán y devuelve predicciones de tres optimizadores distintos para comparar su comportamiento.

- 🧮 Implementación manual de la función sigmoid estable
- 📉 Ajuste de parámetros mediante tres algoritmos de optimización
- 🧠 Normalización con media y desviación estándar almacenadas en el modelo
- 🚀 API FastAPI lista para probar el modelo y comparar optimizadores
- 🐳 Contenedor Docker opcional (secundario frente a la parte matemática)

### 📝 Blog explicativo con demo

> Para un recorrido completo paso a paso con demostración interactiva, [lee el artículo en mi blog](https://jeffangel.github.io/blog/regresion-logistica-glm-sin-caja-negra).


### 📐 Fundamentos estadísticos y matemáticos

- Modelo: $p(y=1 \mid \mathbf{x}) = \sigma(\mathbf{x}^\top \beta)$ con $\sigma$ estable.
- Log-verosimilitud negativa: $\ell(\beta) = - \sum_i \big[y_i \log p_i + (1-y_i) \log (1-p_i)\big]$.
- Gradiente: $\nabla \ell(\beta) = X^\top (p - y)$.
- Hessiano: $H = X^\top W X$ con $W$ diagonal y $w_i = p_i (1-p_i)$.
- Normalización Z-score previa: $\tilde{x}_j = (x_j - \mu_j) / \sigma_j$ y se agrega el intercepto como primer término.

### ⚙️ Optimización implementada (tres rutas de entrenamiento)

- **Newton-Raphson**: usa $H^{-1} \nabla \ell$ para convergencia cuadrática en las últimas iteraciones.
- **Descenso de gradiente** (paso fijo): actualización $\beta \leftarrow \beta - \alpha \nabla \ell$, estable para comparar contra Newton.
- **Descenso de gradiente con backtracking**: búsqueda de paso tipo Armijo para controlar la tasa de aprendizaje y robustecer la convergencia.

Cada inferencia devuelve el resultado de los tres optimizadores para que puedas observar diferencias en $\eta$ y $p$.

> Nota conceptual:
> Los tres métodos optimizan **la misma función de log-verosimilitud**.
> Las diferencias observadas en las predicciones intermedias se deben
> exclusivamente al método de optimización y a su trayectoria de convergencia,
> no a cambios en el modelo estadístico.

### 🔬 Cómo se entrena

- Dataset: **Statlog (German Credit Data)** — 1000 observaciones, 24 variables numéricas,
  problema de clasificación binaria.
  Disponible en el repositorio oficial de UCI:
  https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

- Notebooks paso a paso (derivaciones, optimización y experimentos):
  - Español: `notebooks/es/optimizacion_desde_cero.ipynb`
  - Inglés: `notebooks/en/optimization_from_scratch.ipynb`
- El objeto entrenado contiene: medias, desviaciones estándar y coeficientes `beta_nr`, `beta_gd`, `beta_bt` correspondientes a Newton, GD y GD con backtracking.

### 📂 Estructura del proyecto

```
logistic-regression-no-black-box/
├── notebooks/
│   ├── es/optimizacion_desde_cero.ipynb       # Derivaciones y entrenamiento en español
│   └── en/optimization_from_scratch.ipynb     # Derivaciones y entrenamiento en inglés
├── src/
│   ├── app.py                                 # Aplicación FastAPI
│   ├── config.py                              # Configuración (HF_TOKEN opcional si el repo es privado)
│   ├── routers/
│   │   ├── health.py                          # Health check
│   │   ├── predict.py                         # Predicción con tres optimizadores
│   │   └── sample.py                          # Muestra aleatoria del dataset
│   ├── services/
│   │   ├── inference_service.py               # Normaliza, agrega intercepto y aplica sigmoids
│   │   └── sampler_service.py                 # Selección aleatoria de fila
│   └── utils/
│       ├── activations.py                     # Sigmoid estable
│       └── startup.py                         # Descarga y carga del objeto entrenado
├── data/german.data-numeric                   # Dataset numérico original
├── Dockerfile                                 # Contenedor opcional
├── requirements.txt                           # Dependencias
├── run.sh                                     # Arranque local
├── main.py                                    # Punto de entrada simple
└── README.md
```

### 📌 Licencia

MIT - Úsalo, modifícalo y compártelo con atribución.

---

<a id="english-version"></a>

## In English

### 📘 Description
Project to study **logistic regression** from a statistical and optimization-first perspective. The API serves an already trained model on the numeric German credit dataset and returns the outputs of three different optimizers so you can compare them.

- Stable sigmoid, manual normalization, no ML black boxes
- Three optimizers to contrast: Newton-Raphson, GD, GD with backtracking
- FastAPI endpoints to fetch a sample and score it with all solvers
- Docker is available but secondary to the math and demo

### 📝 Walkthrough blog with demo

> For a guided walkthrough with an interactive demo, [read the article on my blog](https://jeffangel.github.io/blog/regresion-logistica-glm-sin-caja-negra).

### 📐 Statistical and mathematical backbone

- Model: $p(y=1 \mid \mathbf{x}) = \sigma(\mathbf{x}^\top \beta)$.
- Negative log-likelihood: $\ell(\beta) = - \sum_i \big[y_i \log p_i + (1-y_i) \log (1-p_i)\big]$.
- Gradient: $\nabla \ell(\beta) = X^\top (p - y)$.
- Hessian: $H = X^\top W X$ with $w_i = p_i (1-p_i)$.
- Z-score normalization and intercept appended before inference.

### ⚙️ Optimization tracks

- **Newton-Raphson** for fast convergence using $H^{-1} \nabla \ell$.
- **Gradient descent** with fixed step for baseline behavior.
- **Gradient descent with backtracking** (Armijo-like) to adapt the learning rate per step.

Every prediction surfaces $\eta$ and $p$ from each optimizer.

> Conceptual note:
> All three methods optimize **the same log-likelihood function**.
> The differences observed in the intermediate predictions are due
> exclusively to the optimization method and its convergence trajectory,
> not to changes in the underlying statistical model.


### 🔬 Training notes

- Dataset: **Statlog (German Credit Data)** — 1000 observations, 24 numeric features,
  binary classification problem.
  Available from the official UCI repository:
  https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- Walkthrough notebooks (derivations, optimization, experiments):
  - Spanish: `notebooks/es/optimizacion_desde_cero.ipynb`
  - English: `notebooks/en/optimization_from_scratch.ipynb`
- The persisted object packs means, standard deviations and `beta_nr`, `beta_gd`, `beta_bt` coefficients.

### 📡 API endpoints

- `/health/` — health check.
- `/sample/` — random normalized row with its class.
- `/predict/` — accepts `new_sample` (24-length list) and returns all three optimizers.

### 📂 Project structure

```
logistic-regression-no-black-box/
├── notebooks/
│   ├── es/optimizacion_desde_cero.ipynb       # Spanish training notebook
│   └── en/optimization_from_scratch.ipynb     # English training notebook
├── src/
│   ├── app.py                                 # FastAPI app
│   ├── config.py                              # Settings (HF_TOKEN optional)
│   ├── routers/
│   │   ├── health.py                          # Health check
│   │   ├── predict.py                         # Prediction endpoint
│   │   └── sample.py                          # Sampling endpoint
│   ├── services/
│   │   ├── inference_service.py               # Normalize, intercept, sigmoid
│   │   └── sampler_service.py                 # Random sample selection
│   └── utils/
│       ├── activations.py                     # Stable sigmoid
│       └── startup.py                         # Download/load artifacts
├── data/german.data-numeric                   # Original numeric dataset
├── Dockerfile
├── requirements.txt
├── run.sh
├── main.py
└── README.md
```

MIT - Feel free to use, modify and share with attribution.
