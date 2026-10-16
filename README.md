# 🧩 ARC Prize 2026 — Abstraction & Reasoning Corpus (ARC-AGI 2)

> **Competencia Kaggle:** [ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)  
> **Recurso de Cómputo:** 💻 CPU pura (0 slots GPU consumidos — sin restricción de concurrencia)  
> **Internet:** ❌ Desactivado  
> **Métrica:** Exact Match Top-2 (Top-2 Accuracy por problema)  
> **Entregable:** `submissions/submission.json`  

---

## 🎯 Objetivo de la Competencia

Resolver tareas visuales de razonamiento inductivo abstracto donde se entregan pocos pares de demostración (input-grid $\to$ output-grid) y se debe predecir la cuadrícula objetivo para una o dos entradas de prueba.

Kaggle permite **dos intentos por tarea** (`attempt_1` y `attempt_2`). Si cualquiera de las dos coincide exactamente con la matriz esperada, se otorga puntaje completo para esa tarea.

---

## 🏗️ Enfoque Técnico y Metodología

El pipeline implementado en [`kaggle_arc_baseline.ipynb`](kaggle_arc_baseline.ipynb) utiliza un enfoque de **Test-Time Search & Program Synthesis (DSL)**:

1. **Primitivas Geométricas y Simetrías:**
   - D4 symmetry group (rotaciones $90^\circ, 180^\circ, 270^\circ$, reflexiones horizontal/vertical y diagonales).
   - Recorte de fondo / crop no-background (`crop_non_bg`).
2. **Transformaciones Celulares y Físicas:**
   - Gravedad en 4 direcciones (Up, Down, Left, Right).
   - Relleno de regiones encerradas (*fill-enclosed*).
   - Remapeo de frecuencias de color (*freq-rank remap*).
   - Selección de sub-cuadrículas periódicas (*tiling & grid-select*).
3. **Motor de Confianza y Reglas Compuestas:**
   - Cada hipótesis recibe un puntaje de consistencia sobre los ejemplos de entrenamiento de la tarea.
   - Las 2 mejores hipótesis generan `attempt_1` y `attempt_2`.
4. **Time-Budgeting (Control de Tiempo):**
   - Máximo 10–15 segundos por tarea para asegurar que el kernel finalice dentro de las 9 horas en el test set oculto de Kaggle.

---

## 📁 Estructura del Directorio

```text
ARC Prize 2026/ARC Prize 2026/
├── kaggle_arc_baseline.ipynb   # Notebook principal sincronizado con Kaggle
├── kernel-metadata.json        # Metadatos de configuración remota
├── EXPERIMENT_LOG.md           # Registro histórico de versiones y resultados
└── submissions/
    └── submission.json         # Artefacto descargado listo para evaluar
```

---

## 🚀 Cómo Ejecutar

Desde la raíz del repositorio (`Kaggle/`):

```powershell
# Lanzar en Kaggle y descargar submission.json al completar
python kaggle_manager.py run arc

# O mediante PowerShell directo
.\kaggle_remote.ps1 run arc
```
