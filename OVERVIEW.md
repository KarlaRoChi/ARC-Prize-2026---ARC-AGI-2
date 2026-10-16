# 🧩 ARC Prize 2026 — Resumen del Overview Oficial de Kaggle

**Competición:** [ARC Prize 2026 - ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)  
**Organizador:** François Chollet & ARC Prize Foundation  
**Enfoque:** Inteligencia Fluida, Razonamiento Abstracto y Generalización fuera de distribución (OOD)

---

## 1. 🎯 Descripción General y Propósito
El **Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI-2)** representa el benchmark estándar de oro para medir la capacidad de un sistema de IA para resolver problemas novedosos que no pueden solucionarse mediante simple memorización, escala masiva de parámetros o interpolación superficial.

A diferencia de los benchmarks tradicionales de visión o NLP donde los modelos memorizan asociaciones estadísticas de grandes volúmenes de datos preentrenados, ARC evalúa **eficiencia de adquisición de habilidades**:
- Cada tarea plantea una transformación visual y lógica que los humanos resuelven intuitivamente a partir de **unos pocos ejemplos de demostración (few-shot demonstration pairs)**.
- El modelo debe inducir la regla latente y aplicarla sobre una cuadrícula de prueba no vista.
- Las tareas exploran los llamados *Core Knowledge Priors* humanos: permanencia del objeto, topología, simetrías geométricas ($D_4$), aritmética elemental y morfología espacial.

---

## 2. 📏 Métrica y Reglas de Evaluación
- **Métrica:** **Exact Match Top-2 por problema de prueba**.
- **Mecanismo de Evaluación:**
  - Para cada cuadrícula de entrada en el conjunto de prueba (`test`), la solución genera hasta **dos intentos (`attempt_1` y `attempt_2`)**.
  - Se otorga una puntuación de **1** si al menos uno de los dos intentos coincide exactamente en dimensiones y valores de píxeles con la solución real. De lo contrario, la puntuación es **0**.
  - La métrica final en el Leaderboard es la fracción promedio de tareas resueltas exitosamente:
    $$\text{Score} = \frac{1}{N} \sum_{i=1}^N \max\left( \mathbb{I}(\hat{Y}_{i,1} = Y_i), \mathbb{I}(\hat{Y}_{i,2} = Y_i) \right)$$
- **Formato de Salida:** Un único archivo `submission.json` indexado por el identificador de la tarea (`task_id`).

---

## 3. ⏱️ Entorno y Restricciones de Cómputo en Kaggle
- **Tipo de Competición:** *Code Competition* (evaluación a ciegas sobre un conjunto de prueba oculto).
- **Acceso a Internet:** ❌ Estrictamente deshabilitado (`"enable_internet": false`).
- **Aceleración:** CPU o GPU estándar. (En nuestro pipeline multi-agente opera en CPU para no consumir cuota de GPU).
- **Tiempo Límite:** 9 o 12 horas totales para procesar el conjunto de test oculto completo.
- **Estrategia Recomendada:** *Time-budgeting* dinámico por problema (5 a 15 segundos máximo por tarea) con mecanismos de fallback rápidos cuando la síntesis de programas no converge.
