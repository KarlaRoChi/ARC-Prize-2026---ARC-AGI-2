# 📊 ARC Prize 2026 — Resumen del Dataset Description de Kaggle

**Competición:** [ARC Prize 2026 - ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)

---

## 1. 📁 Composición y Estructura del Conjunto de Datos
El dataset de ARC-AGI-2 presenta una calibración humana ampliada diseñada para desafiar los sistemas de razonamiento más avanzados:

| Archivo | Cantidad | Descripción | Propósito |
|---|---|---|---|
| **`arc-agi_training_challenges.json`** | 1,000 tareas | Tareas públicas de dificultad variada (fácil a compleja) | Entrenamiento local, inducción de DSL y definición de primitivas |
| **`arc-agi_training_solutions.json`** | 1,000 tareas | Soluciones completas del conjunto de entrenamiento | Supervisión directa y validación cruzada |
| **`arc-agi_evaluation_challenges.json`** | 120 tareas | Tareas públicas no vistas | Benchmark de validación local (evaluación OOD) |
| **`arc-agi_evaluation_solutions.json`** | 120 tareas | Soluciones del conjunto de evaluación pública | Medición rigurosa de CV local |
| **`arc-agi_test_challenges.json`** | ~100 tareas | Conjunto de prueba privado en la sesión de inferencia oculta de Kaggle | Inferencia final de la competencia |
| **`sample_submission.json`** | Plantilla | Muestra de la estructura JSON esperada con 2 intentos por test pair | Validación de esquema |

---

## 2. 🧩 Formato Interno de los Datos y Codificación de Cuadrículas
Cada archivo JSON contiene un diccionario donde las claves son identificadores alfanuméricos de tarea (`task_id`, e.g. `"007bbfb7"`).

Cada tarea contiene dos listas:
- `"train"`: Lista de 2 a 5 pares de demostración `{"input": [[...]], "output": [[...]]}`.
- `"test"`: Lista de 1 a 3 problemas de prueba `{"input": [[...]]}` (y opcionalmente `"output"` en conjuntos con soluciones).

### Dimensiones y Colores:
- **Dimensiones:** Las matrices tienen dimensiones entre $1 \times 1$ y $30 \times 30$. Las dimensiones de salida pueden ser iguales, submúltiplos o múltiplos de la entrada.
- **Valores de Píxeles (10 colores discretos):**
  - `0`: Negro (usualmente fondo / neutral)
  - `1`: Azul
  - `2`: Rojo
  - `3`: Verde
  - `4`: Amarillo
  - `5`: Gris
  - `6`: Magenta
  - `7`: Naranja
  - `8`: Cian / Teal
  - `9`: Marrón

---

## 3. 📄 Esquema Estricto del Archivo de Submission (`submission.json`)
La sumisión debe ser un archivo JSON válido con la siguiente estructura exacta:

```json
{
  "00576224": [
    {
      "attempt_1": [[0, 0], [0, 0]],
      "attempt_2": [[0, 0], [0, 0]]
    }
  ],
  "007bbfb7": [
    {
      "attempt_1": [[1, 2], [3, 4]],
      "attempt_2": [[1, 1], [1, 1]]
    }
  ]
}
```

### Reglas Críticas del Evaluador:
1. Las dimensiones de cada intento deben ser matrices 2D homogéneas de enteros en el rango $[0, 9]$.
2. Si una tarea contiene múltiples problemas de prueba en `"test"`, la lista debe contener un diccionario por cada problema en orden correspondiente.
3. No debe haber valores nulos (`NaN`), flotantes ni dimensiones irregulares.
