# 🧩 Registro de Experimentos: ARC Prize 2026

**Métrica:** Exact Match Top-2 Accuracy por problema.  
**Recurso:** CPU (0 slots GPU).

| Versión | Fecha | Hipótesis / Descripción | Cambios Técnicos | CV Local | LB Score | Estado / Conclusiones |
|---|---|---|---|---|---|---|
| `v1_baseline` | 2026-10-15 | Pipeline base de inferencia y generación de formato `submission.json` | Heurísticas geométricas directas + matrices nulas de reserva | N/A | `0.00` | Baseline inicial desplegado y funcionando correctamente |
| `v2_dsl_solver` | 2026-10-15 | Motor DSL multi-heurístico con búsqueda de reglas y fallbacks estructurados | Grupo D4, crop bbox no-bg, gravedad up/down, mapeo de color compuesto, fallbacks no triviales | N/A | `0.00` | Ejecutado correctamente. Score bajo por cobertura de reglas limitada. |
| `v3_object_centric` | 2026-10-15 | **Confidence-scored rule engine** + 10 nuevas familias de reglas | Tiling `np.tile`, Kronecker, Grid-select, Fill-enclosed (scipy flood-fill), Mirror-complete H/V, Freq-rank remap, Fixed-output-size crop/fill, gravity L/R, deadline timeout por tarea (10s) | N/A | `0.00` | Reglas evaluadas con score continuo (fracción de train pairs); top-2 → attempt_1/attempt_2. Mayor cobertura de patrones ARC frecuentes. |
| `v4_subgrid_object_algebra` | 2026-10-15 | **Subgrid Partition Logic & Object Extractor** para tareas composicionales complejas | Detección de líneas divisorias horizontales/verticales unicolores (`detect_dividers`) con superposición lógica (`np.where(B != 0, B, T)`), extracción de componente conectado de área máxima (`extract_largest_object`) y mínima (`extract_smallest_object`). Asignación estricta de 2 hipótesis por test. | 0.080 | Listo para push | Resuelve tareas de superposición de cuadrículas y aislamiento de objetos salientes en el dataset de validación sin sobrepasar el presupuesto de cómputo CPU. |

