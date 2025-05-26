# DamasGameIA

Este proyecto implementa un sistema modular para jugar damas usando diferentes técnicas de Inteligencia Artificial, como búsqueda clásica (Minimax, CSP), heurísticas y redes neuronales (MLP).
Video demostrativo: [YouTube](https://youtu.be/eqP38jzPTPY)

---

## Estructura de Archivos

### `board.py`

Define la representación interna del tablero de damas:

- Tablero como matriz 8x8 (`.` casillas claras, `-` casillas oscuras vacías).
- Inicialización de partidas.
- Verificación de movimientos válidos, coronación y clonación de estados.
- Generación de tableros aleatorios válidos/inválidos.
- Función heurística simple: diferencia de material + movilidad.

---

### `search.py`

Contiene todos los algoritmos de búsqueda utilizados:

- **Greedy:** Selección del mejor movimiento inmediato.
- **Minimax con poda alfa-beta:** Análisis profundo de jugadas futuras.
- **Monte Carlo Tree Search (simplificado):** Simulaciones aleatorias por jugada.
- **Hill Climbing:** Optimización local por iteración de jugadas.
- **Algoritmo Genético:** Evolución de tableros con cruce, mutación y evaluación.

---

### `mlp.py`

Define e implementa una red neuronal **MLP** desde cero:

- Arquitectura: 64 entradas → 128 → 64 → 1 neurona.
- Funciones de activación: ReLU y sigmoide.
- Entrenamiento para:
  - **Regresión:** Evaluar calidad de tableros.
  - **Clasificación:** Determinar si un tablero es válido o inválido.
- Optimizadores: `SGD` y `Adam`.

---

### `csp_solver.py`

Agente basado en **problemas de satisfacción de restricciones (CSP)**:

- Genera todas las jugadas legales que respetan reglas del juego.
- Selecciona:
  - la primera jugada válida si no hay red neuronal,
  - o la mejor puntuación si se usa MLP.

---

### `metrics.py`

Evalúa cuantitativamente el rendimiento de los agentes y modelos:

- **MSE (error cuadrático medio)** para regresión.
- **Precisión** para clasificación de tableros.
- **Contadores de nodos explorados** en algoritmos de búsqueda.
- **Tiempos promedio de decisión** por agente.

---

### `main.py`

Archivo principal que coordina todo el sistema:

- Genera datasets de entrenamiento y prueba.
- Entrena la red neuronal (regresión y clasificación).
- Ejecuta comparaciones entre agentes (Greedy, Minimax, MCTS, etc.).
- Calcula estadísticas: coincidencia con Minimax, diferencias de evaluación, tiempos.
