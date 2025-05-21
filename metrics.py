# contadores globales para medir la cantidad de nodos explorados por los algoritmos Minimax y MCTS
minimax_nodes = 0
mcts_nodes = 0

# funcion para reiniciar los contadores de nodos antes de una nueva ejecucion
def reset_counters():
    global minimax_nodes, mcts_nodes
    minimax_nodes = 0
    mcts_nodes = 0

# funcion para evaluar la precision del modelo MLP en una tarea de clasificacion compara la prediccion del modelo contra el valor real usando una tolerancia de error
def evaluate_accuracy(model, X, Y, tolerance=0.5):
    correct = 0
    total = len(X)
    for i in range(total):
        # obtener prediccion del modelo
        pred, _ = model.predict(X[i])
        if abs(pred - Y[i]) <= tolerance:
            correct += 1
    # retornar proporcion de predicciones correctas
    return correct / total if total > 0 else 0.0

# funcion que calcula el error cuadratico medio mide que tan lejos esta la prediccion del valor real en promedio
def mean_squared_error(model, X, Y):
    # acumulador del error total
    total = 0
    for i in range(len(X)):
        pred, _ = model.predict(X[i])
        total += (pred - Y[i]) ** 2
    # retornar promedio del error
    return total / len(X) if X else 0.0
