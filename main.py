import random
import time
import statistics
from statistics import mean, pstdev
from board import Board
from mlp import MLP, board_to_features
from search import *
import csp_solver
from metrics import *

# semilla para reproducibilidad
random.seed(42)

# genera un tablero valido con una cantidad intermedia de piezas y al menos una jugada legal
def generate_midgame_position(min_pieces=8, max_pieces=20):
    while True:
        b = Board.generate_random_valid()
        if min_pieces <= b.count_pieces() <= max_pieces and get_all_moves(b, 'black'):
            return b

# genera tableros invalidos variados forzando violaciones
def generate_varied_invalid():
    b1 = Board.generate_random_valid()
    # poner pieza en casilla clara 
    i, j = random.choice([(i,j) for i in range(8) for j in range(8) if (i+j)%2==0])
    b1.board[i][j] = random.choice(['n','b'])
    if not b1.validate(): return b1
    b2 = Board.generate_random_valid()
    # insertar piezas de forma aleatoria
    for _ in range(5):
        ii, jj = random.choice([(i,j) for i in range(8) for j in range(8) if (i+j)%2==1])
        if b2.board[ii][jj]=='-':
            b2.board[ii][jj] = random.choice(['n','b'])
    if not b2.validate(): return b2
    # insertar una pieza negra fija en la fila central para crear posible conflicto
    b3 = Board.generate_random_valid()
    for jj in range(8):
        if (3+jj)%2==1:
            b3.board[3][jj] = 'n'
            break
    return b3

# evalua un movimiento de forma segura
def safe_eval(move, default):
    return move.heuristic_evaluation() if move else default

# generacion de dataset de entrenamiento y prueba para regresion
# tableros validos
boards_reg = [generate_midgame_position() for _ in range(1000)]
labels_reg = [b.heuristic_evaluation() for b in boards_reg]
# 80% para entrenamiento
split = int(0.8 * len(boards_reg))

# convertir tableros a features
Xr_train = [board_to_features(b) for b in boards_reg[:split]]
yr_train = labels_reg[:split]
Xr_test  = [board_to_features(b) for b in boards_reg[split:]]
yr_test  = labels_reg[split:]

# normalizacion de features media y desviacion estandar para mejorar el aprendizaje
means = [mean([x[i] for x in Xr_train]) for i in range(64)]
stds  = [pstdev([x[i] for x in Xr_train]) or 1.0 for i in range(64)]
def normalize(X):
    return [[(x[i] - means[i]) / stds[i] for i in range(64)] for x in X]

Xr_train = normalize(Xr_train)
Xr_test  = normalize(Xr_test)

# curva de aprendizaje compara SGD y Adam con distintos tamaños de datos
print("--- Curva de aprendizaje MLP (regresión) ---")
for frac in [0.1,0.25,0.5,0.75,1.0]:
    n = int(len(Xr_train) * frac)
    X_sub, y_sub = Xr_train[:n], yr_train[:n]

    # entrenamiento de MLP con SGD y Adam
    mlp_sgd = MLP(); mlp_sgd.train_SGD(X_sub, y_sub, epochs=20, lr=0.01, batch_size=32)
    mse_sgd_tr = mean_squared_error(mlp_sgd, X_sub, y_sub)
    mse_sgd_te = mean_squared_error(mlp_sgd, Xr_test, yr_test)

    mlp_adam = MLP(); mlp_adam.train_Adam(X_sub, y_sub, epochs=20, lr=0.001, batch_size=32)
    mse_adam_tr = mean_squared_error(mlp_adam, X_sub, y_sub)
    mse_adam_te = mean_squared_error(mlp_adam, Xr_test, yr_test)

    print(f"Train={n:4d} | SGD tr={mse_sgd_tr:.4f}, te={mse_sgd_te:.4f} | "
          f"Adam tr={mse_adam_tr:.4f}, te={mse_adam_te:.4f}")
print()

# entrenamiento final de MLP con todos los datos
print("--- Entrenamiento final MLP regresor ---")
# SGD
mlp_sgd = MLP()
t0 = time.perf_counter()
loss_sgd = mlp_sgd.train_SGD(Xr_train, yr_train, epochs=100, lr=0.001, batch_size=32)
t_s = time.perf_counter() - t0
mse_s = mean_squared_error(mlp_sgd, Xr_test, yr_test)
print(f"SGD  -> Loss={loss_sgd:.4f}, Test MSE={mse_s:.4f}, Time={t_s:.2f}s")

# Adam
mlp_adam = MLP()
t0 = time.perf_counter()
loss_a = mlp_adam.train_Adam(Xr_train, yr_train, epochs=100, lr=0.001, batch_size=32)
t_a = time.perf_counter() - t0
mse_a = mean_squared_error(mlp_adam, Xr_test, yr_test)
print(f"Adam -> Loss={loss_a:.4f}, Test MSE={mse_a:.4f}, Time={t_a:.2f}s\n")

optimizers = {'SGD': mlp_sgd, 'Adam': mlp_adam}

# clasificador binario para tableros validos vs invalidos
print("--- Entrenando clasificador inválidos (SGD) ---")
boards_cls = [generate_midgame_position() for _ in range(500)] + \
             [generate_varied_invalid() for _ in range(500)]
labels_cls = [1]*500 + [0]*500
z = list(zip(boards_cls, labels_cls)); random.shuffle(z)
boards_cls, labels_cls = zip(*z)
split2 = int(0.8 * len(boards_cls))

Xc_tr = normalize([board_to_features(b) for b in boards_cls[:split2]])
yc_tr = labels_cls[:split2]
Xc_te = normalize([board_to_features(b) for b in boards_cls[split2:]])
yc_te = labels_cls[split2:]

# se llama clase MLP para clasificacion
clf = MLP()
clf.train_classifier_SGD(Xc_tr, yc_tr, epochs=50, lr=0.001, batch_size=32)
acc = evaluate_accuracy(clf, Xc_te, yc_te)
print(f"SGD classifier accuracy: {acc*100:.1f}%")

# matriz de confusion para analizar predicciones
tp=tn=fp=fn=0
for x,y in zip(Xc_te, yc_te):
    _,p = clf.predict_classifier(x)
    if p==1 and y==1: tp+=1
    elif p==0 and y==0: tn+=1
    elif p==1 and y==0: fp+=1
    else: fn+=1
print("Matriz de confusión:")
print(f"Actual=1 Pred=1:{tp} Pred=0:{fn}")
print(f"Actual=0 Pred=1:{fp} Pred=0:{tn}\n")

# test de agentes sobre tableros reales con evaluacion contra Minimax
test_boards = [generate_midgame_position() for _ in range(16)]
depth_opt = 7

# nombre de cada tecnica con su evaluador y optimizador
techs = {
    'Greedy':      ('Heurístico clásico', '–'),
    'Hill':        ('Heurístico clásico', '–'),
    'Genético':    ('Heurístico clásico', 'GA'),
    'CSP-Clásico': ('Heurístico clásico', '–'),
}
for opt in optimizers:
    techs[f"MLP-Greedy-{opt}"] = ('MLP', opt)
    techs[f"MCTS-{opt}"]       = ('MLP', opt)
    techs[f"CSP-MLP-{opt}"]    = ('MLP', opt)

# inicializar contadores
matches = {k:0 for k in techs}
deltas  = {k:[] for k in techs}
timesd  = {k:[] for k in techs}

for b in test_boards:
    heur0 = b.heuristic_evaluation()
    optm,_,_ = minimax_decision(b, 'black', depth=depth_opt)
    val_opt = optm.heuristic_evaluation()

    # funcion auxiliar para registrar resultados
    def rec_move(name, mv, elapsed):
        if mv and mv.board == optm.board:
            matches[name] += 1
        deltas[name].append(safe_eval(mv, heur0) - val_opt)
        timesd[name].append(elapsed)

    # CSP recorder 
    rec_move_csp = rec_move

    # Greedy, Hill y Genetico ejecuciones
    t0 = time.perf_counter(); rec_move('Greedy',   greedy_choice(b,'black'),      time.perf_counter()-t0)
    t0 = time.perf_counter(); rec_move('Hill',     hill_climbing(b,'black'),       time.perf_counter()-t0)
    t0 = time.perf_counter(); rec_move('Genético', genetic_board_search(),     time.perf_counter()-t0)

    # CSP clasico
    csp_solver.mlp_model = None
    t0 = time.perf_counter()
    # se pasa board como CSP resolverá el primer movimiento
    mv = csp_solver.solve_csp(Board(b.board))  
    rec_move_csp('CSP-Clásico', mv, time.perf_counter()-t0)

    # MLP-Greedy
    for opt, mdl in optimizers.items():
        key = f"MLP-Greedy-{opt}"
        t0 = time.perf_counter(); rec_move(key, mlp_choice(b,'black',mdl), time.perf_counter()-t0)

    # MCTS
    for opt, mdl in optimizers.items():
        key = f"MCTS-{opt}"
        reset_counters()
        t0 = time.perf_counter()
        mv2,_,_ = mcts_choice(b,'black',simulations=100, mlp=mdl) 
        rec_move(key, mv2, time.perf_counter()-t0)

    # CSP guiado por MLP
    for opt, mdl in optimizers.items():
        key = f"CSP-MLP-{opt}"
        csp_solver.mlp_model = mdl
        t0 = time.perf_counter()
        mv3 = csp_solver.solve_csp(Board(b.board))
        rec_move(key, mv3, time.perf_counter()-t0)

# impresion de resultados resumen
print("\n=== Resultados globales vs Minimax d=7 ===")
print("|Técnica|Eval.|Optim.|Match%|Δeval|Time|")
N = len(test_boards)
for name, (ev, opt) in techs.items():
    m = matches[name] / N * 100
    d = statistics.mean(deltas[name])
    t = statistics.mean(timesd[name])
    print(f"{name:<15}|{ev:<17}|{opt:<6}|{m:5.1f}%|{d:6.2f}|{t:6.3f}")
