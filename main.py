import argparse
import random
import time
import statistics
from statistics import mean, pstdev
from board import Board
from mlp import MLP, board_to_features
from search import *
import csp_solver
from metrics import *

def generate_midgame_position(min_pieces=8, max_pieces=20):
    while True:
        b = Board.generate_random_valid()
        if min_pieces <= b.count_pieces() <= max_pieces and get_all_moves(b, 'black'):
            return b

def generate_varied_invalid():
    cases = []

    # Pieza en casilla clara
    def case_piece_on_light():
        b = Board.generate_random_valid()
        i, j = random.choice([(i, j) for i in range(8) for j in range(8) if (i + j) % 2 == 0])
        b.board[i][j] = random.choice(['n', 'b'])
        return b if not b.validate() else None

    #  Más de 12 piezas negras
    def case_more_than_12_black():
        b = Board.generate_random_valid()
        blacks = 0
        for i in range(8):
            for j in range(8):
                if b.is_dark_square(i, j) and b.board[i][j] == '-':
                    b.board[i][j] = 'n'
                    blacks += 1
                if blacks >= 13:
                    break
            if blacks >= 13:
                break
        return b if not b.validate() else None

    # Peón en primera fila
    def case_pawn_first_row():
        b = Board.generate_random_valid()
        b.board[0][random.choice([j for j in range(8) if (0 + j) % 2 == 1])] = 'b'
        return b if not b.validate() else None

    #  Muy pocas piezas
    def case_few_pieces():
        b = Board()
        b.board[1][1] = 'n'
        return b if not b.validate() else None

    # Demasiadas damas negras
    def case_many_black_kings():
        b = Board.generate_random_valid()
        black_kings = 0
        for i in range(8):
            for j in range(8):
                if b.is_dark_square(i,j) and black_kings < 6:
                    b.board[i][j] = 'N'
                    black_kings += 1
        return b if not b.validate() else None

    #  Ninguna pieza (vacío)
    def case_empty_board():
        b = Board()
        return b if not b.validate() else None

    #  Muchas piezas de ambos tipos pero menos de 12 por bando
    def case_many_pieces_balanced():
        b = Board()
        count = 0
        for i in range(8):
            for j in range(8):
                if b.is_dark_square(i,j):
                    b.board[i][j] = 'n' if count % 2 == 0 else 'b'
                    count += 1
        return b if not b.validate() else None

    #  Doble pieza en casilla (simulación de corrupción)
    def case_double_piece():
        b = Board()
        b.board[1][1] = 'n'
        b.board[1][1] = 'b' 
        return b if not b.validate() else None

    cases = [case_piece_on_light, case_more_than_12_black, case_pawn_first_row, case_few_pieces,
             case_many_black_kings, case_empty_board, case_many_pieces_balanced, case_double_piece]

    # Intenta hasta encontrar un caso  inválido 
    for _ in range(10):
        f = random.choice(cases)
        b = f()
        if b is not None:
            return b

    return case_piece_on_light()


def safe_eval(move, default):
    return move.heuristic_evaluation() if move else default

def normalize(X, means, stds):
    n_features = len(X[0])
    return [[(x[i] - means[i]) / stds[i] for i in range(n_features)] for x in X]

def run_regression():
    print("--- Curva de aprendizaje MLP (regresión) ---")
    boards_reg = [generate_midgame_position() for _ in range(1000)]
    labels_reg = [b.heuristic_evaluation() for b in boards_reg]
    split = int(0.8 * len(boards_reg))
    Xr_train = [board_to_features(b) for b in boards_reg[:split]]
    yr_train = labels_reg[:split]
    Xr_test  = [board_to_features(b) for b in boards_reg[split:]]
    yr_test  = labels_reg[split:]
    n_features = len(Xr_train[0])
    means = [mean([x[i] for x in Xr_train]) for i in range(n_features)]
    stds  = [pstdev([x[i] for x in Xr_train]) or 1.0 for i in range(n_features)]
    Xr_train = normalize(Xr_train, means, stds)
    Xr_test  = normalize(Xr_test, means, stds)
    for frac in [0.1,0.25,0.5,0.75,1.0]:
        n = int(len(Xr_train) * frac)
        X_sub, y_sub = Xr_train[:n], yr_train[:n]
        mlp_sgd = MLP(input_size=n_features); mlp_sgd.train_SGD(X_sub, y_sub, epochs=20, lr=0.01, batch_size=32)
        mse_sgd_tr = mean_squared_error(mlp_sgd, X_sub, y_sub)
        mse_sgd_te = mean_squared_error(mlp_sgd, Xr_test, yr_test)
        mlp_adam = MLP(input_size=n_features); mlp_adam.train_Adam(X_sub, y_sub, epochs=20, lr=0.001, batch_size=32)
        mse_adam_tr = mean_squared_error(mlp_adam, X_sub, y_sub)
        mse_adam_te = mean_squared_error(mlp_adam, Xr_test, yr_test)
        print(f"Train={n:4d} | SGD tr={mse_sgd_tr:.4f}, te={mse_sgd_te:.4f} | "
              f"Adam tr={mse_adam_tr:.4f}, te={mse_adam_te:.4f}")
    print()
    print("--- Entrenamiento final MLP regresor ---")
    mlp_sgd = MLP(input_size=n_features)
    t0 = time.perf_counter()
    loss_sgd = mlp_sgd.train_SGD(Xr_train, yr_train, epochs=100, lr=0.001, batch_size=32)
    t_s = time.perf_counter() - t0
    mse_s = mean_squared_error(mlp_sgd, Xr_test, yr_test)
    print(f"SGD  -> Loss={loss_sgd:.4f}, Test MSE={mse_s:.4f}, Time={t_s:.2f}s")
    mlp_adam = MLP(input_size=n_features)
    t0 = time.perf_counter()
    loss_a = mlp_adam.train_Adam(Xr_train, yr_train, epochs=100, lr=0.001, batch_size=32)
    t_a = time.perf_counter() - t0
    mse_a = mean_squared_error(mlp_adam, Xr_test, yr_test)
    print(f"Adam -> Loss={loss_a:.4f}, Test MSE={mse_a:.4f}, Time={t_a:.2f}s\n")

def run_classifier():
    print("--- Entrenando clasificador inválidos (SGD) ---")
    boards_cls = [generate_midgame_position() for _ in range(500)] + [generate_varied_invalid() for _ in range(500)]
    labels_cls = [1]*500 + [0]*500
    z = list(zip(boards_cls, labels_cls)); random.shuffle(z)
    boards_cls, labels_cls = zip(*z)
    split2 = int(0.8 * len(boards_cls))
    Xc_tr = [board_to_features(b) for b in boards_cls[:split2]]
    yc_tr = labels_cls[:split2]
    Xc_te = [board_to_features(b) for b in boards_cls[split2:]]
    yc_te = labels_cls[split2:]
    n_features = len(Xc_tr[0])
    means = [mean([x[i] for x in Xc_tr]) for i in range(n_features)]
    stds  = [pstdev([x[i] for x in Xc_tr]) or 1.0 for i in range(n_features)]
    Xc_tr = normalize(Xc_tr, means, stds)
    Xc_te = normalize(Xc_te, means, stds)
    clf = MLP(input_size=n_features)
    clf.train_classifier_SGD(Xc_tr, yc_tr, epochs=50, lr=0.001, batch_size=32)
    acc = evaluate_classifier_accuracy(clf, Xc_te, yc_te)
    print(f"SGD classifier accuracy: {acc*100:.1f}%")
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

def run_agent_tests():
    print("--- Test de agentes sobre tableros reales con evaluacion contra Minimax ---")
    test_boards = [generate_midgame_position() for _ in range(16)]
    depth_opt = 7
    techs = {
        'Greedy':      ('Heurístico clásico', '–'),
        'Hill':        ('Heurístico clásico', '–'),
        'Genético':    ('Heurístico clásico', 'GA'),
        'CSP-Clásico': ('Heurístico clásico', '–'),
    }
    # Prepare optimizers
    boards_reg = [generate_midgame_position() for _ in range(1000)]
    Xr_train = [board_to_features(b) for b in boards_reg]
    n_features = len(Xr_train[0])
    mlp_sgd = MLP(input_size=n_features)
    mlp_sgd.train_SGD(Xr_train, [b.heuristic_evaluation() for b in boards_reg], epochs=20, lr=0.01, batch_size=32)
    mlp_adam = MLP(input_size=n_features)
    mlp_adam.train_Adam(Xr_train, [b.heuristic_evaluation() for b in boards_reg], epochs=20, lr=0.001, batch_size=32)
    optimizers = {'SGD': mlp_sgd, 'Adam': mlp_adam}
    for opt in optimizers:
        techs[f"MLP-Greedy-{opt}"] = ('MLP', opt)
        techs[f"MCTS-{opt}"]       = ('MLP', opt)
        techs[f"CSP-MLP-{opt}"]    = ('MLP', opt)
    matches = {k:0 for k in techs}
    deltas  = {k:[] for k in techs}
    timesd  = {k:[] for k in techs}
    for b in test_boards:
        heur0 = b.heuristic_evaluation()
        optm,_,_ = minimax_decision(b, 'black', depth=depth_opt)
        val_opt = optm.heuristic_evaluation()
        def rec_move(name, mv, elapsed):
            if mv and mv.board == optm.board:
                matches[name] += 1
            deltas[name].append(safe_eval(mv, heur0) - val_opt)
            timesd[name].append(elapsed)
        rec_move_csp = rec_move
        t0 = time.perf_counter(); rec_move('Greedy',   greedy_choice(b,'black'),      time.perf_counter()-t0)
        t0 = time.perf_counter(); rec_move('Hill',     hill_climbing(b,'black'),       time.perf_counter()-t0)
        t0 = time.perf_counter(); rec_move('Genético', genetic_board_search(),     time.perf_counter()-t0)
        csp_solver.mlp_model = None
        t0 = time.perf_counter()
        mv = csp_solver.solve_csp(Board(b.board))  
        rec_move_csp('CSP-Clásico', mv, time.perf_counter()-t0)
        for opt, mdl in optimizers.items():
            key = f"MLP-Greedy-{opt}"
            t0 = time.perf_counter(); rec_move(key, mlp_choice(b,'black',mdl), time.perf_counter()-t0)
        for opt, mdl in optimizers.items():
            key = f"MCTS-{opt}"
            reset_counters()
            t0 = time.perf_counter()
            mv2,_,_ = mcts_choice(b,'black',simulations=100, mlp=mdl) 
            rec_move(key, mv2, time.perf_counter()-t0)
        for opt, mdl in optimizers.items():
            key = f"CSP-MLP-{opt}"
            csp_solver.mlp_model = mdl
            t0 = time.perf_counter()
            mv3 = csp_solver.solve_csp(Board(b.board))
            rec_move(key, mv3, time.perf_counter()-t0)
    print("\n=== Resultados globales vs Minimax d=7 ===")
    print("|Técnica|Eval.|Optim.|Match%|Δeval|Time|")
    N = len(test_boards)
    for name, (ev, opt) in techs.items():
        m = matches[name] / N * 100
        d = statistics.mean(deltas[name])
        t = statistics.mean(timesd[name])
        print(f"{name:<15}|{ev:<17}|{opt:<6}|{m:5.1f}%|{d:6.2f}|{t:6.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DamasGameIA runner")
    parser.add_argument('--mode', choices=['regression', 'classifier', 'agents'], required=True)
    args = parser.parse_args()
    random.seed(42)
    if args.mode == 'regression':
        run_regression()
    elif args.mode == 'classifier':
        run_classifier()
    elif args.mode == 'agents':
        run_agent_tests()
