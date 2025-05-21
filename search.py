import math
import random
import time
from board import Board
from mlp import board_to_features

# retorna la evaluacion heuristica del tablero actual
def heuristic_evaluation(board):
    return board.heuristic_evaluation()

# selecciona el mejor movimiento posible segun la heuristica clasica
def greedy_choice(board, player):
    moves = get_all_moves(board, player)
    if not moves:
        return None
    # se maximiza si el jugador es negro minimizamos si es blanco
    return max(moves, key=heuristic_evaluation) if player == 'black' else min(moves, key=heuristic_evaluation)


# selecciona el mejor movimiento basado en el modelo MLP entrenado
def mlp_choice(board, player, mlp_model):
    moves = get_all_moves(board, player)
    if not moves:
        return None
    def score(b):
        return mlp_model.predict(board_to_features(b))[0]
    return max(moves, key=score) if player == 'black' else min(moves, key=score)

# aplica busqueda local hill climbing para mejorar una configuracion de tablero
def hill_climbing(board, player, max_steps=50):
    current = board
    current_player = player
    for _ in range(max_steps):
        # mejor movimiento posible
        move = greedy_choice(current, current_player)
        if not move or heuristic_evaluation(move) <= heuristic_evaluation(current):
            break
        current = move
        current_player = 'white' if current_player == 'black' else 'black'
    # retornar mejor alcanzado
    return current

# realiza una busqueda genetica para encontrar buenas configuraciones de tablero
def genetic_board_search(
    pop_size=30,
    generations=50,
    elitism_rate=0.2,
    tournament_size=3,
    mutation_rate=0.1,
    no_improve_limit=10
):
    # poblacion inicial aleatoria
    population = [Board.generate_random_valid() for _ in range(pop_size)]
    elite_k = max(1, int(elitism_rate * pop_size))

    def fitness(b):
        # Combina diferencia de piezas y movilidad
        h = abs(b.heuristic_evaluation())
        m = b.mobility('black') + b.mobility('white')
        max_moves = 20 
        return h - 0.5 * (m / max_moves)

    # Inicializar mejor puntuacion
    best_score = float('inf')
    no_improve = 0

    for gen in range(1, generations+1):
        # Ordenar poblacion
        scored = sorted(population, key=fitness)
        current_best = fitness(scored[0])
        # Early stopping
        if current_best < best_score:
            best_score, no_improve = current_best, 0
        else:
            no_improve += 1
            if no_improve >= no_improve_limit:
                break
        # Elitismo
        new_pop = scored[:elite_k]
        # tasa de mutacion decreciente
        mr = mutation_rate * (1 - gen/generations)
        # Cruce y mutacion
        while len(new_pop) < pop_size:
            # seleccion por torneo
            p1 = min(random.sample(scored, tournament_size), key=fitness)
            p2 = min(random.sample(scored, tournament_size), key=fitness)
            # cruce uniforme entre padres
            child = p1.clone()
            for i in range(8):
                for j in range(8):
                    if child.is_dark_square(i,j) and random.random()<0.5:
                        child.board[i][j] = p2.board[i][j]
            # aplicar mutacion
            num_mut = int(64 * mr)
            for _ in range(num_mut):
                i,j = random.choice([(x,y) for x in range(8) for y in range(8) if child.is_dark_square(x,y)])
                child.board[i][j] = random.choice(['n','b','-'])
            # refinamiento con hill-climbing local
            child = hill_climbing(child, 'black', max_steps=2)
            new_pop.append(child)
        population = new_pop
    # retornar mejor individuo encontrado
    return min(population, key=fitness)


# contador global de nodos explorados por minimax
minimax_nodes = 0

# implementacion de minimax con poda alfa-beta y conteo de nodos
def minimax(board, player, depth, alpha=-math.inf, beta=math.inf):
    global minimax_nodes
    minimax_nodes += 1
    if depth == 0 or game_over(board):
        return heuristic_evaluation(board)
    opp = 'white' if player == 'black' else 'black'
    moves = get_all_moves(board, player)
    if not moves:
        return -1000 if player == 'black' else 1000
    best = -math.inf if player == 'black' else math.inf
    for m in moves:
        # llamada recursiva para el siguiente nivel
        val = minimax(m, opp, depth - 1, alpha, beta)
        if player == 'black':
            best = max(best, val)
            alpha = max(alpha, best)
        else:
            best = min(best, val)
            beta = min(beta, best)
        if alpha >= beta:
            break
    return best

# realiza una jugada usando minimax y devuelve tiempo y nodos explorados
def minimax_decision(board, player, depth=3):
    global minimax_nodes
    minimax_nodes = 0
    # medir tiempo de ejecucion
    start = time.perf_counter()
    best_move = None
    best_val = -math.inf if player == 'black' else math.inf
    for m in get_all_moves(board, player):
        val = minimax(m, 'white' if player == 'black' else 'black', depth - 1)
        if (player == 'black' and val > best_val) or (player == 'white' and val < best_val):
            best_val, best_move = val, m
    elapsed = time.perf_counter() - start
    return best_move, minimax_nodes, elapsed

# contador global de nodos para MCTS
mcts_nodes = 0

# simulacion aleatoria de una partida a partir de un tablero
def simulate_random_game(board, player):
    global mcts_nodes
    current = player
    # limite de turnos
    for _ in range(50):
        moves = get_all_moves(board, current)
        if not moves:
            return 'white' if current=='black' else 'black'
        board = random.choice(moves)
        mcts_nodes += 1
        current = 'white' if current=='black' else 'black'
    # retornar ganador segun estado final
    return get_winner(board)

# selecciona el mejor movimiento usando MCTS y MLP 
def mcts_choice(board, player, simulations=100, mlp=None):
    global mcts_nodes
    mcts_nodes = 0
    start = time.perf_counter()
    
    best_move, best_score = None, float('-inf')
    opponent = 'white' if player == 'black' else 'black'
    
    for m in get_all_moves(board, player):
        wins = draws = 0
        
        # evaluacion previa con MLP si se proporciona
        if mlp:
            features = board_to_features(m)
            prior_score, _ = mlp.predict(features)  
        else:
            prior_score = 0.0
        
        # simulaciones aleatorias desde este movimiento
        for _ in range(simulations):
            result = simulate_random_game(m.clone(), opponent)
            if result == player:
                wins += 1
            elif result == 'draw':
                draws += 1
        
        win_rate = (wins + 0.5 * draws) / simulations
        
        # Combina la salida de la MLP y la tasa de victoria
        combined_score = 0.5 * prior_score + 0.5 * win_rate
        
        if combined_score > best_score:
            best_score, best_move = combined_score, m

    # retornar mejor movimiento encontrado junto con nodos y tiempo
    elapsed = time.perf_counter() - start
    return best_move, mcts_nodes, elapsed

# retorna todos los movimientos posibles para el jugador actual
def get_all_moves(board, player):
    moves = []
    capture_sequences = []
    own_pieces = ['n','N'] if player == 'black' else ['b','B']
    opp_pieces = ['b','B'] if player == 'black' else ['n','N']
    pawn_dirs = [(1,-1),(1,1)] if player == 'black' else [(-1,-1),(-1,1)]

    # busqueda de capturas posibles
    for i in range(8):
        for j in range(8):
            cell = board.board[i][j]
            if cell not in own_pieces:
                continue
            # captura con peon
            if cell.lower() == own_pieces[0]:
                for di, dj in pawn_dirs:
                    ni, nj = i+di, j+dj
                    li, lj = i+2*di, j+2*dj
                    if 0 <= ni < 8 and 0 <= nj < 8 and 0 <= li < 8 and 0 <= lj < 8:
                        if board.board[ni][nj] in opp_pieces and board.board[li][lj] == '-':
                            nb = board.clone()
                            nb.board[i][j] = nb.board[ni][nj] = '-'
                            nb.board[li][lj] = cell
                            caps = get_multi_captures(nb, li, lj, player, 'pawn')
                            capture_sequences.extend(caps or [nb])
            # captura con dama
            if cell == own_pieces[1]:
                for di, dj in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    enemy = None
                    for step in range(1, 8):
                        ni, nj = i+step*di, j+step*dj
                        if not (0 <= ni < 8 and 0 <= nj < 8):
                            break
                        if board.board[ni][nj] in own_pieces:
                            break
                        if board.board[ni][nj] in opp_pieces:
                            enemy = (ni, nj)
                            break
                    if enemy:
                        ei, ej = enemy
                        li, lj = ei+di, ej+dj
                        if 0 <= li < 8 and 0 <= lj < 8 and board.board[li][lj] == '-':
                            nb = board.clone()
                            nb.board[i][j] = nb.board[ei][ej] = '-'
                            nb.board[li][lj] = cell
                            caps = get_multi_captures(nb, li, lj, player, 'king')
                            capture_sequences.extend(caps or [nb])
    # si hay capturas, solo se permiten esas
    if capture_sequences:
        return capture_sequences

    # movimientos simples si no hay capturas
    for i in range(8):
        for j in range(8):
            cell = board.board[i][j]
            if cell not in own_pieces:
                continue
            # peon
            if cell.lower() == own_pieces[0]:
                for di, dj in pawn_dirs:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < 8 and 0 <= nj < 8 and board.board[ni][nj] == '-':
                        nb = board.clone()
                        nb.board[i][j] = '-'
                        nb.board[ni][nj] = cell
                        nb.crown_pieces()
                        moves.append(nb)
            # dama
            else:
                for di, dj in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    for step in range(1, 8):
                        ni, nj = i+step*di, j+step*dj
                        if not (0 <= ni < 8 and 0 <= nj < 8):
                            break
                        if board.board[ni][nj] != '-':
                            break
                        nb = board.clone()
                        nb.board[i][j] = '-'
                        nb.board[ni][nj] = cell
                        moves.append(nb)
    return moves

# busca capturas encadenadas desde una posicion 
def get_multi_captures(board, i, j, player, piece_type):
    result = []
    moved = False
    piece = board.board[i][j]
    opp_pieces = ['b','B'] if player == 'black' else ['n','N']
    pawn_dirs = [(1,-1),(1,1)] if player == 'black' else [(-1,-1),(-1,1)]
    if piece_type == 'pawn':
        for di, dj in pawn_dirs:
            ni, nj = i+di, j+dj
            li, lj = i+2*di, j+2*dj
            if 0 <= ni < 8 and 0 <= nj < 8 and 0 <= li < 8 and 0 <= lj < 8:
                if board.board[ni][nj] in opp_pieces and board.board[li][lj] == '-':
                    moved = True
                    nb = board.clone()
                    nb.board[i][j] = nb.board[ni][nj] = '-'
                    nb.board[li][lj] = piece
                    caps = get_multi_captures(nb, li, lj, player, 'pawn')
                    result.extend(caps or [nb])
    # captura con dama
    else:
        for di, dj in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            for step in range(1, 8):
                ni, nj = i+step*di, j+step*dj
                if not (0 <= ni < 8 and 0 <= nj < 8):
                    break
                if board.board[ni][nj] in opp_pieces:
                    li, lj = ni+di, nj+dj
                    if 0 <= li < 8 and 0 <= lj < 8 and board.board[li][lj] == '-':
                        moved = True
                        nb = board.clone()
                        nb.board[i][j] = nb.board[ni][nj] = '-'
                        nb.board[li][lj] = piece
                        caps = get_multi_captures(nb, li, lj, player, 'king')
                        result.extend(caps or [nb])
                    break
                if board.board[ni][nj] != '-':
                    break
    return result if moved else None

# retorna True si el juego ya termino (un jugador no tiene piezas)
def game_over(board):
    black = sum(cell in ['n','N'] for row in board.board for cell in row)
    white = sum(cell in ['b','B'] for row in board.board for cell in row)
    return black == 0 or white == 0

# determina quien gano segun cantidad de piezas
def get_winner(board):
    black = sum(cell in ['n','N'] for row in board.board for cell in row)
    white = sum(cell in ['b','B'] for row in board.board for cell in row)
    return 'black' if black > white else 'white' if white > black else 'draw'
