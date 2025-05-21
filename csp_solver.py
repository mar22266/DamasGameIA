from board import Board
from search import get_all_moves
from mlp import board_to_features

# modelo MLP a usar para evaluar tableros 
mlp_model = None

# resuelve el CSP selecciona el mejor movimiento a jugar desde un tablero dado
def solve_csp(board: Board):
    moves = get_all_moves(board, 'black')
    if not moves:
        return None
    
    # devolver la primera jugada valida
    if mlp_model is None:
        return moves[0]

    # puntuar cada tablero resultante
    best = None
    best_score = float('-inf')
    for m in moves:
        # aplicar la jugada sin modificar el tablero original
        b2 = m.clone()  
        features = board_to_features(b2)
        score, _ = mlp_model.predict(features)
        if score > best_score:
            best_score = score
            best = m
    # retornar movimiento con mayor score
    return best
