import random
import copy

# Clase para representar el tablero de damas
class Board:
    def __init__(self, board=None):
        # Si se pasa un board hacemos una copia si no se crea tablero vacio
        if board:
            self.board = copy.deepcopy(board)
        else:
            # Inicializar casillas '.' en casillas claras, '-' en oscuras
            self.board = [
                ['.' if (i + j) % 2 == 0 else '-' for j in range(8)]
                for i in range(8)
            ]
            # colocar piezas en pos inicial
            self.init_default_position()

    # funcion para inicializar la pos por defecto
    def init_default_position(self):
        # Colocar peones blancos b en filas 0-2 indices 0 a 2 en casillas oscuras
        for i in range(3):
            for j in range(8):
                if (i + j) % 2 == 1:
                    self.board[i][j] = 'b'
        # Colocar peones negros n en filas 5-7 indices 5 a 7 en casillas oscuras
        for i in range(5, 8):
            for j in range(8):
                if (i + j) % 2 == 1:
                    self.board[i][j] = 'n'

    # coronar peones que alcanzan la fila opuesta.
    def crown_pieces(self):
        # Negros alcanzan fila 0
        for j in range(8):
            if self.board[0][j] == 'n':
                self.board[0][j] = 'N'
        # Blancos alcanzan fila 7
        for j in range(8):
            if self.board[7][j] == 'b':
                self.board[7][j] = 'B'

    # clonar el mismo tablero
    def clone(self):
        return Board(self.board)

    # representacion en string filas con coordenadas y labels de columnas
    def __str__(self):
        lines = []
        for i in range(8):
            row = f"{8 - i}  " + ' '.join(self.board[i])
            lines.append(row)
        lines.append("   a b c d e f g h")
        return '\n'.join(lines)

    ## Determina si la casilla es oscura, solo en esas se pueden colocar piezas
    def is_dark_square(self, i, j):
        return (i + j) % 2 == 1

    # verifica que el tablero cumpla las reglas para ser tablero valido
    def validate(self):
        black_pawns = white_pawns = black_kings = white_kings = 0
        for i in range(8):
            for j in range(8):
                cell = self.board[i][j]
                # Si hay pieza debe ser casilla oscura
                if cell != '.' and not self.is_dark_square(i, j):
                    return False
                 # Contar piezas por tipo
                if cell == 'n':
                    black_pawns += 1
                elif cell == 'N':
                    black_kings += 1
                elif cell == 'b':
                    white_pawns += 1
                elif cell == 'B':
                    white_kings += 1
        # maximo 12 peones por bando
        if black_pawns > 12 or white_pawns > 12:
            return False
        return True

    # contar todas las piezas en el tablero sin diferenciar tipo
    def count_pieces(self):
        return sum(
            1
            for i in range(8)
            for j in range(8)
            if self.board[i][j] in ['n', 'N', 'b', 'B']
        )

    @staticmethod
    def generate_random_valid():
        # crear tablero vacio para colocar piezas aleatorias validas
        board = Board()
        # Limpiar todas las piezas
        board.board = [
            ['.' if (i + j) % 2 == 0 else '-' for j in range(8)]
            for i in range(8)
        ]
        # lista de coordenadas de casillas oscuras
        positions = [
            (i, j)
            for i in range(8)
            for j in range(8)
            if board.is_dark_square(i, j)
        ]
        # aletoriamente alterar posiciones
        random.shuffle(positions)
        num_black = random.randint(1, 6)
        num_white = random.randint(1, 6)
        # colocar piezas en las primeras posiciones disponibles
        pieces = ['n'] * num_black + ['b'] * num_white
        for p, (i, j) in zip(pieces, positions):
            board.board[i][j] = p
        return board

    @staticmethod
    # intentar hasta maximo de veces crear tablero invalido
    def generate_random_invalid(max_attempts=1000):
        for _ in range(max_attempts):
            b = Board.generate_random_valid()
             # alterar casilla aleatoria para invalidar configuracion
            i, j = random.choice(
                [(x, y) for x in range(8) for y in range(8)]
            )
            b.board[i][j] = random.choice(['n', 'N', 'b', 'B'])
            if not b.validate():
                return b
        # si no se encuentra devolver caso obviamente invalido
        b = Board()
        b.board[0][0] = 'n'
        return b

    # calcular diferencia de valor entre piezas negras y blancas
    def heuristic_evaluation(self):
        black_pawns = white_pawns = black_kings = white_kings = 0
        for i in range(8):
            for j in range(8):
                cell = self.board[i][j]
                if cell == 'n':
                    black_pawns += 1
                elif cell == 'N':
                    black_kings += 1
                elif cell == 'b':
                    white_pawns += 1
                elif cell == 'B':
                    white_kings += 1
        return (black_pawns + 1.5 * black_kings) - (
            white_pawns + 1.5 * white_kings
        )
        
    # contar movimientos legales disponibles para jugador dado
    def mobility(self, player):
        from search import get_all_moves
        moves = get_all_moves(self, player)
        return len(moves)
