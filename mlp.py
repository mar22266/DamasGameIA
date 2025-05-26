import random
import math

# convertir cada casilla del tablero en valor numerico segun pieza
def board_to_features(board):
    mapping = {'.': 0.0, '-': 0.0, 'n': 1.0, 'N': 1.5, 'b': -1.0, 'B': -1.5}
    features = [mapping[board.board[i][j]] for i in range(8) for j in range(8)]
    
    # Nuevo: features booleanos
    has_piece_on_light = 0
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0 and board.board[i][j] in ['n', 'N', 'b', 'B']:
                has_piece_on_light = 1
    # Más de 12 piezas por bando
    black = sum(board.board[i][j] in ['n', 'N'] for i in range(8) for j in range(8))
    white = sum(board.board[i][j] in ['b', 'B'] for i in range(8) for j in range(8))
    more_than_12_black = 1 if black > 12 else 0
    more_than_12_white = 1 if white > 12 else 0
    # Peón en fila 0 o 7
    pawn_on_first_or_last = 0
    for j in range(8):
        if board.board[0][j] in ['n', 'b'] or board.board[7][j] in ['n', 'b']:
            pawn_on_first_or_last = 1
    # Menos de 2 piezas totales
    few_pieces = 1 if (black + white) < 2 else 0
    
    features += [has_piece_on_light, more_than_12_black, more_than_12_white, pawn_on_first_or_last, few_pieces]
    return features

# clase MLP 
class MLP:
    # constructor de la red neuronal
    def __init__(self, input_size=69, hidden_sizes=[128, 64]):
        # semilla para reproducibilidad
        random.seed(0)
        # definir tamanos de cada capa incluyendo salida
        layer_sizes = [input_size] + hidden_sizes + [1]
        # matrices de pesos
        self.W = []  
        # vectores de bias
        self.b = []  
        for idx in range(len(layer_sizes)-1):
            in_size = layer_sizes[idx]
            out_size = layer_sizes[idx+1]
            # inicializacion Xavier para pesos
            limit = math.sqrt(6/(in_size+out_size))
            W = [[random.uniform(-limit, limit) for _ in range(out_size)] for _ in range(in_size)]
            bias = [0.0]*out_size
            self.W.append(W)
            self.b.append(bias)
        # buffers para SGD con momentum
        self.vW = [ [[0.0]*len(self.W[l][0]) for _ in self.W[l]] for l in range(len(self.W)) ]
        self.vb = [ [0.0]*len(self.b[l]) for l in range(len(self.b)) ]
        # buffers para Adam
        self.mW = [ [[0.0]*len(self.W[l][0]) for _ in self.W[l]] for l in range(len(self.W)) ]
        self.vW_adam = [ [[0.0]*len(self.W[l][0]) for _ in self.W[l]] for l in range(len(self.W)) ]
        self.mb = [ [0.0]*len(self.b[l]) for l in range(len(self.b)) ]
        self.vb_adam = [ [0.0]*len(self.b[l]) for l in range(len(self.b)) ]

    # funcion de propagacion hacia adelante
    def _forward(self, x):
        activations = [x]
        for l in range(len(self.W)):
            prev = activations[-1]
            # lista para almacenar sumas
            z = []
            for j in range(len(self.b[l])):
                s = self.b[l][j] + sum(prev[i]*self.W[l][i][j] for i in range(len(prev)))
                z.append(s)
            if l < len(self.W)-1:
                # funcion de activacion ReLU para capas ocultas
                a = [zi if zi>0 else 0.0 for zi in z]
            else:
                # no se aplica ReLU en la capa de salida
                a = z
            # almacenar activaciones
            activations.append(a)
        return activations

    # calcula las activaciones hacia adelante
    def predict(self, x):
        acts = self._forward(x)
        out = acts[-1][0]
        hidden = acts[-2]
        # retorna salida y ultima capa oculta
        return out, hidden

    # calcular error cuadratico medio sobre dataset
    def _compute_loss(self, X, Y):
        return sum((self.predict(x)[0]-y)**2 for x,y in zip(X,Y))/len(X)

    # funcion de entrenamiento SGD con entrada X, Y, epochs, lr, batch_size y momentum
    def train_SGD(self, X, Y, epochs=50, lr=0.01, batch_size=32, momentum=0.9):
        n = len(X)
        for _ in range(epochs):
            # shuffle indices para cada epoca
            idxs = list(range(n)); random.shuffle(idxs)
            for start in range(0, n, batch_size):
                batch = idxs[start:start+batch_size]
                # inicializar acumuladores de gradientes
                gradW = [ [[0.0]*len(self.W[l][0]) for _ in self.W[l]] for l in range(len(self.W)) ]
                gradb = [ [0.0]*len(self.b[l]) for l in range(len(self.b)) ]
                for idx in batch:
                    x, y = X[idx], Y[idx]
                    acts = self._forward(x)
                    err = acts[-1][0] - y
                    # gradiente inicial
                    delta = [err]
                    for l in reversed(range(len(self.W))):
                        prev = acts[l]
                        # acumular gradientes para pesos
                        for i in range(len(prev)):
                            for j in range(len(delta)):
                                gradW[l][i][j] += delta[j] * prev[i]
                        for j in range(len(delta)):
                            gradb[l][j] += delta[j]
                        # calcular delta para capa anterior si no es la entrada
                        if l > 0:
                            new_delta = []
                            for i in range(len(self.W[l])):
                                s = sum(self.W[l][i][j] * delta[j] for j in range(len(delta)))
                                new_delta.append(s if acts[l][i] > 0 else 0.0)
                            delta = new_delta
                bs = len(batch)
                # actualizar pesos y bias con SGD y momentum
                for l in range(len(self.W)):
                    for i in range(len(self.W[l])):
                        for j in range(len(self.W[l][i])):
                            g = gradW[l][i][j] / bs
                            self.vW[l][i][j] = momentum*self.vW[l][i][j] + lr*g
                            self.W[l][i][j] -= self.vW[l][i][j]
                    for j in range(len(self.b[l])):
                        gb = gradb[l][j] / bs
                        self.vb[l][j] = momentum*self.vb[l][j] + lr*gb
                        self.b[l][j] -= self.vb[l][j]
        # retorna perdida final tras entrenamiento
        return self._compute_loss(X,Y)

    # funcion de entrenamiento Adam con entrada X, Y, epochs, lr, beta1, beta2, eps y batch_size
    def train_Adam(self, X, Y, epochs=50, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, batch_size=32):
        # n es tamano del dataset t es contador global de pasos
        n, t = len(X), 0
        for _ in range(epochs):
            idxs = list(range(n)); random.shuffle(idxs)
            for start in range(0, n, batch_size):
                batch = idxs[start:start+batch_size]
                # inicializar gradientes acumulados para pesos y bias
                gradW = [ [[0.0]*len(self.W[l][0]) for _ in self.W[l]] for l in range(len(self.W)) ]
                gradb = [ [0.0]*len(self.b[l]) for l in range(len(self.b)) ]
                for idx in batch:
                    t += 1
                    x, y = X[idx], Y[idx]
                    acts = self._forward(x)
                    delta = [acts[-1][0] - y]
                    # backpropagation para todas las capas
                    for l in reversed(range(len(self.W))):
                        prev = acts[l]
                        for i in range(len(prev)):
                            for j in range(len(delta)):
                                gradW[l][i][j] += delta[j] * prev[i]
                        for j in range(len(delta)):
                            gradb[l][j] += delta[j]
                        if l > 0:
                            new_delta = []
                            for i in range(len(self.W[l])):
                                # propagar error hacia atras
                                s = sum(self.W[l][i][j] * delta[j] for j in range(len(delta)))
                                # aplicar derivada de ReLU
                                new_delta.append(s if acts[l][i] > 0 else 0.0)
                            delta = new_delta
                bs = len(batch)
                # actualizar pesos y bias usando Adam
                for l in range(len(self.W)):
                    for i in range(len(self.W[l])):
                        for j in range(len(self.W[l][i])):
                            # promedio del gradiente
                            g = gradW[l][i][j] / bs
                            self.mW[l][i][j] = beta1*self.mW[l][i][j] + (1-beta1)*g
                            self.vW_adam[l][i][j] = beta2*self.vW_adam[l][i][j] + (1-beta2)*(g**2)
                            m_hat = self.mW[l][i][j] / (1-beta1**t)
                            v_hat = self.vW_adam[l][i][j] / (1-beta2**t)
                            # actualizacion final del peso
                            self.W[l][i][j] -= lr * m_hat / (math.sqrt(v_hat)+eps)
                    for j in range(len(self.b[l])):
                        gb = gradb[l][j] / bs
                        self.mb[l][j] = beta1*self.mb[l][j] + (1-beta1)*gb
                        self.vb_adam[l][j] = beta2*self.vb_adam[l][j] + (1-beta2)*(gb**2)
                        m_hat = self.mb[l][j] / (1-beta1**t)
                        v_hat = self.vb_adam[l][j] / (1-beta2**t)
                        self.b[l][j] -= lr * m_hat / (math.sqrt(v_hat)+eps)
        # retorna la perdida total al final
        return self._compute_loss(X,Y)

    # entrenamiento de clasificador binario con SGD y funcion sigmoide
    def train_classifier_SGD(self, X, Y, epochs=50, lr=0.01, batch_size=32):
        # funcion sigmoide
        def sigmoid(z): return 1/(1+math.exp(-z))
        n = len(X)
        for _ in range(epochs):
            idxs=list(range(n)); random.shuffle(idxs)
            for start in range(0,n,batch_size):
                for idx in idxs[start:start+batch_size]:
                    x, y = X[idx], Y[idx]
                    acts = self._forward(x)
                    # prediccion probabilistica
                    p = sigmoid(acts[-1][0])
                    err = p - y
                    delta=[err]
                    # backpropagation para ajustar pesos y bias
                    for l in reversed(range(len(self.W))):
                        prev = acts[l]
                        for i in range(len(prev)):
                            for j in range(len(delta)):
                                # actualizacion directa
                                self.W[l][i][j] -= lr * delta[j] * prev[i]
                        for j in range(len(delta)):
                            self.b[l][j] -= lr * delta[j]
                        if l>0:
                            new_delta=[]
                            for i in range(len(self.W[l])):
                                s = sum(self.W[l][i][j]*delta[j] for j in range(len(delta)))
                                new_delta.append(s if acts[l][i]>0 else 0.0)
                            delta=new_delta

    # prediccion binaria usando la red entrenada como clasificador
    def predict_classifier(self, x):
        out,_ = self.predict(x)
        # aplicar sigmoide
        p = 1/(1+math.exp(-out))
        # retorna probabilidad y clase
        return p, (1 if p>=0.5 else 0)
