import numpy as np

# Vizinhança 5x5 (linha a linha)
Znb = np.array([
    [10, 11, 12, 12, 13],
    [10, 12, 13, 14, 15],
    [11, 13, 14, 16, 16],
    [13, 14, 17, 17, 18],
    [15, 17, 18, 18, 19]
], dtype=float)

# Centro (i,j) = (2,2) zero-index (valor 14 no enunciado)
i = j = 2

# 1) Derivadas por diferenças centrais
Zx = (Znb[i, j+1] - Znb[i, j-1]) / 2.0
Zy = (Znb[i+1, j] - Znb[i-1, j]) / 2.0

# 2) Tangentes e normal unitária
Tx = np.array([1.0, 0.0, Zx])
Ty = np.array([0.0, 1.0, Zy])
N  = np.cross(Tx, Ty)
N  = N / np.linalg.norm(N)

# 3) Iluminação Phong
L = np.array([0.0, -1.0,  1.0])   # luz
O = np.array([1.0,  0.0,  0.0])   # observador

L_hat = L / np.linalg.norm(L)
O_hat = O / np.linalg.norm(O)

Id = max(0.0, float(np.dot(N, L_hat)))
R  = 2 * Id * N - L_hat
alpha = 16
Is = max(0.0, float(np.dot(R, O_hat))) ** alpha

print(f"Zx = {Zx:.3f}, Zy = {Zy:.3f}")
print(f"N = ({N[0]:.6f}, {N[1]:.6f}, {N[2]:.6f})")
print(f"Id = {Id:.6f}, Is = {Is:.6f}")
