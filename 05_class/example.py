import numpy as np 
import matplotlib.pyplot as plt 

A = np.zeros([400,400])

def drawline(x1, y1, x2, y2):
    # Garantir que os pontos fiquem dentro da imagem
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if abs(x1 - x2) >= abs(y1 - y2):
        # Linha mais "horizontal"
        if x1 > x2:  # garantir que o loop vá no sentido correto
            x1, y1, x2, y2 = x2, y2, x1, y1
        dydx = (y2 - y1) / (x2 - x1)
        for x in range(x1, x2 + 1):
            y = int(dydx * (x - x1) + y1)
            if 0 <= x < A.shape[0] and 0 <= y < A.shape[1]:
                A[y, x] = 255
    else:
        # Linha mais "vertical"
        if y1 > y2:  # garantir que o loop vá no sentido correto
            x1, y1, x2, y2 = x2, y2, x1, y1
        dxdy = (x2 - x1) / (y2 - y1)
        for y in range(y1, y2 + 1):
            x = int(dxdy * (y - y1) + x1)
            if 0 <= x < A.shape[0] and 0 <= y < A.shape[1]:
                A[y, x] = 255

# Desenhar uma linha
drawline(100, 120, 200, 150)

# Mostrar imagem
plt.imshow(A, cmap="gray")
plt.show()