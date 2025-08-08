import numpy as np
import matplotlib.pyplot as plt

# Criar uma imagem RGB 4x4 (altura x largura x canais)
# Cada pixel tem 3 valores: [R, G, B], de 0 a 255
A = np.array([
[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
[[0, 255, 255], [255, 0, 255], [128, 128, 128], [0, 0, 0]],
[[255, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128]],
[[64, 64, 64], [192, 192, 192], [255, 128, 0], [128, 0, 255]]
], dtype=np.uint8)

print(A[:,:,0])
print(A[:,:,1])
print(A[:,:,2])

# Mostrar imagem simulada
plt.imshow(A)
plt.title("Imagem RGB Simulada (4x4 pixels)")
plt.axis('off') # Remover eixos
plt.show()