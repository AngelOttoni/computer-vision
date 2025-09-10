import numpy as np
import matplotlib.pyplot as plt
import os

image_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'example_images', 'man4.png')

A = plt.imread(image_path)
A = np.dot(A[..., :3], [0.2989, 0.5870, 0.1140]) # Conversão para escala de cinza
A = A.astype(float) * 255
l, c = A.shape

luz = np.array([0, 1, 1], dtype=float)
luz = luz / np.linalg.norm(luz)
o = np.array([0, 0, 1], dtype=float)
D = np.zeros((l, c))
Amb, Dif, Esp = np.copy(D), np.copy(D), np.copy(D)

for i in range(4, l - 4):
    for j in range(4, c - 4):
        if A[i, j] > 1:
            u = np.array([i + 5, j, A[i + 5, j]])
            -np.array([i - 5, j, A[i - 5, j]])
            v = np.array([i, j + 5, A[i, j + 5]])
            -np.array([i, j - 5, A[i, j - 5]])
            n = np.cross(u, v)
            n = n / np.linalg.norm(n)
            # Reflexo especular
            R = 2 * n * np.dot(n, luz) - luz
            R = R / np.linalg.norm(R)
            # Componentes de iluminao
            Amb[i, j] = 1
            Dif[i, j] = max(0, np.dot(luz, n))
            Esp[i, j] = max(0, np.dot(R, o)) ** 11
            # Composio final (ponderada)
            D[i, j] = 0.2 * Amb[i, j]
            +0.4 * Dif[i, j]
            +0.4 * Esp[i, j]

# Exibir imagens lado a lado
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].imshow(A, cmap='gray')
axs[0].set_title('Original (Gray)')
axs[1].imshow(Dif, cmap='gray')
axs[1].set_title('Difusa')
axs[2].imshow(Esp, cmap='gray')
axs[2].set_title('Especular')
axs[3].imshow(D, cmap='gray')
axs[3].set_title(' Iluminao Phong')
plt.show()
