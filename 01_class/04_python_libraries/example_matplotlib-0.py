import matplotlib.pyplot as plt
import numpy as np

imagem = np.array([
[ 0, 50, 100, 150, 200],
[ 20, 70, 120, 170, 220],
[ 40, 90, 140, 190, 240],
[ 60, 110, 160, 210, 255],
[ 80, 130, 180, 230, 250]
], dtype=np.uint8)


plt.imshow(imagem, cmap='gray')
plt.title("Imagem simulada")
plt.colorbar()
plt.show()