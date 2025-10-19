import cv2 as cv
import numpy as np
import os

image_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'example_images', 'entrada.png')


original_image = cv.imread(image_path)

if original_image is None:
    raise FileNotFoundError(f"Não foi possível carregar a imagem: {image_path}")

# --- Pontos de origem ---
# Ajuste conforme necessário observando os cantos do conjunto de quadros na imagem original
# usando o script 8b_select_perspective_points.py
pts_src = np.float32([
    [62, 159],    # canto superior esquerdo
    [490, 4],     # canto superior direito
    [486, 690],   # canto inferior direito
    [64, 517]     # canto inferior esquerdo
])

# --- Pontos de destino (imagem corrigida, 900x600) ---
width, height = 900, 600
pts_dst = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])

# Cálcula a matriz de transformação perspectiva
M = cv.getPerspectiveTransform(pts_src, pts_dst)

# Aplica a transformação
output = cv.warpPerspective(original_image, M, (width, height))

# Salva o resultado
output_path = os.path.join(os.path.dirname(image_path), 'saida.png')
cv.imwrite(output_path, output)

# Exibi a matriz e mensagem final
print("Matriz de transformação (3x3):")
print(M)
print(f"\nImagem transformada salva em: {output_path}")
