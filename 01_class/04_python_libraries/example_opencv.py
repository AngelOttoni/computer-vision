import cv2
import matplotlib.pyplot as plt
import os

image_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'example_images', 'serro.bmp')

img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 4. Deteco de bordas com Canny ( funo poderosa do OpenCV)
edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_bgr) # ERRADO em termos de cor: fica azulada
plt.subplot(1, 3, 2)
plt.imshow(img_rgb) # Correto
plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.show()