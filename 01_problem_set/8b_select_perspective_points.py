import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

image_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'example_images', 'entrada.png')

img = mpimg.imread(image_path)

plt.imshow(img)
plt.title("Use o cursor para identificar as coordenadas (x, y)")
plt.show()
