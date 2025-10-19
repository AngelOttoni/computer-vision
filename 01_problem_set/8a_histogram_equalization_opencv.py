import cv2 as cv
import matplotlib.pyplot as plt 
import os

image_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'example_images', 'livro.png')

original_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if original_image is None:
    raise FileNotFoundError(f"Não foi possível carregar a imagem: {image_path}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.hist(original_image.ravel(), bins=256, range=[0,256])
plt.title('Histogram - Original Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


equalized_image = cv.equalizeHist(original_image)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.hist(equalized_image.ravel(), bins=256, range=[0,256])
plt.title('Histogram - Equalized Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
