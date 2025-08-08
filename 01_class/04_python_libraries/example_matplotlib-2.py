import matplotlib.pyplot as plt

A = plt.imread('../example_images/serro.bmp')
plt.imshow(A)
plt.title("Imagem exibida com plt.imread()")
plt.axis('on')
plt.show()

print(A.shape)
print(A[90:100,90:100,0])

A=A.copy()
A[100:150,100:150,:]=0
plt.imshow(A)