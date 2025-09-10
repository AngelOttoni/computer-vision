import numpy as np
import matplotlib.pyplot as plt
import os

image_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'example_images', 'man4.png')

A = plt.imread(image_path)
A = np.dot(A[..., :3], [0.2989, 0.5870, 0.1140])  # Conversão para escala de cinza
A = A.astype(float)*255
l, c = A.shape

t1,t2=np.pi/2,np.pi/2
luz = np.array([np.sin(t1) * np.cos(t2), np.sin(t1) * np.sin(t2), np.cos(t1)], dtype=float)
o = np.array([0, 0, 1], dtype=float)
D = np.zeros((l, c))
Amb, Dif, Esp = np.copy(D),np.copy(D),np.copy(D)

N = np.zeros((l,c,3))
L = np.zeros((l,c,3))
O = np.zeros((l,c,3))
D = np.zeros((l,c,3))

for i in range(4, l - 4):
    for j in range(4, c - 4):
        if A[i, j] > 1:
           u=np.array([i+5,j,A[i+5,j]]) - np.array([i-5,j,A[i-5,j]])
           v=np.array([i,j+5,A[i,j+5]]) - np.array([i,j-5,A[i,j-5]])
           n = np.cross(u, v)
           N[i,j] = n / np.linalg.norm(n)
           L[i,j] = luz
           O[i,j] = o 
           
def Atualiza(luz):
    L = luz
    R = 2 * N * np.sum(L * N, axis=-1, keepdims=True) - L           
    R = np.divide(R, np.linalg.norm(R, axis=-1, keepdims=True), out=np.zeros_like(R), where=R > 0)
        
    Amb = 1
    Dif = np.sum(L * N, axis=-1, keepdims=True)
    Esp = np.sum(R * O, axis=-1, keepdims=True)**101
        
    # Composição final (ponderada)
        
    D = 0.2 * Amb + 0.4 * Dif  + 0.4 * Esp
    return D

D=Atualiza(luz)

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(D, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
ax.set_title('Clique X para mover a Luz')

lx, ly = 10,10
circ = ax.scatter([lx], [ly], s=200, facecolors='none', edgecolors='w',
                  linewidths=2, zorder=3, marker='o')

def on_key(event):
    global luz, t1, t2
    k = event.key

    t1=0.87
    if k == 'x': t2 += 0.1
    elif k == 'X': t2 -= 0.1

    # Atualiza vetor de luz (unitário pela parametrização esférica)
    luz[:] = [np.sin(t1)*np.cos(t2),  np.sin(t1)*np.sin(t2),  np.cos(t1)]

    # Recalcula imagem e atualiza SEM recriar imshow
    E = Atualiza(luz)              # deve retornar array 2D do mesmo shape
    im.set_data(E)

    circ.set_offsets([[np.int16(l/2*(1+luz[1])),np.int16(c/2*(1+ luz[0]))]])
    circ.set_facecolors('white')
    circ.set_sizes([100+luz[2]*100])
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
