# Versão minimalista do código original (aula j Slide 288), mantendo o mesmo estilo e lógica, 
# mas adicionando a varredura de (K), cálculo de acurácia, plot e matriz de confusão

import cv2 as cv, numpy as np, os, glob
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

ROOT = "images"
CLASSES = ["cars", "people", "tree"]
K_values = list(range(10, 201, 10))  # 10,20,...,200

# monta caminhos e rótulos
train_paths, y_train = [], []
for ci, c in enumerate(CLASSES):
    ps = sorted(glob.glob(os.path.join(ROOT, "train", c, "*")))
    train_paths.extend(ps)
    y_train.extend([ci] * len(ps))
y_train = np.array(y_train, int)

test_paths, y_test = [], []
for ci, c in enumerate(CLASSES):
    ps = sorted(glob.glob(os.path.join(ROOT, "test", c, "*")))
    test_paths.extend(ps)
    y_test.extend([ci] * len(ps))
y_test = np.array(y_test, int)

# extrai SIFT
sift = cv.SIFT_create()
img_des_train = []
for p in train_paths:
    img = cv.imread(p, cv.IMREAD_GRAYSCALE)
    kps, des = sift.detectAndCompute(img, None)
    if des is None:
        des = np.zeros((0, 128), dtype=np.float32)
    img_des_train.append(des)

# empilha todos os descritores não-vazios
des_list = [d for d in img_des_train if d is not None and d.shape[0] > 0]
if len(des_list) == 0:
    raise SystemExit("Nenhum descritor SIFT encontrado nas imagens de treino.")
des_all = np.vstack(des_list)

results = []  # (K, accuracy, y_pred, best_clf, best_kmeans)

for K in K_values:
    print(f"Rodando K = {K} ...")
    kmeans = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=0)
    kmeans.fit(des_all)

    # monta X_train (BoVW)
    X_train = np.empty((0, K), dtype=np.float32)
    for des in img_des_train:
        if des is None or des.shape[0] == 0:
            h = np.zeros(K, dtype=np.float32)
        else:
            words = kmeans.predict(des)
            h = np.bincount(words, minlength=K).astype(np.float32)
            norm = np.linalg.norm(h)
            if norm > 0:
                h /= (norm + 1e-8)
        X_train = np.vstack([X_train, h])

    clf = LinearSVC(C=1.0, dual=False, max_iter=10000)
    clf.fit(X_train, y_train)

    # monta X_test e prediz
    X_test = np.empty((0, K), dtype=np.float32)
    for p in test_paths:
        img = cv.imread(p, cv.IMREAD_GRAYSCALE)
        kps, des = sift.detectAndCompute(img, None)
        if des is None or des.shape[0] == 0:
            h = np.zeros(K, dtype=np.float32)
        else:
            words = kmeans.predict(des)
            h = np.bincount(words, minlength=K).astype(np.float32)
            norm = np.linalg.norm(h)
            if norm > 0:
                h /= (norm + 1e-8)
        X_test = np.vstack([X_test, h])

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Acurácia (K={K}): {acc:.4f}")
    results.append((K, acc, y_pred, clf, kmeans))

# extrai resultados e plota acurácia vs K
Ks = [r[0] for r in results]
accs = [r[1] for r in results]

plt.figure(figsize=(8,4))
plt.plot(Ks, accs, marker='o')
plt.title("Acurácia vs K")
plt.xlabel("K (número de clusters)")
plt.ylabel("Acurácia")
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_K.png")
plt.show()
print("Gráfico salvo em: accuracy_vs_K.png")

# melhor K -> matriz de confusão
best_idx = int(np.argmax(accs))
best_K, best_acc, best_y_pred, best_clf, best_kmeans = results[best_idx]
print(f"\nMelhor K = {best_K} com acurácia = {best_acc:.4f}")

cm = confusion_matrix(y_test, best_y_pred)
print("Matriz de confusão (linhas=verdadeiro, colunas=previsto):")
print(cm)

# opcional: salvar matriz como figura
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest', aspect='auto')
plt.title(f"Matriz de Confusão (K={best_K})")
plt.xlabel("Previsto")
plt.ylabel("Verdadeiro")
plt.xticks(range(len(CLASSES)), CLASSES, rotation=45)
plt.yticks(range(len(CLASSES)), CLASSES)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, int(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(f"confusion_matrix_K{best_K}.png")
plt.show()
print(f"Matriz de confusão salva em: confusion_matrix_K{best_K}.png")
