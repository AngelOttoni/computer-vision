import os, glob
import numpy as np
import cv2 as cv
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ========== CONFIG ===========
ROOT = "02_problem_set/bovf_classification/images_resized"
CLASSES = ["cars", "people", "tree"]
K_values = list(range(10, 201, 10))  # 10,20,...,200
USE_MINIBATCH = True
RANDOM_STATE = 42
SAMPLE_DESCRIPTORS = None  # None -> sem amostragem (usa todos os descritores)
# =============================

def gather_paths(root, classes, split):
    paths, ys = [], []
    for ci, c in enumerate(classes):
        ps = sorted(glob.glob(os.path.join(root, split, c, "*")))
        paths.extend(ps)
        ys.extend([ci] * len(ps))
    return paths, np.array(ys, dtype=int)

train_paths, y_train = gather_paths(ROOT, CLASSES, "train")
test_paths, y_test   = gather_paths(ROOT, CLASSES, "test")

# checagem rápida: cada classe deve ter 20 train e 20 test
ok = True
for c in CLASSES:
    nt = len(glob.glob(os.path.join(ROOT, "train", c, "*")))
    ne = len(glob.glob(os.path.join(ROOT, "test", c, "*")))
    if nt != 20 or ne != 20:
        print(f"[AVISO] classe '{c}' tem {nt} train e {ne} test (esperado 20/20).")
        ok = False
if not ok:
    raise SystemExit("Organize as imagens conforme solicitado e rode novamente.")

sift = cv.SIFT_create()

# extrai descritores SIFT do treino
img_des_train = []
print("Extraindo SIFT das imagens de treino...")
for p in train_paths:
    img = cv.imread(p, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler {p}")
    kps, des = sift.detectAndCompute(img, None)
    if des is None:
        des = np.zeros((0, 128), dtype=np.float32)
    img_des_train.append(des)

# empilha todos os descritores não-vazios
des_list = [d for d in img_des_train if d is not None and d.shape[0] > 0]
if len(des_list) == 0:
    raise SystemExit("Nenhum descritor SIFT encontrado nas imagens de treino.")
des_all = np.vstack(des_list)
print(f"Total descritores SIFT no treino: {des_all.shape[0]}")

# decidir descritores para KMeans (amostragem opcional)
if SAMPLE_DESCRIPTORS is not None and des_all.shape[0] > SAMPLE_DESCRIPTORS:
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(des_all.shape[0], SAMPLE_DESCRIPTORS, replace=False)
    des_for_kmeans = des_all[idx]
    print(f"Amostrando {SAMPLE_DESCRIPTORS} descritores para KMeans.")
else:
    des_for_kmeans = des_all
    print("Usando TODOS os descritores para KMeans.")

results = []

def descriptors_to_bovw(des, kmeans, K):
    if des is None or des.shape[0] == 0:
        h = np.zeros(K, dtype=np.float32)
    else:
        words = kmeans.predict(des)
        h = np.bincount(words, minlength=K).astype(np.float32)
        norm = np.linalg.norm(h)
        if norm > 0:
            h /= (norm + 1e-8)
    return h

for K in K_values:
    print(f"\n=== K = {K} ===")
    # MiniBatchKMeans (rápido) — compatível com conjuntos de descritores maiores
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=RANDOM_STATE, batch_size=1000)
    print("Treinando KMeans...")
    kmeans.fit(des_for_kmeans)

    # construir X_train
    X_train = np.vstack([descriptors_to_bovw(des, kmeans, K) for des in img_des_train])
    # treinar SVM linear
    clf = LinearSVC(C=1.0, dual=False, max_iter=20000, random_state=RANDOM_STATE)
    print("Treinando LinearSVC...")
    clf.fit(X_train, y_train)

    # construir X_test
    X_test = []
    print("Processando imagens de teste...")
    for p in test_paths:
        img = cv.imread(p, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Não foi possível ler {p}")
        kps, des = sift.detectAndCompute(img, None)
        X_test.append(descriptors_to_bovw(des, kmeans, K))
    X_test = np.vstack(X_test)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia (K={K}): {acc:.4f}")

    results.append((K, acc, y_pred, clf, kmeans))

# coletar as acurácias
Ks = [r[0] for r in results]
accs = [r[1] for r in results]

# plot
plt.figure(figsize=(8,5))
plt.plot(Ks, accs, marker='o')
plt.title("Acurácia vs K (número de clusters)")
plt.xlabel("K (número de clusters)")
plt.ylabel("Acurácia")
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_K.png")
print("Gráfico salvo em: accuracy_vs_K.png")
plt.show()

# melhor K
best_idx = int(np.argmax(accs))
best_K, best_acc, best_y_pred, best_clf, best_kmeans = results[best_idx]
print(f"\nMelhor K = {best_K} com acurácia = {best_acc:.4f}")

# matriz de confusão e relatório
cm = confusion_matrix(y_test, best_y_pred)
print("Matriz de confusão:\n", cm)
print("\nRelatório de classificação:\n", classification_report(y_test, best_y_pred, target_names=CLASSES, digits=4))

# plot matriz de confusão
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', aspect='auto')
plt.title(f"Matriz de Confusão (K={best_K})")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.xticks(np.arange(len(CLASSES)), CLASSES, rotation=45)
plt.yticks(np.arange(len(CLASSES)), CLASSES)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, int(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(f"confusion_matrix_K{best_K}.png")
print(f"Matriz de confusão salva em: confusion_matrix_K{best_K}.png")
plt.show()
