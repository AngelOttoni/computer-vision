import cv2 as cv, numpy as np, os, glob
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

ROOT = "images"
CLASSES, K = ["cars", "people", "tree"], 100

train_paths, y_train = [], []  # Caminhos de treino
for ci, c in enumerate(CLASSES):
    ps = sorted(glob.glob(os.path.join(ROOT, "train", c, "*")))
    train_paths.extend(ps)
    y_train.extend([ci] * len(ps))
y_train = np.array(y_train, int)

test_paths, y_test = [], []  # Caminhos de teste
for ci, c in enumerate(CLASSES):
    ps = sorted(glob.glob(os.path.join(ROOT, "test", c, "*")))
    test_paths.extend(ps)
    y_test.extend([ci] * len(ps))
y_test = np.array(y_test, int)

sift = cv.SIFT_create()
img_des_train = []  # Descritores de treino
for p in train_paths:
    img = cv.imread(p, cv.IMREAD_GRAYSCALE)
    kps, des = sift.detectAndCompute(img, None)
    img_des_train.append(des)  # lista por imagem

# vocabulrio visual (k-means)
des_all = np.vstack(img_des_train)  # lista nica
kmeans = KMeans(n_clusters=K, n_init=10, max_iter=300)
kmeans.fit(des_all)

X_train = np.empty((0, K))  # BoVW imagem treino
for des in img_des_train:
    words = kmeans.predict(des)
    h = np.bincount(words, minlength=K)
    h = h.astype(np.float32)
    h /= (np.linalg.norm(h) + 1e-8)
    X_train = np.vstack([X_train, h])

clf = LinearSVC(C=1.0, dual=False)  # Classificador
clf.fit(X_train, y_train)

X_test = np.empty((0, K))  # BoVW imagem de teste
for p in test_paths:
    img = cv.imread(p, cv.IMREAD_GRAYSCALE)
    kps, des = sift.detectAndCompute(img, None)
    words = kmeans.predict(des)
    h = np.bincount(words, minlength=K)
    h = h.astype(np.float32)
    h /= (np.linalg.norm(h) + 1e-8)
    X_test = np.vstack([X_test, h])

y_pred = clf.predict(X_test)
print(y_test)
print(y_pred)