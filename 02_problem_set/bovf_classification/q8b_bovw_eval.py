import os, glob, csv, uuid, shutil
from datetime import datetime
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ========== CONFIG ===========

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # pasta onde está este script
ROOT = os.path.join(BASE_DIR, "data", "images_resized")
CLASSES = ["cars", "people", "tree"]
K_values = list(range(10, 201, 10))  # 10,20,...,200
RANDOM_STATE = 42
SAMPLE_DESCRIPTORS = None  # None -> sem amostragem (usa todos os descritores)
OUT_DIR = os.path.join(BASE_DIR, "bovw_results")   # pasta onde figuras e relatórios serão salvos
# =============================

os.makedirs(OUT_DIR, exist_ok=True)

# gerar run id (timestamp + short uuid) para evitar sobrescrita
def make_run_id():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{short}"

RUN_ID = make_run_id()

# função extra para garantir nomes únicos caso um arquivo já exista
def unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}_v{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

def gather_paths(root, classes, split):
    paths, ys = [], []
    for ci, c in enumerate(classes):
        ps = sorted(glob.glob(os.path.join(root, split, c, "*")))
        paths.extend(ps)
        ys.extend([ci] * len(ps))
    return paths, np.array(ys, dtype=int)

train_paths, y_train = gather_paths(ROOT, CLASSES, "train")
test_paths, y_test   = gather_paths(ROOT, CLASSES, "test")

# checagem rápida: cada classe deve ter 20 train e 20 test (conforme enunciado)
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
    # KMeans (full batch)
    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10, max_iter=300)
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

# salvar acurácia vs K em CSV
csv_name = f"accuracy_vs_K_{RUN_ID}.csv"
csv_path = os.path.join(OUT_DIR, csv_name)
csv_path = unique_path(csv_path)
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["K", "accuracy"])
    for k, a in zip(Ks, accs):
        writer.writerow([k, f"{a:.6f}"])
print(f"Acurácia por K salva em: {csv_path}")

# plot acurácia vs K (com run id no nome)
plt.figure(figsize=(8,5))
plt.plot(Ks, accs, marker='o')
plt.title("Acurácia vs K (número de clusters)")
plt.xlabel("K (número de clusters)")
plt.ylabel("Acurácia")
plt.grid(True)
plt.tight_layout()
acc_png = os.path.join(OUT_DIR, f"accuracy_vs_K_{RUN_ID}.png")
acc_png = unique_path(acc_png)
plt.savefig(acc_png)
print(f"Gráfico salvo em: {acc_png}")
plt.close()

# melhor K
best_idx = int(np.argmax(accs))
best_K, best_acc, best_y_pred, best_clf, best_kmeans = results[best_idx]
print(f"\nMelhor K = {best_K} com acurácia = {best_acc:.4f}")

# salvar best_K em txt
best_txt = os.path.join(OUT_DIR, f"best_K_{RUN_ID}.txt")
best_txt = unique_path(best_txt)
with open(best_txt, "w") as f:
    f.write(f"RUN_ID: {RUN_ID}\n")
    f.write(f"best_K: {best_K}\n")
    f.write(f"accuracy: {best_acc:.6f}\n")
print(f"Melhor K salvo em: {best_txt}")

# matriz de confusão e relatório
cm = confusion_matrix(y_test, best_y_pred)
report = classification_report(y_test, best_y_pred, target_names=CLASSES, digits=4)
print("Matriz de confusão:\n", cm)
print("\nRelatório de classificação:\n", report)

# salvar relatório em txt
report_txt = os.path.join(OUT_DIR, f"classification_report_K{best_K}_{RUN_ID}.txt")
report_txt = unique_path(report_txt)
with open(report_txt, "w") as f:
    f.write(f"RUN_ID: {RUN_ID}\n")
    f.write(f"best_K: {best_K}\n")
    f.write(f"accuracy: {best_acc:.6f}\n\n")
    f.write("confusion_matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nclassification_report:\n")
    f.write(report)
print(f"Relatório de classificação salvo em: {report_txt}")

# -----------------------
# salvar e copiar imagens mal classificadas
# -----------------------
mis_idx = np.where(y_test != best_y_pred)[0]
mis_count = len(mis_idx)
print(f"\nTotal de imagens mal classificadas: {mis_count}")

# CSV com detalhes dos mal classificados
mis_csv = os.path.join(OUT_DIR, f"misclassified_K{best_K}_{RUN_ID}.csv")
mis_csv = unique_path(mis_csv)
with open(mis_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "filepath", "true_label", "pred_label"])
    for i in mis_idx:
        fp = test_paths[i]
        true = CLASSES[int(y_test[i])]
        pred = CLASSES[int(best_y_pred[i])]
        writer.writerow([i, fp, true, pred])
print(f"CSV de mal classificados salvo em: {mis_csv}")

# copiar imagens mal classificadas para pasta separada
mis_dir = os.path.join(OUT_DIR, f"misclassified_K{best_K}_{RUN_ID}")
mis_dir = unique_path(mis_dir)
os.makedirs(mis_dir, exist_ok=True)

for i in mis_idx:
    src = test_paths[i]
    true = CLASSES[int(y_test[i])]
    pred = CLASSES[int(best_y_pred[i])]
    base = os.path.basename(src)
    dst_name = f"{i:03d}_true-{true}_pred-{pred}_{base}"
    dst = os.path.join(mis_dir, dst_name)
    try:
        shutil.copy(src, dst)
    except Exception as e:
        print(f"Falha ao copiar {src} -> {dst}: {e}")

print(f"Imagens mal classificadas copiadas para: {mis_dir}")

# plot matriz de confusão (com run id)
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
cm_png = os.path.join(OUT_DIR, f"confusion_matrix_K{best_K}_{RUN_ID}.png")
cm_png = unique_path(cm_png)
plt.savefig(cm_png)
print(f"Matriz de confusão salva em: {cm_png}")
plt.close()

print("\nExecução concluída. Arquivos foram salvos em:", OUT_DIR)
print("RUN_ID:", RUN_ID)
