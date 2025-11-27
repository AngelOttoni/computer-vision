import os
import cv2 as cv
import glob

ROOT_IN = '02_problem_set/bovf_classification/images'
ROOT_OUT = '02_problem_set/bovf_classification/images_resized'

TARGET_MAX_SIZE = 400  # lado maior

CLASSES = ['cars', 'people', 'tree']
SUBSETS = ['train', 'test']

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resize_keep_aspect(img, max_side):
    h, w = img.shape[:2]
    # já pequeno → não aumentar
    if max(h, w) <= max_side:
        return img
    # calcular escala
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
    return resized

for subset in SUBSETS:
    for cls in CLASSES:
        in_dir = os.path.join(ROOT_IN, subset, cls)
        out_dir = os.path.join(ROOT_OUT, subset, cls)
        ensure_dir(out_dir)

        paths = sorted(glob.glob(os.path.join(in_dir, '*')))
        print(f"Processando {subset}/{cls}: {len(paths)} imagens")

        for p in paths:
            img = cv.imread(p)
            if img is None:
                print("Falha ao ler:", p)
                continue

            resized = resize_keep_aspect(img, TARGET_MAX_SIZE)

            out_path = os.path.join(out_dir, os.path.basename(p))
            cv.imwrite(out_path, resized)

print("\nProcessamento concluído!")
print("Novas imagens salvas em: ", ROOT_OUT)
