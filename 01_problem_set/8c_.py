import os
import cv2 as cv
import numpy as np

def load_gray_equalized(path):
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    g = cv.equalizeHist(g)
    return img, g

def match_single_scale(scene_gray, template_gray, method=cv.TM_CCOEFF_NORMED):
    res = cv.matchTemplate(scene_gray, template_gray, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    h, w = template_gray.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left, bottom_right, float(max_val)

def match_multi_scale(scene_gray, template_gray,
                      method=cv.TM_CCOEFF_NORMED,
                      scales=np.linspace(0.2, 2.0, 37)):
    best = {'score': -1.0, 'scale': None, 'top_left': None, 'bottom_right': None}
    th, tw = template_gray.shape[:2]

    for s in scales:
        new_w = int(tw * s)
        new_h = int(th * s)
        if new_w < 10 or new_h < 10:
            continue
        # área/qualidade de interpolação adequada
        interp = cv.INTER_AREA if s < 1.0 else cv.INTER_CUBIC
        tpl = cv.resize(template_gray, (new_w, new_h), interpolation=interp)
        if tpl.shape[0] > scene_gray.shape[0] or tpl.shape[1] > scene_gray.shape[1]:
            continue
        res = cv.matchTemplate(scene_gray, tpl, method)
        _, max_val, _, max_loc = cv.minMaxLoc(res)
        if max_val > best['score']:
            best['score'] = float(max_val)
            best['scale'] = float(s)
            best['top_left'] = max_loc
            best['bottom_right'] = (max_loc[0] + tpl.shape[1], max_loc[1] + tpl.shape[0])
    return best

def draw_box(image_bgr, top_left, bottom_right, color=(0, 255, 0), thickness=2):
    out = image_bgr.copy()
    cv.rectangle(out, top_left, bottom_right, color, thickness)
    return out

if __name__ == '__main__':
    # Pasta das imagens (relativa ao script)
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'utils', 'example_images')

    # Arquivos
    pessoa1_path = os.path.join(base_dir, 'pessoa1.png')
    pessoa2_path = os.path.join(base_dir, 'pessoa2.png')
    gente1_path  = os.path.join(base_dir, 'gente1.png')
    gente2_path  = os.path.join(base_dir, 'gente2.png')

    # Carregar + equalizar
    pessoa1_bgr, pessoa1_gray = load_gray_equalized(pessoa1_path)
    pessoa2_bgr, pessoa2_gray = load_gray_equalized(pessoa2_path)
    gente1_bgr,  gente1_gray  = load_gray_equalized(gente1_path)
    gente2_bgr,  gente2_gray  = load_gray_equalized(gente2_path)

    # Caso 1: sem mudança de escala (pessoa1 em gente1)
    tl1, br1, score1 = match_single_scale(gente1_gray, pessoa1_gray)
    print('[Caso 1] top_left:', tl1, 'bottom_right:', br1, "score:", round(score1, 4))
    vis1 = draw_box(gente1_bgr, tl1, br1)
    cv.imwrite('match_case1_gente1.png', vis1)

    # Caso 2: com mudança de escala (pessoa2 em gente2)
    best = match_multi_scale(gente2_gray, pessoa2_gray)
    print("[Caso 2] top_left:", best['top_left'],
          "bottom_right:", best["bottom_right"],
          "scale:", round(best["scale"], 4),
          "score:", round(best["score"], 4))
    vis2 = draw_box(gente2_bgr, best['top_left'], best['bottom_right'])
    cv.imwrite('match_case2_gente2.png', vis2)
