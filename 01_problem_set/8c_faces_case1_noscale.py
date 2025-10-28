import os
import cv2 as cv

def load_gray_equalized(path):
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    g = cv.equalizeHist(g)
    return img, g

def match_single_scale(scene_gray, template_gray, method=cv.TM_CCOEFF_NORMED):
    res = cv.matchTemplate(scene_gray, template_gray, method)
    _, max_val, _, max_loc = cv.minMaxLoc(res)
    h, w = template_gray.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left, bottom_right, float(max_val)

def draw_box(image_bgr, top_left, bottom_right, color=(0, 255, 0), thickness=2):
    out = image_bgr.copy()
    cv.rectangle(out, top_left, bottom_right, color, thickness)
    return out

if __name__ == '__main__':
    # Pasta das imagens (relativa ao script)
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'utils', 'example_images')

    # Arquivos
    pessoa1_path = os.path.join(base_dir, 'pessoa1.png')
    gente1_path  = os.path.join(base_dir, 'gente1.png')

    # Carregar + equalizar
    pessoa1_bgr, pessoa1_gray = load_gray_equalized(pessoa1_path)
    gente1_bgr,  gente1_gray  = load_gray_equalized(gente1_path)

    # Caso 1: sem mudança de escala (pessoa1 em gente1)
    tl1, br1, score1 = match_single_scale(gente1_gray, pessoa1_gray)
    print('[Caso 1] top_left:', tl1, 'bottom_right:', br1, "score:", round(score1, 4))
    vis1 = draw_box(gente1_bgr, tl1, br1)
    cv.imwrite('out_case1_gente1.png', vis1)
