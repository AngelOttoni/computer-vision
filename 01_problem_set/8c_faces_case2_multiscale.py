import os
import cv2 as cv
import numpy as np

def draw_and_save(image_bgr, top_left, bottom_right, out_path):
    vis = image_bgr.copy()
    cv.rectangle(vis, top_left, bottom_right, (0, 255, 0), 2)
    cv.imwrite(out_path, vis)

def search_best(template, scene, mask=None, scales=None, angles=None):
    """
    Busca Wally (template) na cena com varredura de escalas e pequenas rotações.
    Usa TM_CCORR_NORMED (aceita máscara).
    """
    if scales is None: scales = np.linspace(0.2, 1.2, 41)     # 0.20x a 1.20x
    if angles is None: angles = [-10, -5, 0, 5, 10]           # pequenas rotações

    th, tw = template.shape[:2]
    best = {'score': -1.0, 'scale': None, 'angle': None, 'top_left': None, 'bottom_right': None}

    for ang in angles:
        M = cv.getRotationMatrix2D((tw // 2, th // 2), ang, 1.0)
        t_rot = cv.warpAffine(template, M, (tw, th), flags=cv.INTER_LINEAR, borderValue=0)
        m_rot = cv.warpAffine(mask,     M, (tw, th), flags=cv.INTER_NEAREST, borderValue=0) if mask is not None else None

        for s in scales:
            w = int(tw * s); h = int(th * s)
            if w < 20 or h < 20:
                continue
            t_rs = cv.resize(t_rot, (w, h), interpolation=cv.INTER_AREA if s < 1 else cv.INTER_CUBIC)
            m_rs = cv.resize(m_rot, (w, h), interpolation=cv.INTER_NEAREST) if m_rot is not None else None

            if h > scene.shape[0] or w > scene.shape[1]:
                continue

            res = cv.matchTemplate(scene, t_rs, cv.TM_CCORR_NORMED, mask=m_rs)
            _, max_val, _, max_loc = cv.minMaxLoc(res)

            if max_val > best['score']:
                best = {
                    'score': float(max_val),
                    'scale': float(s),
                    'angle': float(ang),
                    'top_left': max_loc,
                    'bottom_right': (max_loc[0] + w, max_loc[1] + h),
                }
    return best

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'utils', 'example_images')
    pessoa_path = os.path.join(base_dir, 'pessoa2.png')  # Wally recortado (fundo preto)
    gente_path  = os.path.join(base_dir, 'gente2.png')   # cena do Waldo

    tpl_bgr = cv.imread(pessoa_path, cv.IMREAD_COLOR)
    scn_bgr = cv.imread(gente_path,  cv.IMREAD_COLOR)
    if tpl_bgr is None or scn_bgr is None:
        raise FileNotFoundError("Verifique os caminhos das imagens.")

    # grayscale + equalização suave
    tpl_gray = cv.cvtColor(tpl_bgr, cv.COLOR_BGR2GRAY)
    scn_gray = cv.cvtColor(scn_bgr, cv.COLOR_BGR2GRAY)
    tpl_gray = cv.equalizeHist(cv.GaussianBlur(tpl_gray, (3, 3), 0))
    scn_gray = cv.equalizeHist(cv.GaussianBlur(scn_gray, (3, 3), 0))

    # máscara: ignora fundo preto do recorte do Wally
    mask = (tpl_gray > 8).astype(np.uint8) * 255

    # também busca no canal de bordas e escolhe o melhor
    tpl_edges = cv.Canny(tpl_gray, 60, 180)
    scn_edges = cv.Canny(scn_gray, 60, 180)
    mask_edges = (mask & ((tpl_edges > 0).astype(np.uint8) * 255))

    best_gray = search_best(tpl_gray,  scn_gray,  mask=mask)
    best_edge = search_best(tpl_edges, scn_edges, mask=mask_edges)

    best = best_gray if best_gray['score'] >= best_edge['score'] else best_edge
    print({
        'top_left': best['top_left'],
        'bottom_right': best['bottom_right'],
        'scale': round(best['scale'], 4),
        'angle': round(best['angle'], 2),
        'score': round(best['score'], 6),
        'channel': 'gray' if best is best_gray else 'edges'
    })

    out_file = 'out_case2_gente2.png'
    draw_and_save(scn_bgr, best['top_left'], best['bottom_right'], out_file)
    print(f"Visualização salva em: {out_file}")
