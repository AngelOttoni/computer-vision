# -------------------------------------------------------------
# Projeta uma imagem (ex: lupita.jpg) sobre o tabuleiro calibrado
# usando os parâmetros obtidos em "results/camera_calibration.npz"
# -------------------------------------------------------------

import cv2 as cv
import numpy as np
import os
from datetime import datetime

# ------------------ CONFIGURAÇÕES ------------------
BASE_DIR = os.path.dirname(__file__)
CALIB_PATH = os.path.join(
    BASE_DIR, 'results', 'camera_calibration.npz'
)  # gerado pelo 8d_camera_calibration.py
OVERLAY_IMAGE_PATH = os.path.join(
    BASE_DIR, 'assets', 'lupita.jpg'
)  # 01_problem_set/assets/lupita.jpg

PATTERN_SIZE = (9, 6)  # cantos internos do tabuleiro
SQUARE_SIZE = 24.0  # tamanho do quadrado (mm)
RECORD_FPS = 30.0


# ------------------ FUNÇÕES AUXILIARES ------------------
def find_corners(gray):
    flags = (
        cv.CALIB_CB_ADAPTIVE_THRESH
        + cv.CALIB_CB_NORMALIZE_IMAGE
        + cv.CALIB_CB_FAST_CHECK
    )
    found, corners = cv.findChessboardCorners(gray, PATTERN_SIZE, flags)
    if found:
        corners = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )
    return found, corners


def build_board_coords_2d():
    cols, rows = PATTERN_SIZE
    objp = np.zeros((rows * cols, 2), np.float32)
    xs, ys = np.mgrid[0:cols, 0:rows]
    objp[:, :] = np.column_stack((xs.ravel(), ys.ravel())) * SQUARE_SIZE
    return objp


def overlay_with_homography(frame, overlay_bgr, H):
    h, w = frame.shape[:2]
    warped = cv.warpPerspective(overlay_bgr, H, (w, h))
    mask = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
    inv = cv.bitwise_not(mask)
    bg = cv.bitwise_and(frame, frame, mask=inv)
    fg = cv.bitwise_and(warped, warped, mask=mask)
    return cv.add(bg, fg)


# ------------------ CARREGA CALIBRAÇÃO E IMAGEM ------------------
try:
    D = np.load(CALIB_PATH, allow_pickle=True)
    K, dist = D["K"], D["dist"]
    print(f"[OK] Calibração carregada de: {CALIB_PATH}")
except Exception as e:
    raise SystemExit(f"[ERRO] Não foi possível carregar '{CALIB_PATH}': {e}")

overlay = cv.imread(OVERLAY_IMAGE_PATH)
if overlay is None:
    raise SystemExit(f"[ERRO] Não foi possível carregar a imagem: {OVERLAY_IMAGE_PATH}")

# Reduz overlay para desempenho, se necessário
h0, w0 = overlay.shape[:2]
if max(h0, w0) > 1200:
    scale = 1200.0 / max(h0, w0)
    overlay = cv.resize(
        overlay, (int(w0 * scale), int(h0 * scale)), interpolation=cv.INTER_AREA
    )

# Pontos de origem (pixels) da imagem a projetar
h_img, w_img = overlay.shape[:2]
src_img = np.float32([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]])

# Pontos do “mundo” plano do tabuleiro (em mm)
board_2d = build_board_coords_2d()

# ------------------ LOOP DE PROJEÇÃO ------------------
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("[ERRO] Não foi possível abrir a câmera.")

print(
    """
[8d] Controles:
  p  -> ligar/desligar projeção
  r  -> iniciar/parar gravação
  q  -> sair
"""
)

do_project = False
recording = False
writer = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners = find_corners(gray)

    view = frame.copy()
    if found:
        cv.drawChessboardCorners(view, PATTERN_SIZE, corners, True)

    if do_project and found:
        # Homografia: coords do tabuleiro (mm) -> imagem da câmera
        H_board, _ = cv.findHomography(
            board_2d,
            corners.reshape(-1, 2),
            method=cv.RANSAC,
            ransacReprojThreshold=3.0,
        )

        # Queremos mapear a imagem (em px) para cobrir todo o retângulo do tabuleiro.
        # Aproximação: largura = n_cols * SQUARE_SIZE ; altura = n_rows * SQUARE_SIZE
        cols, rows = PATTERN_SIZE
        w_board = cols * SQUARE_SIZE
        h_board = rows * SQUARE_SIZE
        dst_board_rect = np.float32(
            [[0, 0], [w_board, 0], [w_board, h_board], [0, h_board]]
        )

        # Homografia da imagem para o plano do tabuleiro
        H_img_to_board, _ = cv.findHomography(src_img, dst_board_rect)
        H_total = H_board @ H_img_to_board

        view = overlay_with_homography(frame, overlay, H_total)

    # Gravação
    if recording:
        if writer is None:
            h, w = view.shape[:2]
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            out_name = os.path.join(BASE_DIR, f"results/8d_projection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            writer = cv.VideoWriter(out_name, fourcc, RECORD_FPS, (w, h))
            print(f"[i] Gravando vídeo em: {out_name}")
        writer.write(view)

    # Exibição
    cv.imshow("8d - Projecao de Imagem no Tabuleiro", cv.flip(view, 1))
    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('p'):
        do_project = not do_project
        print(f"[i] Projecao {'ativada' if do_project else 'pausada'}.")
    elif k == ord('r'):
        recording = not recording
        if not recording and writer is not None:
            writer.release()
            writer = None
            print("[OK] Gravação finalizada e vídeo salvo.")

cap.release()
if writer is not None:
    writer.release()
cv.destroyAllWindows()
