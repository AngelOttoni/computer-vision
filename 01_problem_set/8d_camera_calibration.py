# -------------------------------------------------------------
# Calibra a câmera usando um tabuleiro de cantos internos (9x6)
# e salva o resultado em "01_problem_set/results/camera_calibration.npz".
# -------------------------------------------------------------

import cv2 as cv
import numpy as np
import os

# ------------------ CONFIGURAÇÕES ------------------
pattern_size = (9, 6)  # (cols, rows) cantos internos
square_size = 25.0 # mm
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
flags_cb = (
    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
)

# Caminho absoluto da pasta "results/"
BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SAVE_PATH = os.path.join(RESULTS_DIR, "camera_calibration.npz")

# ------------------ PREPARAÇÃO ------------------
cols, rows = pattern_size
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

objpoints, imgpoints = [], []
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("[ERRO] Não foi possível acessar a câmera.")

img_size = None
samples = 0

print("""
[8d] Calibração da Câmera
---------------------------------
Comandos:
  x -> capturar amostra (tabuleiro visível)
  c -> calcular e salvar calibração
  q -> sair
---------------------------------
""")

# ------------------ LOOP PRINCIPAL ------------------
while True:
    ok, frame = cap.read()
    if not ok:
        break

    if img_size is None:
        img_size = (frame.shape[1], frame.shape[0])

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, pattern_size, flags_cb)

    if found:
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv.drawChessboardCorners(frame, pattern_size, corners, True)

    # Espelhamento apenas para visualização
    view = cv.flip(frame, 1)
    cv.putText(view, f"samples: {samples}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv.imshow("Camera Calibration", view)

    k = cv.waitKey(1) & 0xFF
    if k == ord('x') and found:
        objpoints.append(objp.copy())
        imgpoints.append(corners.reshape(-1, 2))
        samples += 1
        print(f"[+] Amostra capturada: {samples}")

    elif k == ord('c') and len(objpoints) >= 10:
        ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, [ip.reshape(-1, 1, 2) for ip in imgpoints], img_size, None, None
        )
        np.savez(
            SAVE_PATH,
            K=K,
            dist=dist,
            rvecs=rvecs,
            tvecs=tvecs,
            img_size=img_size,
            pattern_size=pattern_size,
            square_size=square_size,
        )

        print(f"\n[OK] Calibração concluída (rmse={ret:.4f})")
        print(f"Arquivo salvo em: {SAVE_PATH}")
        print("K=\n", K, "\ndist=\n", dist.ravel())

    elif k == ord('q'):
        break

# ------------------ FINALIZAÇÃO ------------------
cap.release()
cv.destroyAllWindows()
print("\nEncerrado.")