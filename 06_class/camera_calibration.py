import cv2 as cv
import numpy as np

pattern_size = (9, 6)  # (cols, rows) cantos internos
square_size = 25.0
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
flags_cb = (
    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
)
cols, rows = pattern_size
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
objpoints, imgpoints = [], []
cap = cv.VideoCapture(0)
img_size = None

while True:
    ok, frame = cap.read()
    if img_size is None:
        img_size = (frame.shape[1], frame.shape[0])
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, pattern_size, flags_cb)
    if found:
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv.drawChessboardCorners(frame, pattern_size, corners, True)
    frame = cv.flip(frame, 1)
    cv.imshow("Calibracao", frame)
    k = cv.waitKey(1) & 0xFF
    if k == ord('x') and found:
        objpoints.append(objp.copy())
        imgpoints.append(corners.reshape(-1, 2))
    elif k == ord('c') and len(objpoints) >= 5:
        ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, [ip.reshape(-1, 1, 2) for ip in imgpoints], img_size, None, None
        )
        np.savez(
            "calibracao_camera.npz",
            K=K,
            dist=dist,
            rvecs=rvecs,
            tvecs=tvecs,
            img_size=img_size,
            pattern_size=pattern_size,
            square_size=square_size,
        )
    elif k == ord('q'):
        cap.release()
        cv.destroyAllWindows()
    break