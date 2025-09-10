import cv2 as cv, numpy as np

CALIB = "calibracao_camera.npz"  # Tem que ser gerado com camera_calibration.py

ps = (9, 6)
sq = 25.0
cols, rows = ps
L = 3 * sq
D = np.load(CALIB, allow_pickle=True)
K, dist = D["K"], D["dist"]
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq
cube = np.float32(
    [
        [0, 0, 0],
        [L, 0, 0],
        [L, L, 0],
        [0, L, 0],
        [0, 0, -L],
        [L, 0, -L],
        [L, L, -L],
        [0, L, -L],
    ]
)
axis = np.float32([[2 * L, 0, 0], [0, 2 * L, 0], [0, 0, -2 * L]])
crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
flags = (
    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
)

cap = cv.VideoCapture(0)
while True:
    ok, f = cap.read()
    g = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
    found, c = cv.findChessboardCorners(g, ps, flags)
    if found:
        c = cv.cornerSubPix(g, c, (11, 11), (-1, -1), crit)
        okp, rvec, tvec = cv.solvePnP(objp, c, K, dist, flags=cv.SOLVEPNP_ITERATIVE)
        P, _ = cv.projectPoints(cube, rvec, tvec, K, dist)
        P = P.reshape(-1, 2).astype(int)
        A, _ = cv.projectPoints(axis, rvec, tvec, K, dist)
        A = A.reshape(-1, 2).astype(int)
        c0, _ = cv.projectPoints(np.float32([[0, 0, 0]]), rvec, tvec, K, dist)
        c0 = tuple(c0.reshape(2).astype(int))
        for a, b in [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
        ]:
            cv.line(f, tuple(P[a]), tuple(P[b]), (50, 180, 255), 2)
        cv.line(f, c0, tuple(A[0]), (0, 0, 255), 3)
        cv.line(f, c0, tuple(A[1]), (0, 255, 0), 3)
        cv.line(f, c0, tuple(A[2]), (255, 0, 0), 3)
    f = cv.flip(f, 1)
    cv.imshow("AR", f)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break
