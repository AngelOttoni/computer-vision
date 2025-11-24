'''
Implementação da detecção de múltiplos círculos via Transformada de Hough 2.5D
(a, b, r), desenvolvida para a questão 8(a) da Lista 2 de Visão Computacional.

O método não utiliza funções prontas de detecção de círculos do OpenCV.
A implementação segue a lógica clássica da Transformada de Hough:

1. Pré-computação de offsets de circunferência para cada raio.
2. Votação no espaço dos parâmetros (a, b, r).
3. Detecção de picos nos acumuladores.
4. Supressão não máxima para obter círculos distintos.

Autora: Angelina Meiras-Ottoni/ChatGPT 5.1
'''

import argparse
import csv
import os
import numpy as np
from math import cos, sin, pi
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path):
    # Carrega imagem, converte para escala de cinza e binariza.
    img = Image.open(path).convert('L')
    arr = np.array(img)
    binmap = (arr > 128).astype(np.uint8)
    return arr, binmap


def compute_perimeter_offsets(rmin, rmax, rstep, n_angles=180):
    # Pré-computa deslocamentos (dx, dy) para cada raio.
    angles = np.linspace(0, 2 * pi, n_angles, endpoint=False)
    radii = list(range(rmin, rmax + 1, rstep))
    offsets = {}
    for r in radii:
        pts = set()
        for th in angles:
            dx = int(round(r * cos(th)))
            dy = int(round(r * sin(th)))
            pts.add((dx, dy))
        offsets[r] = np.array(list(pts), dtype=int)
    return radii, offsets


def hough_vote(binmap, radii, offsets):
    # Executa votação no espaço de parâmetros (a, b, r).
    h, w = binmap.shape
    ys, xs = np.nonzero(binmap)
    accs = {r: np.zeros((h, w), dtype=np.uint16) for r in radii}

    for x, y in zip(xs, ys):
        for r in radii:
            offs = offsets[r]
            cx = x - offs[:, 0]
            cy = y - offs[:, 1]
            valid = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
            if np.any(valid):
                accs[r][cy[valid], cx[valid]] += 1
    return accs


def find_circle_candidates(accs, thresh_rel=0.5):
    # Extrai candidatos acima de um limiar relativo.
    candidates = []
    for r, acc in accs.items():
        maxv = acc.max()
        if maxv == 0:
            continue
        thresh = max(1, int(thresh_rel * maxv))
        ys, xs = np.nonzero(acc >= thresh)
        for y, x in zip(ys, xs):
            candidates.append((acc[y, x], x, y, r))
    candidates.sort(reverse=True, key=lambda t: t[0])
    return candidates


def non_max_suppression(candidates, dist=20, max_circles=10):
    # Filtra círculos redundantes mantendo apenas picos distintos.
    out = []
    for votes, x, y, r in candidates:
        if len(out) >= max_circles:
            break
        too_close = False
        for _, xc, yc, _ in out:
            if (xc - x)**2 + (yc - y)**2 <= dist**2:
                too_close = True
                break
        if not too_close:
            out.append((votes, x, y, r))
    return [(x, y, r) for _, x, y, r in out]

def save_results(circles, out_dir='results'):
    # Salva CSV com colunas a,b,r e retorna o caminho do arquivo.
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'detected_circles.csv')
    # Abrir com newline='' e usar csv.writer para compatibilidade
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['a', 'b', 'r'])
        for a, b, r in circles:
            writer.writerow([int(a), int(b), int(r)])
    return csv_path

def plot_and_save(gray, circles, out_dir='results'):
    # Plota os círculos sobre a imagem e salva o PNG em out_dir.
    os.makedirs(out_dir, exist_ok=True)
    h, w = gray.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gray, cmap='gray', origin='upper')
    ax.set_title("Circles detected via Hough Transform")
    for a, b, r in circles:
        circ = plt.Circle((a, b), r, fill=False, linewidth=1.5)
        ax.add_patch(circ)
        ax.plot(a, b, marker='x')
    ax.set_xlim([-0.5, w - 0.5])
    ax.set_ylim([h - 0.5, -0.5])
    ax.axis('off')

    out_path = os.path.join(out_dir, 'detected_circles.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='02_problem_set/assets/bolateste.png')
    parser.add_argument('--rmin', type=int, default=10)
    parser.add_argument('--rmax', type=int, default=90)
    parser.add_argument('--rstep', type=int, default=1)
    parser.add_argument('--thresh_rel', type=float, default=0.5)
    parser.add_argument('--n_angles', type=int, default=180)
    parser.add_argument('--nms_dist', type=int, default=20)
    parser.add_argument('--max_circles', type=int, default=10)
    args = parser.parse_args()

    gray, binmap = load_image(args.image)
    radii, offsets = compute_perimeter_offsets(args.rmin, args.rmax, args.rstep, n_angles=args.n_angles)
    accs = hough_vote(binmap, radii, offsets)
    candidates = find_circle_candidates(accs, thresh_rel=args.thresh_rel)
    circles = non_max_suppression(candidates, dist=args.nms_dist, max_circles=args.max_circles)

    print("Circles detected (a, b, r):")
    for a, b, r in circles:
        print(f"({a}, {b}, {r})")

    csv_path = save_results(circles, out_dir='02_problem_set/results/')
    png_path = plot_and_save(gray, circles, out_dir='02_problem_set/results/')

    print(f"Saved CSV -> {csv_path}")
    print(f"Saved PNG -> {png_path}")

if __name__ == '__main__':
    main()
