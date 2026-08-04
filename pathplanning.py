"""
pathplanning.py — A* path planning pada grid 2D + utilitas jalur.

A* 8-arah dengan biaya masuk sel = 1/T (sel T rendah = mahal).
Obstacle (T=T_CLAMP_MIN) diblokir.
Heuristik: jarak Euclidean.
"""

import heapq
import numpy as np
from scipy.ndimage import gaussian_filter1d

import config


# ══════════════════════════════════════════════════════════════════════════════
# A* Path Planning
# ══════════════════════════════════════════════════════════════════════════════

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1),
           (0, -1),           (0,  1),
           (1, -1),  (1, 0),  (1,  1)]

_OBSTACLE_T = config.T_CLAMP_MIN + 1e-6   # T ≤ ini dianggap obstacle


def astar(T_grid: np.ndarray,
          start_rc: tuple,
          goal_rc: tuple) -> list:
    """
    Cari jalur terpendek (berdasarkan biaya traversal) dari start ke goal.

    Biaya masuk sel (nr, nc) dari arah (dr, dc):
        step_dist = hypot(dr, dc)   # 1.0 atau sqrt(2)
        cost      = step_dist / T[nr, nc]

    Kembalikan list[(row, col)] dari start ke goal,
    atau None jika tidak ada jalur.
    """
    nrows, ncols = T_grid.shape
    start = tuple(start_rc)
    goal  = tuple(goal_rc)

    def h(r, c):
        return float(np.hypot(r - goal[0], c - goal[1]))

    # (f, g, (row, col))
    open_heap = []
    heapq.heappush(open_heap, (h(*start), 0.0, start))

    g_score   = {start: 0.0}
    came_from = {}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == goal:
            return _reconstruct(came_from, current)

        if g > g_score.get(current, np.inf) + 1e-9:
            continue   # entri usang

        r, c = current
        for dr, dc in _DIRS8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < nrows and 0 <= nc < ncols):
                continue
            T_n = float(T_grid[nr, nc])
            if T_n <= _OBSTACLE_T:
                continue   # obstacle atau sangat rendah

            step_dist = float(np.hypot(dr, dc))
            new_g = g + step_dist / T_n

            neighbor = (nr, nc)
            if new_g < g_score.get(neighbor, np.inf):
                g_score[neighbor]   = new_g
                came_from[neighbor] = current
                f_new = new_g + h(nr, nc)
                heapq.heappush(open_heap, (f_new, new_g, neighbor))

    return None   # tidak ada jalur


def _reconstruct(came_from: dict, current: tuple) -> list:
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return list(reversed(path))


# ══════════════════════════════════════════════════════════════════════════════
# Konversi & utilitas jalur
# ══════════════════════════════════════════════════════════════════════════════

def path_to_world(path_rc: list) -> list:
    """Konversi list (row, col) → list (x, y) dunia (meter)."""
    return [(c * config.CELL_SIZE, r * config.CELL_SIZE)
            for r, c in path_rc]


def smooth_path(path_xy: list, sigma: float = None) -> list:
    """
    Gaussian smoothing koordinat x dan y secara terpisah.
    Titik awal dan akhir tetap (tidak bergeser).
    """
    if len(path_xy) < 5:
        return path_xy
    sigma = sigma or config.PATH_SMOOTH_SIGMA
    xs = np.array([p[0] for p in path_xy])
    ys = np.array([p[1] for p in path_xy])
    xs_s = gaussian_filter1d(xs, sigma=sigma)
    ys_s = gaussian_filter1d(ys, sigma=sigma)
    # Pertahankan titik ujung
    xs_s[0] = xs[0];  xs_s[-1] = xs[-1]
    ys_s[0] = ys[0];  ys_s[-1] = ys[-1]
    return list(zip(xs_s.tolist(), ys_s.tolist()))


def resample_path(path_xy: list, ds: float = None) -> list:
    """
    Resampling jalur ke jarak uniform ds meter antar titik.
    Menghasilkan interpolasi lebih halus untuk referensi MPC.
    """
    ds = ds or config.PATH_RESAMPLE_DS
    if len(path_xy) < 2:
        return path_xy
    s_arr = compute_arc_lengths(path_xy)
    total = s_arr[-1]
    if total < 1e-6:
        return path_xy
    s_new = np.arange(0.0, total, ds)
    if s_new[-1] < total:
        s_new = np.append(s_new, total)
    xs = np.array([p[0] for p in path_xy])
    ys = np.array([p[1] for p in path_xy])
    xs_new = np.interp(s_new, s_arr, xs)
    ys_new = np.interp(s_new, s_arr, ys)
    return list(zip(xs_new.tolist(), ys_new.tolist()))


def compute_arc_lengths(path_xy: list) -> np.ndarray:
    """Panjang busur kumulatif (meter) dari titik pertama jalur."""
    s = [0.0]
    for i in range(1, len(path_xy)):
        dx = path_xy[i][0] - path_xy[i - 1][0]
        dy = path_xy[i][1] - path_xy[i - 1][1]
        s.append(s[-1] + float(np.hypot(dx, dy)))
    return np.array(s)


def compute_headings(path_xy: list) -> np.ndarray:
    """
    Sudut heading (rad) di setiap titik jalur.
    Dihitung dari arah ke titik berikutnya; titik terakhir mewarisi heading sebelumnya.
    """
    n  = len(path_xy)
    th = np.zeros(n)
    for i in range(n - 1):
        dx = path_xy[i + 1][0] - path_xy[i][0]
        dy = path_xy[i + 1][1] - path_xy[i][1]
        th[i] = np.arctan2(dy, dx)
    th[-1] = th[-2] if n >= 2 else 0.0
    return th


# ══════════════════════════════════════════════════════════════════════════════
# Proyeksi state ke jalur
# ══════════════════════════════════════════════════════════════════════════════

def project_to_path(xy: tuple,
                    path_xy: list,
                    path_s: np.ndarray,
                    hint_idx: int = 0) -> tuple:
    """
    Temukan titik terdekat pada jalur dari posisi (x, y).

    Pencarian dimulai dari hint_idx - 5 untuk efisiensi
    (asumsi robot bergerak maju di sepanjang jalur).

    Kembalikan (s_closest, closest_idx).
    """
    x, y = xy
    n    = len(path_xy)
    search_start = max(0, hint_idx - 5)
    search_end   = min(n, hint_idx + 80)   # jendela pencarian ke depan

    best_idx  = hint_idx
    best_dist = np.inf
    for i in range(search_start, search_end):
        px, py = path_xy[i]
        d = (x - px) ** 2 + (y - py) ** 2
        if d < best_dist:
            best_dist = d
            best_idx  = i

    return float(path_s[best_idx]), best_idx


def interp_path_at_s(s_query: float,
                     path_s: np.ndarray,
                     path_x: np.ndarray,
                     path_y: np.ndarray,
                     path_th: np.ndarray) -> tuple:
    """
    Interpolasi (x_ref, y_ref, th_ref) pada panjang busur s_query.
    s_query dijepit ke [path_s[0], path_s[-1]].
    """
    s_q = float(np.clip(s_query, path_s[0], path_s[-1]))
    x_r  = float(np.interp(s_q, path_s, path_x))
    y_r  = float(np.interp(s_q, path_s, path_y))
    th_r = float(np.interp(s_q, path_s, path_th))
    return x_r, y_r, th_r


def build_path_arrays(path_xy: list, path_s: np.ndarray, path_th: np.ndarray):
    """Kembalikan numpy arrays untuk x, y, th (efisien untuk interpolasi berulang)."""
    path_x  = np.array([p[0] for p in path_xy])
    path_y  = np.array([p[1] for p in path_xy])
    path_th = np.asarray(path_th)
    path_s  = np.asarray(path_s)
    return path_x, path_y, path_th, path_s


def full_pipeline(T_grid: np.ndarray,
                  start_rc: tuple,
                  goal_rc: tuple) -> tuple:
    """
    Jalankan A* + smooth + resample + hitung arc-length & heading.
    Kembalikan (path_xy, path_s, path_x, path_y, path_th).
    Raise RuntimeError jika A* tidak menemukan jalur.
    """
    path_rc = astar(T_grid, start_rc, goal_rc)
    if path_rc is None:
        raise RuntimeError(
            f"A* tidak menemukan jalur dari {start_rc} ke {goal_rc}. "
            "Pastikan tidak ada obstacle yang memblokir seluruh jalur."
        )

    raw_xy   = path_to_world(path_rc)
    smooth   = smooth_path(raw_xy)
    final_xy = resample_path(smooth)

    path_s   = compute_arc_lengths(final_xy)
    path_th  = compute_headings(final_xy)
    path_x   = np.array([p[0] for p in final_xy])
    path_y   = np.array([p[1] for p in final_xy])

    return final_xy, path_s, path_x, path_y, path_th
