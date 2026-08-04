"""
model.py — Kinematik unicycle diskret (Euler) + traksi + Lyapunov.

State  x = [px, py, theta]
Input  u = [v, omega]

px(k+1)    = px(k) + v(k)*cos(theta(k))*Ts + noise
py(k+1)    = py(k) + v(k)*sin(theta(k))*Ts + noise
theta(k+1) = theta(k) + omega(k)*Ts
"""

import numpy as np
import config


def step(state: np.ndarray, v: float, omega: float,
         noise: bool = False,
         rng: np.random.Generator = None) -> np.ndarray:
    """
    Satu langkah Euler. State: [px, py, theta].
    noise=True  → tambah noise proses (hanya di loop simulasi riil, BUKAN di MPC cost)
    noise=False → deterministik (dipakai di dalam optimasi MPC)
    rng         → Generator lokal per-run (wajib jika noise=True) untuk reproduktibilitas
    """
    px, py, th = state
    px_new = px + v * np.cos(th) * config.Ts
    py_new = py + v * np.sin(th) * config.Ts
    th_new = th + omega * config.Ts

    if noise and rng is not None:
        px_new += rng.normal(0.0, config.NOISE_SIGMA)
        py_new += rng.normal(0.0, config.NOISE_SIGMA)
        th_new += rng.normal(0.0, config.NOISE_SIGMA * 0.5)

    return np.array([px_new, py_new, th_new])


def v_achievable(T_score: float) -> float:
    """Kecepatan maksimum yang dapat dicapai pada terrain dengan skor T."""
    return config.V_MAX * max(T_score, 0.01)


def wrap_angle(a: float) -> float:
    """Bawa sudut ke [-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


# ══════════════════════════════════════════════════════════════════════════════
# Fungsi Lyapunov
# ══════════════════════════════════════════════════════════════════════════════

def lyapunov_V(state: np.ndarray,
               x_ref: float, y_ref: float, th_ref: float,
               Q_xy: float = None, Q_th: float = None) -> float:
    """
    V(e) = Q_xy*(ex²+ey²) + Q_th*e_theta²

    Error dihitung relatif terhadap titik TERDEKAT pada A*-path
    (bukan terhadap lookahead reference MPC).
    V_0 > 0 karena ada offset awal; harus turun monoton saat robot tracking path.
    """
    Q_xy = Q_xy if Q_xy is not None else config.MPC_Q_xy
    Q_th = Q_th if Q_th is not None else config.MPC_Q_th

    ex  = state[0] - x_ref
    ey  = state[1] - y_ref
    eth = wrap_angle(state[2] - th_ref)
    return float(Q_xy * (ex ** 2 + ey ** 2) + Q_th * eth ** 2)


def path_error(state: np.ndarray,
               x_ref: float, y_ref: float, th_ref: float) -> tuple:
    """
    Kembalikan (dist_error, heading_error) terhadap referensi terdekat.
    dist_error = jarak Euclidean ke titik referensi (m).
    heading_error = selisih sudut terbungkus (rad).
    """
    ex  = state[0] - x_ref
    ey  = state[1] - y_ref
    eth = wrap_angle(state[2] - th_ref)
    return float(np.hypot(ex, ey)), float(eth)
