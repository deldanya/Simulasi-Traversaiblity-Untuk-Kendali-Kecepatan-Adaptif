"""
controller.py — Tiga kontroler UGV.

Skenario A — P-Controller
Skenario B — MPC tanpa constraint traversability  (use_T=False)
Skenario C — MPC + constraint v(k) <= v_max*T(k)  (use_T=True)

PERBAIKAN BUG KRITIS (vs ugv_sim sebelumnya):
MPC menggunakan referensi LOOKAHEAD — titik target bergerak maju di sepanjang
A*-path dengan laju v_target = V_MAX * T(terrain saat ini).
Dengan ini robot selalu memiliki error posisi ke depan yang mendorongnya maju,
sehingga solver tidak lagi memilih v=0 sebagai solusi optimal.
"""

import numpy as np
from scipy.optimize import minimize

import config
import model
import pathplanning as pp


# ══════════════════════════════════════════════════════════════════════════════
# Skenario A — P-Controller
# ══════════════════════════════════════════════════════════════════════════════

def ctrl_P(state: np.ndarray,
           path_xy: list,
           path_s: np.ndarray,
           path_x: np.ndarray,
           path_y: np.ndarray,
           path_th: np.ndarray,
           T_score: float,
           hint_idx: int) -> tuple:
    """
    v     = v_max * T(terrain)            — kecepatan proporsional traversabilitas
    omega = Kp * e_theta                  — heading error ke titik lookahead
    Kembalikan (v, omega, new_hint_idx).
    """
    # Temukan titik lookahead: s_current + LOOKAHEAD
    s_cur, idx_cur = pp.project_to_path((state[0], state[1]),
                                         path_xy, path_s, hint_idx)
    s_look = s_cur + config.LOOKAHEAD
    x_ref, y_ref, th_ref = pp.interp_path_at_s(
        s_look, path_s, path_x, path_y, path_th)

    th_des = np.arctan2(y_ref - state[1], x_ref - state[0])
    e_th   = model.wrap_angle(th_des - state[2])

    v     = float(np.clip(model.v_achievable(T_score), 0.0, config.V_MAX))
    omega = float(np.clip(config.KP_OMEGA * e_th,
                          -config.OMEGA_MAX, config.OMEGA_MAX))
    return v, omega, idx_cur


# ══════════════════════════════════════════════════════════════════════════════
# Skenario B & C — MPC dengan referensi LOOKAHEAD
# ══════════════════════════════════════════════════════════════════════════════

class MPCController:
    """
    MPC horizon Np, input horizon Nc.
    use_T=True  → tambahkan constraint v(k) <= V_MAX * T_horizon[k].

    REFERENSI LOOKAHEAD (perbaikan bug utama):
        Untuk langkah j = 0..Np-1 dalam horizon:
            s_ref_j = s_current + v_target * Ts * (j+1)
            (x_ref_j, y_ref_j, th_ref_j) = interpolasi jalur di s_ref_j

        v_target = V_MAX * T_current   ← kecepatan target berdasarkan terrain saat ini

        Ini menciptakan "titik tarik" yang selalu berada di DEPAN robot,
        memaksa solver memilih v > 0 agar error berkurang.
    """

    def __init__(self,
                 use_T: bool,
                 path_xy: list,
                 path_s: np.ndarray,
                 path_x: np.ndarray,
                 path_y: np.ndarray,
                 path_th: np.ndarray,
                 Np: int = None,
                 Nc: int = None,
                 Q_xy: float = None,
                 Q_th: float = None,
                 Rv: float = None,
                 Rw: float = None):

        self.use_T   = use_T
        self.Np      = Np   or config.MPC_Np
        self.Nc      = Nc   or config.MPC_Nc
        self.Q_xy    = Q_xy or config.MPC_Q_xy
        self.Q_th    = Q_th or config.MPC_Q_th
        self.Rv      = Rv   or config.MPC_Rv
        self.Rw      = Rw   or config.MPC_Rw

        # Cache jalur sebagai numpy arrays untuk interpolasi cepat
        self.path_xy = path_xy
        self.path_s  = np.asarray(path_s)
        self.path_x  = np.asarray(path_x)
        self.path_y  = np.asarray(path_y)
        self.path_th = np.asarray(path_th)

        # Warm-start: solusi kontrol horizon sebelumnya
        self._U_prev = None

    def __call__(self,
                 state: np.ndarray,
                 s_current: float,
                 T_current: float,
                 T_horizon: list,
                 hint_idx: int,
                 v_prev_actual: float = 0.0) -> tuple:
        """
        Hitung (v_opt, omega_opt, new_hint_idx).

        state         : [px, py, theta]
        s_current     : panjang busur ke titik terdekat pada jalur
        T_current     : skor T terrain di posisi robot saat ini
        T_horizon     : list skor T sepanjang Np langkah horizon (untuk MPC+T)
        hint_idx      : indeks waypoint terdekat (untuk P-proyeksi berikutnya)
        v_prev_actual : kecepatan aktual dari langkah sebelumnya (untuk constraint Δv)
        """
        Np, Nc = self.Np, self.Nc

        # ── Batas atas v per langkah horizon ──────────────────────────────
        # v_ub[0] harus menggunakan T_current (terrain SAAT INI), bukan
        # T_horizon[0] (satu langkah ke depan).  Bug lama: T_horizon[0]
        # terlalu longgar saat transisi low-T→high-T (mis. puddle→grass),
        # menyebabkan v[0] > V_MAX*T_current → violations.
        if self.use_T:
            v_ub = ([max(config.V_MAX * T_current, 0.10)] +
                    [max(config.V_MAX * T_horizon[j], 0.10)
                     for j in range(Nc - 1)])
        else:
            v_ub = [config.V_MAX] * Nc

        # ── Referensi lookahead — precompute SEBELUM optimasi ─────────────
        # Gunakan V_MAX penuh sebagai kecepatan referensi lookahead.
        # Ini memastikan titik target selalu JAUH di depan robot, sehingga
        # solver tidak pernah menemukan v=0 sebagai local minimum.
        # Constraint v ≤ v_max*T tetap aktif untuk MPCT; tapi insentif maju tetap kuat.
        v_target = config.V_MAX
        refs = []
        for j in range(Np):
            s_ref = s_current + v_target * config.Ts * (j + 1)
            xr, yr, thr = pp.interp_path_at_s(
                s_ref, self.path_s,
                self.path_x, self.path_y, self.path_th)
            refs.append((xr, yr, thr))

        # ── Constraint percepatan: |v(k) - v(k-1)| ≤ A_MAX * Ts ─────────
        # Mencegah lompatan kecepatan antar-langkah (fisika nyata).
        a_step = config.A_MAX * config.Ts   # = 1.0 * 0.1 = 0.1 m/s per langkah

        def _acc_upper(U):
            vs_u = U[:Nc]
            r = np.empty(Nc)
            r[0] = a_step - (vs_u[0] - v_prev_actual)
            for k in range(1, Nc):
                r[k] = a_step - (vs_u[k] - vs_u[k - 1])
            return r

        def _acc_lower(U):
            vs_u = U[:Nc]
            r = np.empty(Nc)
            r[0] = a_step + (vs_u[0] - v_prev_actual)
            for k in range(1, Nc):
                r[k] = a_step + (vs_u[k] - vs_u[k - 1])
            return r

        mpc_constraints = [
            {'type': 'ineq', 'fun': _acc_upper},
            {'type': 'ineq', 'fun': _acc_lower},
        ]

        # ── Bounds: [v_0..v_{Nc-1}, omega_0..omega_{Nc-1}] ───────────────
        bounds = (
            [(0.0, v_ub[j]) for j in range(Nc)] +
            [(-config.OMEGA_MAX, config.OMEGA_MAX)] * Nc
        )

        # ── Warm-start ────────────────────────────────────────────────────
        if self._U_prev is not None and len(self._U_prev) == 2 * Nc:
            vs = np.concatenate([self._U_prev[1:Nc],    [self._U_prev[Nc - 1]]])
            ws = np.concatenate([self._U_prev[Nc + 1:], [self._U_prev[-1]]])
            U0 = np.concatenate([vs, ws])
            # Reset warm-start jika solusi sebelumnya sangat lambat (stuck di local min)
            if np.mean(vs) < 0.25 * v_ub[0]:
                U0 = np.array([0.8 * v_ub[0]] * Nc + [0.0] * Nc)
        else:
            U0 = np.array([0.85 * v_ub[0]] * Nc + [0.0] * Nc)

        # Clip U0 ke bounds terlebih dahulu
        for j in range(Nc):
            U0[j]      = np.clip(U0[j],      bounds[j][0],      bounds[j][1])
            U0[Nc + j] = np.clip(U0[Nc + j], bounds[Nc+j][0],   bounds[Nc+j][1])

        # Kemudian buat U0 feasible terhadap constraint akselerasi
        U0[0] = np.clip(U0[0], max(0.0, v_prev_actual - a_step),
                        min(v_ub[0], v_prev_actual + a_step))
        for j in range(1, Nc):
            U0[j] = np.clip(U0[j], max(0.0, U0[j - 1] - a_step),
                            min(v_ub[j], U0[j - 1] + a_step))

        # ── Fungsi biaya MPC ──────────────────────────────────────────────
        Q_xy, Q_th = self.Q_xy, self.Q_th
        Rv, Rw     = self.Rv,   self.Rw

        def cost_fn(U):
            vs_c = U[:Nc]
            ws_c = U[Nc:]
            # Perpanjang ke Np dengan nilai terakhir
            v_ext = np.empty(Np); v_ext[:Nc] = vs_c; v_ext[Nc:] = vs_c[-1]
            w_ext = np.empty(Np); w_ext[:Nc] = ws_c; w_ext[Nc:] = ws_c[-1]

            total = 0.0
            s = state.copy()
            for j in range(Np):
                s = model.step(s, float(v_ext[j]), float(w_ext[j]), noise=False)
                x_r, y_r, th_r = refs[j]
                ex  = s[0] - x_r
                ey  = s[1] - y_r
                eth = model.wrap_angle(s[2] - th_r)
                total += Q_xy * (ex * ex + ey * ey) + Q_th * eth * eth
                if j < Nc:
                    total += Rv * vs_c[j] * vs_c[j] + Rw * ws_c[j] * ws_c[j]
            return total

        # ── Solve ─────────────────────────────────────────────────────────
        res = minimize(cost_fn, U0, method="SLSQP", bounds=bounds,
                       constraints=mpc_constraints,
                       options={"maxiter": 50, "ftol": 1e-6})

        U_opt = res.x
        self._U_prev = U_opt.copy()

        v_opt  = float(np.clip(U_opt[0],     0.0,              v_ub[0]))
        om_opt = float(np.clip(U_opt[Nc], -config.OMEGA_MAX, config.OMEGA_MAX))

        # Hitung hint_idx terbaru (untuk Lyapunov & T-query berikutnya)
        _, new_idx = pp.project_to_path(
            (state[0], state[1]), self.path_xy, self.path_s, hint_idx)

        return v_opt, om_opt, new_idx


# ══════════════════════════════════════════════════════════════════════════════
# Utilitas horizon T
# ══════════════════════════════════════════════════════════════════════════════

def build_T_horizon(state: np.ndarray,
                    T_grid: np.ndarray,
                    path_xy: list,
                    path_s: np.ndarray,
                    path_x: np.ndarray,
                    path_y: np.ndarray,
                    drivetrain: str,
                    s_current: float,
                    Np: int) -> list:
    """
    Estimasi T sepanjang horizon MPC dengan roll-forward sederhana.
    Posisi prediksi dihitung dari jalur A* (bukan simulasi penuh).
    """
    import terrain as trn

    T_cur    = trn.get_T_at_world(state[0], state[1], T_grid)
    v_nom    = config.V_MAX * max(T_cur, 0.10)

    T_seq = []
    for j in range(Np):
        s_pred = s_current + v_nom * config.Ts * (j + 1)
        s_pred = float(np.clip(s_pred, path_s[0], path_s[-1]))
        x_pred = float(np.interp(s_pred, path_s, path_x))
        y_pred = float(np.interp(s_pred, path_s, path_y))
        T_j    = trn.get_T_at_world(x_pred, y_pred, T_grid)
        T_seq.append(T_j)
    return T_seq
