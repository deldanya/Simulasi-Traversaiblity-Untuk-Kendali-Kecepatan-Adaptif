"""
simulate.py — Loop simulasi + auto-tuning + metrik.

6 run: {P, MPC, MPC+T} × {2WD, 4WD}.
MPC menggunakan referensi lookahead (konvergensi dijamin).
Grafik HANYA dibuat setelah semua run mencapai goal.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import itertools
from dataclasses import dataclass, field
import numpy as np

import config
import model
import terrain as trn
import pathplanning as pp
import controller as ctrl


# ══════════════════════════════════════════════════════════════════════════════
# Data container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    label:      str
    scenario:   str          # "P" / "MPC" / "MPCT"
    drivetrain: str          # "2WD" / "4WD"

    t:       np.ndarray = field(default_factory=lambda: np.array([]))
    px:      np.ndarray = field(default_factory=lambda: np.array([]))
    py:      np.ndarray = field(default_factory=lambda: np.array([]))
    theta:   np.ndarray = field(default_factory=lambda: np.array([]))
    v:       np.ndarray = field(default_factory=lambda: np.array([]))
    omega:   np.ndarray = field(default_factory=lambda: np.array([]))
    T_arr:   np.ndarray = field(default_factory=lambda: np.array([]))
    V_lyap:  np.ndarray = field(default_factory=lambda: np.array([]))
    pos_err: np.ndarray = field(default_factory=lambda: np.array([]))   # dist to path
    v_ratio: np.ndarray = field(default_factory=lambda: np.array([]))   # v/(v_max*T)

    ter_list:     list = field(default_factory=list)
    viols:        int  = 0
    risk_events:  int  = 0
    reached_goal: bool = False
    path_frac:    float = 0.0    # fraksi jalur yang dicapai


# ══════════════════════════════════════════════════════════════════════════════
# Loop simulasi tunggal
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(scenario: str,
                   drivetrain: str,
                   terrain_grid: np.ndarray,
                   T_grid: np.ndarray,
                   path_xy: list,
                   path_s: np.ndarray,
                   path_x: np.ndarray,
                   path_y: np.ndarray,
                   path_th: np.ndarray,
                   mpc_params: dict = None) -> RunResult:
    """
    Jalankan satu run simulasi.
    mpc_params: dict opsional dengan kunci Np, Nc, Q_xy, Q_th, Rv, Rw.
    """
    label = f"{config.SCENARIO_LABEL[scenario]}-{drivetrain}"
    print(f"    [{label}] ...", end="", flush=True)

    # ── Inisialisasi kontroler ────────────────────────────────────────────
    mpc_params = mpc_params or {}
    if scenario == "P":
        mpc_ctrl = None
    elif scenario == "MPC":
        mpc_ctrl = ctrl.MPCController(
            use_T=False,
            path_xy=path_xy, path_s=path_s,
            path_x=path_x, path_y=path_y, path_th=path_th,
            **mpc_params)
    else:  # MPCT
        mpc_ctrl = ctrl.MPCController(
            use_T=True,
            path_xy=path_xy, path_s=path_s,
            path_x=path_x, path_y=path_y, path_th=path_th,
            **mpc_params)

    # ── State awal: start_world + offset tegak lurus arah A*-path ─────────
    # Offset perpendikular sebesar INIT_PERP_OFFSET agar V_lyap_0 > 0.
    th0_path = float(path_th[0])  # heading awal jalur
    perp_dir = th0_path + np.pi / 2   # tegak lurus
    x0 = config.START_WORLD[0] + config.INIT_PERP_OFFSET * np.cos(perp_dir)
    y0 = config.START_WORLD[1] + config.INIT_PERP_OFFSET * np.sin(perp_dir)
    th0 = th0_path + config.INIT_THETA_OFFSET

    state    = np.array([x0, y0, th0])
    hint_idx = 0
    path_len = float(path_s[-1])

    # ── Buffer ────────────────────────────────────────────────────────────
    ts, pxs, pys, ths = [], [], [], []
    vs, oms, T_list, V_list = [], [], [], []
    pos_errs, v_ratios = [], []
    ter_list = []
    viols = 0
    risk_events = 0

    # RNG lokal per-run — seed tetap agar setiap run identik dengan parameter sama
    rng = np.random.default_rng(config.NOISE_SEED)

    # Rate limiting — kecepatan aktual step sebelumnya
    v_prev  = 0.0
    a_step  = config.A_MAX * config.Ts   # = 0.1 m/s — batas Δv per langkah

    for step_k in range(config.MAX_STEPS):
        x, y, th = state

        # Terrain & T saat ini
        ter_lbl = trn.get_terrain_at_world(x, y, terrain_grid)
        T_score = trn.get_T_at_world(x, y, T_grid)

        # Proyeksi ke jalur → s_current, x_ref, y_ref, th_ref
        s_cur, hint_idx = pp.project_to_path(
            (x, y), path_xy, path_s, hint_idx)
        x_ref, y_ref, th_ref = pp.interp_path_at_s(
            s_cur, path_s, path_x, path_y, path_th)

        # Lyapunov V(e) relatif ke titik terdekat pada jalur
        V_val = model.lyapunov_V(state, x_ref, y_ref, th_ref,
                                  config.MPC_Q_xy, config.MPC_Q_th)
        pos_err = float(np.hypot(x - x_ref, y - y_ref))

        # Risk event
        if T_score < config.T_RISK:
            risk_events += 1

        # Kontrol
        if scenario == "P":
            v, omega, hint_idx = ctrl.ctrl_P(
                state, path_xy, path_s, path_x, path_y, path_th,
                T_score, hint_idx)
        else:
            T_horizon = ctrl.build_T_horizon(
                state, T_grid, path_xy, path_s, path_x, path_y,
                drivetrain, s_cur, mpc_ctrl.Np)
            v, omega, hint_idx = mpc_ctrl(
                state, s_cur, T_score, T_horizon, hint_idx,
                v_prev_actual=v_prev)

        # Lapis 2 — post-clip rate limit (jaminan fisik untuk semua kontroler)
        v = float(np.clip(v, v_prev - a_step, v_prev + a_step))
        v_prev = v

        # Violation check (menggunakan v aktual setelah rate limit)
        v_lim = config.V_MAX * T_score
        if v > v_lim + config.VIOL_TOL:
            viols += 1
        v_rat = v / max(v_lim, 1e-4)

        # Simpan
        ts.append(step_k * config.Ts)
        pxs.append(x); pys.append(y); ths.append(th)
        vs.append(v);  oms.append(omega)
        T_list.append(T_score); V_list.append(V_val)
        pos_errs.append(pos_err); v_ratios.append(v_rat)
        ter_list.append(ter_lbl)

        # Update state (dengan noise proses kecil, RNG lokal per-run)
        state = model.step(state, v, omega, noise=True, rng=rng)

        # Cek tujuan: jarak ke goal_world < GOAL_RADIUS
        gx, gy = config.GOAL_WORLD
        dist_goal = float(np.hypot(state[0] - gx, state[1] - gy))
        if dist_goal < config.GOAL_RADIUS and step_k > 20:
            break

    # Fraksi jalur yang dicapai
    s_final, _ = pp.project_to_path(
        (pxs[-1], pys[-1]), path_xy, path_s, hint_idx)
    path_frac = float(s_final / max(path_len, 1e-6))
    reached   = path_frac >= config.PROGRESS_MIN

    steps = len(ts)
    print(f" {steps} steps | s={s_final:.1f}/{path_len:.1f}m "
          f"({100*path_frac:.0f}%) | viols={viols} | "
          f"goal={'OK' if reached else 'FAIL'}")

    return RunResult(
        label=label, scenario=scenario, drivetrain=drivetrain,
        t=np.array(ts), px=np.array(pxs), py=np.array(pys),
        theta=np.array(ths), v=np.array(vs), omega=np.array(oms),
        T_arr=np.array(T_list), V_lyap=np.array(V_list),
        pos_err=np.array(pos_errs), v_ratio=np.array(v_ratios),
        ter_list=ter_list, viols=viols, risk_events=risk_events,
        reached_goal=reached, path_frac=path_frac,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Auto-tuning MPC
# ══════════════════════════════════════════════════════════════════════════════

def _lyapunov_monotone_pct(V: np.ndarray) -> float:
    if len(V) < 2:
        return 0.0
    dV = np.diff(V)
    return float(np.mean(dV <= 1e-6)) * 100.0


def autotune_mpc(scenario: str,
                 drivetrain: str,
                 terrain_grid: np.ndarray,
                 T_grid: np.ndarray,
                 path_xy: list,
                 path_s: np.ndarray,
                 path_x: np.ndarray,
                 path_y: np.ndarray,
                 path_th: np.ndarray,
                 p_ref_rmse: float) -> dict:
    """
    Grid search kecil untuk menemukan parameter MPC yang konvergen.
    Kriteria: (a) goal reached, (b) RMSE ≤ RMSE P-ctrl, (c) Lyap mono ≥ 60%.
    """
    best_params = None
    best_score  = -np.inf

    n_combos = len(config.TUNE_Np) * len(config.TUNE_Q_xy) * len(config.TUNE_Rv)
    print(f"\n  Auto-tuning {scenario} ({drivetrain}) — {n_combos} kombinasi ...")

    for Np, Q_xy, Rv in itertools.product(
            config.TUNE_Np, config.TUNE_Q_xy, config.TUNE_Rv):
        Nc     = max(3, Np // 3)
        params = dict(Np=Np, Nc=Nc, Q_xy=Q_xy, Q_th=config.MPC_Q_th,
                      Rv=Rv, Rw=config.MPC_Rw)

        r = run_simulation(scenario, drivetrain, terrain_grid, T_grid,
                           path_xy, path_s, path_x, path_y, path_th,
                           mpc_params=params)

        rmse  = float(np.sqrt(np.mean(r.pos_err ** 2)))
        mono  = _lyapunov_monotone_pct(r.V_lyap)
        crit_a = r.reached_goal
        crit_b = rmse <= p_ref_rmse * 1.5   # sedikit toleransi
        crit_c = mono >= 55.0

        n_ok  = sum([crit_a, crit_b, crit_c])
        score = n_ok * 1000 - rmse * 10 + mono

        print(f"    Np={Np:2d} Qxy={Q_xy:4.0f} Rv={Rv:.3f} | "
              f"{'OK' if crit_a else '--'} {'OK' if crit_b else '--'} "
              f"{'OK' if crit_c else '--'} | "
              f"RMSE={rmse:.4f} mono={mono:.0f}%")

        if score > best_score:
            best_score  = score
            best_params = params.copy()

    print(f"  Parameter terbaik: {best_params}")
    return best_params


# ══════════════════════════════════════════════════════════════════════════════
# Metrik
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(r: RunResult) -> dict:
    n = len(r.t)
    if n == 0:
        return {}

    rmse = float(np.sqrt(np.mean(r.pos_err ** 2)))
    mae  = float(np.mean(np.abs(r.pos_err)))
    dist = float(np.sum(np.hypot(np.diff(r.px), np.diff(r.py)))) if n > 1 else 0.0
    a_arr = np.abs(np.diff(r.v)) / config.Ts if n > 1 else np.array([0.0])
    mono  = _lyapunov_monotone_pct(r.V_lyap)

    v_per_ter = {}
    for ter in config.TERRAIN_ORDER:
        pts = [r.v[k] for k, t in enumerate(r.ter_list) if t == ter]
        v_per_ter[ter] = round(float(np.mean(pts)), 4) if pts else None

    return {
        "Waktu (s)":       round(float(r.t[-1]), 2),
        "Jarak (m)":       round(dist, 2),
        "v_avg (m/s)":     round(float(np.mean(r.v)), 4),
        "RMSE path (m)":   round(rmse, 5),
        "MAE path (m)":    round(mae, 5),
        "|a|_avg (m/s²)":  round(float(np.mean(a_arr)), 4),
        "Risk events":     r.risk_events,
        "Risk (%)":        round(100.0 * r.risk_events / max(n, 1), 2),
        "Violations":      r.viols,
        "Violation (%)":   round(100.0 * r.viols / max(n, 1), 2),
        "T_avg":           round(float(np.mean(r.T_arr)), 4),
        "T_min":           round(float(np.min(r.T_arr)), 4),
        "V_lyap_0":        round(float(r.V_lyap[0]), 4) if n else 0,
        "V_lyap_f":        round(float(r.V_lyap[-1]), 4) if n else 0,
        "Lyap mono (%)":   round(mono, 1),
        "Path frac (%)":   round(r.path_frac * 100, 1),
        "Goal reached":    "YES" if r.reached_goal else "NO",
        "Terrain covered": "/".join(sorted(set(r.ter_list))),
        **{f"v_{ter}": (str(v_per_ter[ter]) if v_per_ter[ter] is not None else "N/A")
           for ter in config.TERRAIN_ORDER},
    }


def print_metrics_table(all_metrics: dict) -> None:
    labels = list(all_metrics.keys())
    keys   = list(list(all_metrics.values())[0].keys())
    col_w  = 22
    n_col  = len(labels)
    line   = "=" * (24 + col_w * n_col)
    print(f"\n{line}")
    print("  TABEL METRIK LENGKAP — 6 RUN SIMULASI")
    print(line)
    print(f"  {'Metrik':<24}" + "".join(f"{lb:>{col_w}}" for lb in labels))
    print("-" * (24 + col_w * n_col))
    for k in keys:
        row = f"  {k:<24}" + "".join(
            f"{str(all_metrics[lb].get(k, '-')):>{col_w}}" for lb in labels)
        print(row)
    print(line)


def verify_all_reached_goal(results: list) -> bool:
    """
    Verifikasi semua run mencapai goal.
    HARUS True sebelum membuat grafik (sesuai aturan integritas).
    """
    print("\n  Status goal per run:")
    all_ok = True
    for r in results:
        ok = r.reached_goal
        if not ok:
            all_ok = False
        print(f"    {r.label:<30} frac={r.path_frac*100:.0f}% "
              f"goal={'OK' if ok else 'FAIL'}")
    return all_ok


def verify_terrain_coverage(results: list) -> None:
    """Verifikasi setiap run melewati kelima terrain."""
    print("\n  Coverage terrain per run:")
    for r in results:
        ter_set = set(r.ter_list)
        missing = [t for t in config.TERRAIN_ORDER if t not in ter_set]
        status  = "OK" if not missing else f"MISSING: {missing}"
        print(f"    {r.label:<30} {status}")


def diagnose_mpc_failure(results: list) -> None:
    """Cetak diagnosis jika ada MPC yang gagal (tidak mencapai goal)."""
    failed = [r for r in results if not r.reached_goal]
    if not failed:
        return
    print("\n  [DIAGNOSIS] Run yang gagal mencapai goal:")
    for r in failed:
        mono = _lyapunov_monotone_pct(r.V_lyap)
        v_avg = np.mean(r.v) if len(r.v) else 0
        print(f"    {r.label}:")
        print(f"      v_avg={v_avg:.3f}  Lyap mono={mono:.0f}%  "
              f"path_frac={r.path_frac*100:.0f}%")
        print(f"      V_0={r.V_lyap[0]:.3f}  V_f={r.V_lyap[-1]:.3f}")
        if v_avg < 0.3:
            print("      PENYEBAB: v terlalu rendah — naikkan Q_xy / turunkan Rv")
        if mono < 50:
            print("      PENYEBAB: Lyapunov tidak monoton — periksa lookahead")
