"""
dashboard.py — Dashboard interaktif UGV Traversability-Aware Speed Control.
Jalankan: streamlit run dashboard.py  (dari folder ugv_sim_2d/)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
import terrain as trn
import pathplanning as pp
import simulate as sim

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    layout="wide",
    page_title="UGV Traversability Dashboard",
    page_icon="🚗",
)

st.title("UGV Off-Road — Traversability-Aware Speed Control")

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Parameter Simulasi")

    v_max_val = st.slider(
        "v_max (m/s)", min_value=0.5, max_value=3.0, value=2.0, step=0.1)
    drivetrain = st.radio(
        "Drivetrain", options=["2WD", "4WD"], index=1)
    terrain_scale_pct = st.slider(
        "Skala Terrain (%)", min_value=50, max_value=150, value=100, step=5)
    scenario_choice = st.selectbox(
        "Skenario Kontrol",
        options=["P-Controller", "MPC (no T)", "MPC + T"], index=0)

    st.divider()

    run_btn = st.button("▶  Jalankan", type="primary", width="stretch")

    st.divider()
    st.markdown("**Kecepatan Batas per Terrain**")
    T_table_display = trn.build_T_table()
    for ter in config.TERRAIN_ORDER:
        T_val = T_table_display[ter][drivetrain]
        st.metric(
            label=f"{ter.capitalize()}  (T = {T_val:.3f})",
            value=f"{v_max_val * T_val:.2f} m/s")

# ══════════════════════════════════════════════════════════════════════════════
# Cache & runner helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_cached_map(scale_pct: int, dt_key: str):
    scale = scale_pct / 100.0
    tg_   = trn.build_terrain_grid(terrain_scale=scale)
    ttab_ = trn.build_T_table()
    Tg_   = trn.build_T_grid(tg_, dt_key, ttab_)
    pxy_, ps_, px_, py_, pth_ = pp.full_pipeline(
        Tg_, (config.START_ROW, config.START_COL),
              (config.GOAL_ROW,  config.GOAL_COL))
    return tg_, Tg_, ttab_, pxy_, ps_, px_, py_, pth_


def _run(scenario_key, dt_key, vmax, scale_pct):
    orig = config.V_MAX
    config.V_MAX = vmax
    try:
        tg, Tg, _, pxy, ps, px, py, pth = build_cached_map(scale_pct, dt_key)
        r = sim.run_simulation(scenario_key, dt_key, tg, Tg, pxy, ps, px, py, pth)
    finally:
        config.V_MAX = orig
    return r, tg, Tg, pxy, ps

# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_MAP = {"P-Controller": "P", "MPC (no T)": "MPC", "MPC + T": "MPCT"}
SCENARIO_LABELS = {"P": "P-Controller", "MPC": "MPC (no T)", "MPCT": "MPC + T"}
scenario_key = SCENARIO_MAP[scenario_choice]

for k, v in [("result", None), ("tg", None), ("Tg", None),
             ("pxy", None), ("path_s", None), ("last_params", None),
             ("all_results", None)]:
    st.session_state.setdefault(k, v)

current_params = (drivetrain, v_max_val, terrain_scale_pct)

# Run on button press — jalankan semua 3 skenario sekaligus
if run_btn:
    with st.spinner("Menghitung 3 skenario (P / MPC / MPC+T)..."):
        try:
            all_r = {}
            tg_ = Tg_ = pxy_ = ps_ = None
            for sc in ["P", "MPC", "MPCT"]:
                r, tg_, Tg_, pxy_, ps_ = _run(
                    sc, drivetrain, v_max_val, terrain_scale_pct)
                all_r[sc] = r
            st.session_state.update(
                result=all_r[scenario_key],
                all_results=all_r,
                tg=tg_, Tg=Tg_, pxy=pxy_, path_s=ps_,
                last_params=current_params)
        except Exception as e:
            st.error(str(e))
            st.stop()

# Always load map (cached, instant)
if st.session_state.tg is None:
    tg_, Tg_, _, pxy_, ps_, *_ = build_cached_map(terrain_scale_pct, drivetrain)
    st.session_state.update(tg=tg_, Tg=Tg_, pxy=pxy_, path_s=ps_)

tg         = st.session_state.tg
Tg         = st.session_state.Tg
pxy        = st.session_state.pxy
path_s     = st.session_state.path_s
all_results = st.session_state.all_results
# result selalu dari skenario yang dipilih di sidebar
result = all_results[scenario_key] if all_results else st.session_state.result

sx, sy = config.START_WORLD
gx, gy = config.GOAL_WORLD
ext    = [0, config.NCOLS * config.CELL_SIZE,
          config.NROWS * config.CELL_SIZE, 0]

# ══════════════════════════════════════════════════════════════════════════════
# Row 1 — Terrain map + T heatmap
# ══════════════════════════════════════════════════════════════════════════════

col_map, col_heat = st.columns([3, 2])

with col_map:
    st.subheader("Peta Terrain & Trajektori")
    fig_m, ax_m = plt.subplots(figsize=(8, 5))
    ax_m.imshow(trn.terrain_color_image(tg),
                extent=ext, aspect="equal", origin="upper", alpha=0.9)

    xs = [p[0] for p in pxy]; ys = [p[1] for p in pxy]
    ax_m.plot(xs, ys, "k--", lw=2, alpha=0.85, label="Lintasan referensi", zorder=5)

    if result is not None:
        col_r = config.DRIVETRAIN_COLORS[drivetrain]
        ax_m.plot(result.px, result.py,
                  color=col_r, ls=config.SCENARIO_LINESTYLE[scenario_key],
                  lw=2.2, label=scenario_choice, alpha=0.9, zorder=6)
        ax_m.plot(result.px[-1], result.py[-1], "o", color=col_r, ms=8, zorder=7)

    ax_m.plot(sx, sy, "g^", ms=12, zorder=8, label="Start")
    ax_m.plot(gx, gy, "r*", ms=14, zorder=8, label="Goal")

    patches = [mpatches.Patch(color=config.TERRAIN_COLORS[t],
                               label=t.capitalize(), alpha=0.7)
               for t in config.TERRAIN_ORDER]
    patches.append(mpatches.Patch(color=config.TERRAIN_COLORS["obstacle"],
                                   label="Obstacle"))
    leg1 = ax_m.legend(handles=patches, fontsize=7, loc="lower left",
                        framealpha=0.85, ncol=2)
    ax_m.add_artist(leg1)
    ax_m.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_m.set_xlabel("x (m)"); ax_m.set_ylabel("y (m)")
    ax_m.set_title(f"{drivetrain}  |  {scenario_choice}  |  Terrain scale {terrain_scale_pct}%")
    ax_m.invert_yaxis()
    ax_m.grid(alpha=0.15, color="white")
    plt.tight_layout()
    st.pyplot(fig_m, width="stretch")
    plt.close(fig_m)

with col_heat:
    st.subheader(f"Heatmap Traversabilitas — {drivetrain}")
    fig_h, ax_h = plt.subplots(figsize=(6, 5))
    im = ax_h.imshow(Tg, extent=ext, cmap="RdYlGn",
                      vmin=0.0, vmax=1.0, aspect="equal", origin="upper")
    plt.colorbar(im, ax=ax_h, label="T score", shrink=0.85)
    ax_h.plot(xs, ys, "w--", lw=1.8, alpha=0.85, label="Lintasan referensi")
    ax_h.plot(sx, sy, "g^", ms=10, zorder=8)
    ax_h.plot(gx, gy, "r*", ms=12, zorder=8)
    if result is not None:
        ax_h.plot(result.px, result.py, color="white", ls=":", lw=1.5, alpha=0.7)
    ax_h.set_title("Hijau = traversabilitas tinggi")
    ax_h.invert_yaxis()
    ax_h.grid(alpha=0.1, color="white")
    plt.tight_layout()
    st.pyplot(fig_h, width="stretch")
    plt.close(fig_h)

# ── Info lintasan (selalu tampil setelah peta dimuat) ─────────────────────────
path_len_m  = float(path_s[-1]) if path_s is not None else 0.0
n_waypoints = len(pxy) if pxy is not None else 0

pi1, pi2 = st.columns(2)
pi1.metric("Panjang Lintasan", f"{path_len_m:.1f} m")
pi2.metric("Jumlah Waypoint",  str(n_waypoints))

# ══════════════════════════════════════════════════════════════════════════════
# Row 2 — Metrics + charts (only when result available)
# ══════════════════════════════════════════════════════════════════════════════

st.divider()

if result is not None:
    metrics = sim.compute_metrics(result)

    # ── Metric cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = st.columns(10)
    c1.metric("Goal",           "YES" if result.reached_goal else "NO")
    c2.metric("Waktu (s)",      metrics["Waktu (s)"])
    c3.metric("v_avg (m/s)",    metrics["v_avg (m/s)"])
    c4.metric("RMSE (m)",       metrics["RMSE path (m)"])
    c5.metric("Violations",     metrics["Violations"])
    c6.metric("|a|_avg (m/s²)", metrics["|a|_avg (m/s²)"])
    c7.metric("Risk events",    metrics["Risk events"])
    c8.metric("V_lyap 0",       metrics["V_lyap_0"])
    c9.metric("V_lyap f",       metrics["V_lyap_f"])
    c10.metric("Lyap mono %",   metrics["Lyap mono (%)"])

    # ── Terrain speed comparison — semua 3 skenario ──────────────────────────
    st.subheader("Perbandingan Kecepatan per Terrain")
    T_tab = trn.build_T_table()

    if all_results:
        all_metrics = {sc: sim.compute_metrics(r) for sc, r in all_results.items()}
        cmp_rows = []
        for ter in config.TERRAIN_ORDER:
            T_v = T_tab[ter][drivetrain]
            vb  = round(v_max_val * T_v, 3)
            row = {
                "Terrain":       ter.capitalize(),
                "T score":       round(T_v, 4),
                "v_batas (m/s)": vb,
            }
            for sc in ["P", "MPC", "MPCT"]:
                raw = all_metrics[sc].get(f"v_{ter}", None)
                row[SCENARIO_LABELS[sc]] = (
                    round(float(raw), 4) if raw not in (None, "N/A") else None)
            cmp_rows.append(row)

        df_cmp = pd.DataFrame(cmp_rows)
        st.dataframe(df_cmp, hide_index=True, width="stretch")

        # Bar chart perbandingan
        fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4))
        ter_labels  = [r["Terrain"] for r in cmp_rows]
        v_batas_arr = [r["v_batas (m/s)"] for r in cmp_rows]
        sc_colors   = {"P-Controller": "#4C72B0", "MPC (no T)": "#DD8452", "MPC + T": "#55A868"}
        n_sc = 3
        bar_w = 0.22
        x = np.arange(len(ter_labels))
        for i, sc_lbl in enumerate(["P-Controller", "MPC (no T)", "MPC + T"]):
            vals = [r[sc_lbl] if r[sc_lbl] is not None else 0 for r in cmp_rows]
            ax_cmp.bar(x + (i - 1) * bar_w, vals, bar_w,
                       label=sc_lbl, color=sc_colors[sc_lbl], alpha=0.85)
        ax_cmp.step(np.append(x - 1.5 * bar_w, x[-1] + 1.5 * bar_w),
                    v_batas_arr + [v_batas_arr[-1]],
                    where="post", color="red", lw=1.5, ls="--", label="v_batas")
        ax_cmp.set_xticks(x); ax_cmp.set_xticklabels(ter_labels)
        ax_cmp.set_ylabel("v_aktual rata-rata (m/s)")
        ax_cmp.set_title(f"Kecepatan aktual per terrain — {drivetrain}  |  v_max={v_max_val} m/s")
        ax_cmp.legend(fontsize=8); ax_cmp.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig_cmp, width="stretch")
        plt.close(fig_cmp)
    else:
        # Sebelum run: tampilkan tabel batas saja
        rows = []
        for ter in config.TERRAIN_ORDER:
            T_v = T_tab[ter][drivetrain]
            raw = metrics.get(f"v_{ter}", None)
            rows.append({
                "Terrain":        ter.capitalize(),
                "T score":        round(T_v, 4),
                "v_batas (m/s)":  round(v_max_val * T_v, 3),
                "v_aktual (m/s)": float(raw) if raw not in (None, "N/A") else None,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # ── Speed + Lyapunov plots ────────────────────────────────────────────────
    col_v, col_lp = st.columns(2)
    col_r = config.DRIVETRAIN_COLORS[drivetrain]

    with col_v:
        st.subheader("Profil Kecepatan v(t)")
        fig_v, ax_v = plt.subplots(figsize=(6, 4))
        ax_v.plot(result.t, result.v, color=col_r, lw=2, alpha=0.9)
        ax_v.axhline(v_max_val, color="k", lw=0.8, ls="--", alpha=0.5,
                     label=f"v_max = {v_max_val} m/s")
        ax_v.fill_between(result.t, 0, result.T_arr * v_max_val,
                          alpha=0.15, color="seagreen", label="v_max · T")
        ax_v.set_xlabel("Waktu (s)"); ax_v.set_ylabel("v (m/s)")
        ax_v.legend(fontsize=8); ax_v.grid(alpha=0.22)
        plt.tight_layout()
        st.pyplot(fig_v, width="stretch")
        plt.close(fig_v)

    with col_lp:
        st.subheader("Lyapunov V(e)")
        fig_l, ax_l = plt.subplots(figsize=(6, 4))
        ax_l.plot(result.t, result.V_lyap, color=col_r, lw=2, alpha=0.9)
        ax_l.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax_l.set_yscale("symlog", linthresh=0.01)
        ax_l.set_xlabel("Waktu (s)"); ax_l.set_ylabel("V(e)")
        ax_l.set_title("V(e) = Q_xy·(ex²+ey²) + Q_θ·eθ²")
        ax_l.grid(alpha=0.22)
        plt.tight_layout()
        st.pyplot(fig_l, width="stretch")
        plt.close(fig_l)

    # ── Trajectory replay ─────────────────────────────────────────────────────
    st.subheader("Tayangan Ulang Lintasan")
    n_frames  = len(result.px)
    frame_idx = st.slider("Frame", 0, n_frames - 1, 0,
                           step=max(1, n_frames // 50), label_visibility="collapsed")

    fig_a, ax_a = plt.subplots(figsize=(11, 5))
    ax_a.imshow(trn.terrain_color_image(tg),
                extent=ext, aspect="equal", origin="upper", alpha=0.85)
    ax_a.plot(xs, ys, "k--", lw=1.5, alpha=0.7, label="Lintasan referensi")
    ax_a.plot(result.px[:frame_idx+1], result.py[:frame_idx+1],
              color=col_r, lw=2.0, alpha=0.9)

    rx, ry, rth = result.px[frame_idx], result.py[frame_idx], result.theta[frame_idx]
    ax_a.plot(rx, ry, "o", color=col_r, ms=10, zorder=8)
    ax_a.annotate("",
        xy=(rx + 1.5*np.cos(rth), ry + 1.5*np.sin(rth)), xytext=(rx, ry),
        arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax_a.plot(sx, sy, "g^", ms=10, zorder=9)
    ax_a.plot(gx, gy, "r*", ms=12, zorder=9)
    ax_a.set_title(
        f"t = {result.t[frame_idx]:.1f} s  |  v = {result.v[frame_idx]:.2f} m/s  |  "
        f"{scenario_choice} ({drivetrain})")
    ax_a.invert_yaxis()
    ax_a.grid(alpha=0.12, color="white")
    plt.tight_layout()
    st.pyplot(fig_a, width="stretch")
    plt.close(fig_a)

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.caption(
    f"UGV Traversability Simulation  |  Grid {config.NROWS}×{config.NCOLS} sel  |  "
    f"2WD cap={config.CAPABILITY['2WD']}  /  4WD cap={config.CAPABILITY['4WD']}"
)
