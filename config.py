"""
config.py — Semua parameter simulasi UGV 2D grid + A*.
Satu-satunya file yang perlu diedit untuk tuning/eksperimen.
"""

import numpy as np

# ── Grid 2D ──────────────────────────────────────────────────────────────────
NROWS      = 60          # baris grid
NCOLS      = 80          # kolom grid
CELL_SIZE  = 0.5         # meter per sel

# ── Koordinat dunia (meter) ───────────────────────────────────────────────────
# Konvensi: x = col * CELL_SIZE (kanan), y = row * CELL_SIZE (bawah)
START_WORLD = (2.0, 15.0)   # (x, y) titik awal
GOAL_WORLD  = (38.0, 15.0)  # (x, y) titik tujuan
GOAL_RADIUS = 1.5            # m — jarak ke goal = "tercapai"

# Konversi ke indeks grid
START_COL = round(START_WORLD[0] / CELL_SIZE)   # 4
START_ROW = round(START_WORLD[1] / CELL_SIZE)   # 30
GOAL_COL  = round(GOAL_WORLD[0] / CELL_SIZE)    # 76
GOAL_ROW  = round(GOAL_WORLD[1] / CELL_SIZE)    # 30

# ── Parameter terrain ────────────────────────────────────────────────────────
TERRAIN_ORDER = ["grass", "sand", "puddle", "rubble", "mud"]

DEPTH_BASE = {
    "grass":  0.02,
    "sand":   0.04,
    "puddle": 0.15,
    "rubble": 0.20,
    "mud":    0.35,
}

CAPABILITY = {
    "2WD": 0.50,
    "4WD": 0.90,
}

W_DEPTH     = 0.60
W_WIDTH     = 0.40
WIDTH_MAX   = 3.0
SIM_WIDTH_M = 1.5    # proxy lebar jalur representatif (tanpa sensor)

T_CLAMP_MIN  = 0.01
T_CLAMP_MAX  = 1.0
T_OBSTACLE   = 0.01  # traversabilitas sel obstacle

# ── Blob generation (seed=42, reproducible) ───────────────────────────────────
# Format: (terrain_label, n_blobs, r_min_cell, r_max_cell)
BLOB_SEED  = 42
BLOB_SPECS = [
    ("sand",   3,  5,  9),
    ("puddle", 2,  4,  8),
    ("rubble", 3,  5, 10),
    ("mud",    2,  6,  9),
]
N_OBSTACLES = 8    # kotak obstacle kecil (sel tak dapat dilalui)

# ── Dinamika kendaraan ────────────────────────────────────────────────────────
Ts        = 0.1          # s — langkah waktu
V_MAX     = 2.0          # m/s
OMEGA_MAX = np.pi / 4    # rad/s
A_MAX     = 1.0          # m/s²
NOISE_SIGMA = 0.01       # m — sigma noise proses posisi
NOISE_SEED  = 0          # seed RNG noise proses (reproduktibilitas)

# ── P-Controller ──────────────────────────────────────────────────────────────
KP_OMEGA  = 3.0    # P-gain heading
LOOKAHEAD = 1.5    # m — jarak lookahead sepanjang A*-path

# ── MPC — parameter awal ──────────────────────────────────────────────────────
MPC_Np   = 15
MPC_Nc   = 6
MPC_Q_xy = 20.0   # penalti error posisi (ex²+ey²)
MPC_Q_th = 2.0    # penalti error heading
MPC_Rv   = 0.05   # penalti effort v
MPC_Rw   = 0.10   # penalti effort omega

# ── Auto-tuning grid (dicoba jika MPC gagal konvergen) ───────────────────────
TUNE_Np   = [10, 15, 20]
TUNE_Q_xy = [20, 30, 45]
TUNE_Rv   = [0.02, 0.05]

# ── Kondisi awal — offset dari A*-path ───────────────────────────────────────
# Menciptakan V_lyap_0 > 0 yang bermakna agar analisis Lyapunov non-trivial.
INIT_PERP_OFFSET  = 0.5    # m perpendikular terhadap heading awal path
INIT_THETA_OFFSET = 0.2    # rad

# ── Simulasi ──────────────────────────────────────────────────────────────────
MAX_STEPS    = 3500   # langkah maksimum per run (2WD mud perlu lebih banyak)
PROGRESS_MIN = 0.90   # fraksi path minimal sebelum dianggap konvergen

# ── Risk & constraint ─────────────────────────────────────────────────────────
T_RISK       = 0.30   # T < T_RISK → risk event
VIOL_TOL     = 0.05   # toleransi v > v_max*T sebelum dihitung violation

# ── Path smoothing ────────────────────────────────────────────────────────────
PATH_SMOOTH_SIGMA  = 1.5   # sigma Gaussian smoothing (cell units)
PATH_RESAMPLE_DS   = 0.25  # m — jarak antar waypoint setelah resampling

# ── Output ────────────────────────────────────────────────────────────────────
DPI = 180   # resolusi PNG

TERRAIN_COLORS = {
    "grass":    "#3db54a",
    "sand":     "#d4b96a",
    "puddle":   "#2777d4",
    "rubble":   "#8c8c8c",
    "mud":      "#7a4a1e",
    "obstacle": "#1a0a00",
}

DRIVETRAIN_COLORS = {
    "2WD": "#E05C2A",
    "4WD": "#2A7BE0",
}

SCENARIO_LINESTYLE = {
    "P":    "-",
    "MPC":  "--",
    "MPCT": "-.",
}

SCENARIO_LABEL = {
    "P":    "P-Controller",
    "MPC":  "MPC (no T)",
    "MPCT": "MPC + T",
}
