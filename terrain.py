"""
terrain.py — Peta terrain 2D (blob acak) + formula skor traversabilitas 6-langkah.

Semua skor T DIHITUNG dari formula; tidak ada nilai hardcode.
"""

import numpy as np
import config


# ══════════════════════════════════════════════════════════════════════════════
# Formula traversabilitas 6-langkah (terverifikasi)
# ══════════════════════════════════════════════════════════════════════════════

def compute_T(terrain_label: str, drivetrain: str) -> float:
    """
    Langkah 1: depth_eff  = W_DEPTH * depth_norm + (1-W_DEPTH) * depth_base
               (sim: depth_norm = depth_base sebagai proxy, tanpa sensor)
    Langkah 2: width_norm = min(SIM_WIDTH_M / WIDTH_MAX, 1.0)
    Langkah 3: cost       = W_DEPTH * depth_eff + W_WIDTH * width_norm
    Langkah 4: T_raw      = 1.0 - cost / capability
    Langkah 5: T          = clamp(T_raw, T_CLAMP_MIN, T_CLAMP_MAX)
    """
    if terrain_label == "obstacle":
        return config.T_OBSTACLE

    db         = config.DEPTH_BASE[terrain_label]
    depth_eff  = config.W_DEPTH * db + (1.0 - config.W_DEPTH) * db   # = db
    width_norm = min(config.SIM_WIDTH_M / config.WIDTH_MAX, 1.0)
    cost       = config.W_DEPTH * depth_eff + config.W_WIDTH * width_norm
    cap        = config.CAPABILITY[drivetrain]
    T_raw      = 1.0 - cost / cap
    return float(np.clip(T_raw, config.T_CLAMP_MIN, config.T_CLAMP_MAX))


def build_T_table() -> dict:
    """Precomputed T_table[terrain][drivetrain] — semua kelas terrain + obstacle."""
    table = {}
    for ter in config.TERRAIN_ORDER:
        table[ter] = {dt: compute_T(ter, dt) for dt in config.CAPABILITY}
    table["obstacle"] = {dt: config.T_OBSTACLE for dt in config.CAPABILITY}
    return table


def print_T_table(T_table: dict) -> None:
    dts  = list(config.CAPABILITY)
    line = "=" * 62
    print(f"\n{line}")
    print("  TABEL SKOR TRAVERSABILITAS T (formula 6-langkah)")
    print(line)
    header = f"  {'Terrain':<10} {'depth_base':>12}" + "".join(
        f"  {f'T({dt})':>10}" for dt in dts)
    print(header)
    print("-" * 62)
    for ter in config.TERRAIN_ORDER:
        db  = config.DEPTH_BASE[ter]
        row = f"  {ter:<10} {db:>12.4f}"
        for dt in dts:
            row += f"  {T_table[ter][dt]:>10.4f}"
        print(row)
    print("-" * 62)
    row = f"  {'obstacle':<10} {'---':>12}"
    for dt in dts:
        row += f"  {config.T_OBSTACLE:>10.4f}"
    print(row)
    print(line)
    print("\n  Verifikasi: T(4WD) > T(2WD) untuk setiap terrain?")
    for ter in config.TERRAIN_ORDER:
        ok = T_table[ter]["4WD"] > T_table[ter]["2WD"]
        print(f"    {ter:<10}: {'OK' if ok else 'FAIL'}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Konversi koordinat grid ↔ dunia
# ══════════════════════════════════════════════════════════════════════════════

def grid_to_world(row: int, col: int) -> tuple:
    """Grid cell center → world (x, y) meter. x ke kanan, y ke bawah."""
    return (col * config.CELL_SIZE, row * config.CELL_SIZE)


def world_to_grid(x: float, y: float) -> tuple:
    """World (x, y) → grid (row, col). Clamp ke dalam batas grid."""
    col = int(np.clip(round(x / config.CELL_SIZE), 0, config.NCOLS - 1))
    row = int(np.clip(round(y / config.CELL_SIZE), 0, config.NROWS - 1))
    return (row, col)


def get_terrain_at_world(x: float, y: float, terrain_grid: np.ndarray) -> str:
    row, col = world_to_grid(x, y)
    return str(terrain_grid[row, col])


def get_T_at_world(x: float, y: float, T_grid: np.ndarray) -> float:
    row, col = world_to_grid(x, y)
    return float(T_grid[row, col])


# ══════════════════════════════════════════════════════════════════════════════
# Pembangkit peta terrain 2D (blob acak)
# ══════════════════════════════════════════════════════════════════════════════

def build_terrain_grid(terrain_scale: float = 1.0) -> np.ndarray:
    """
    Buat peta terrain 80×60 sel dengan blob acak (seed=42).

    terrain_scale: skala radius blob (1.0 = default; >1.0 = lebih besar/sulit)
    Obstacle: kotak kecil (T=0.01, tidak dapat dilalui A*)
    """
    rng  = np.random.default_rng(config.BLOB_SEED)
    grid = np.full((config.NROWS, config.NCOLS), "grass", dtype=object)

    # ── Tempatkan blob terrain ─────────────────────────────────────────────
    for terrain, n_blobs, r_min, r_max in config.BLOB_SPECS:
        for _ in range(n_blobs):
            cx  = rng.integers(8, config.NCOLS - 8)
            cy  = rng.integers(6, config.NROWS - 6)
            r   = int(round(rng.integers(r_min, r_max + 1) * terrain_scale))
            r   = max(2, r)
            _paint_blob(grid, cy, cx, r, terrain)

    # ── Tempatkan obstacle (kotak kecil) ───────────────────────────────────
    margin_start = 6   # jaga area sekitar start/goal bebas obstacle
    for _ in range(config.N_OBSTACLES):
        cx = rng.integers(8, config.NCOLS - 8)
        cy = rng.integers(6, config.NROWS - 6)
        w  = rng.integers(2, 5)
        h  = rng.integers(2, 4)
        # Skip jika terlalu dekat start atau goal
        if (_near_start_goal(cx, cy, margin_start)):
            continue
        r0 = max(0, cy - h); r1 = min(config.NROWS, cy + h + 1)
        c0 = max(0, cx - w); c1 = min(config.NCOLS, cx + w + 1)
        grid[r0:r1, c0:c1] = "obstacle"

    # ── Pastikan sel start & goal bukan obstacle ───────────────────────────
    _clear_area(grid, config.START_ROW, config.START_COL, radius=3)
    _clear_area(grid, config.GOAL_ROW,  config.GOAL_COL,  radius=3)

    return grid


def _paint_blob(grid: np.ndarray, cy: int, cx: int, r: int, label: str) -> None:
    """Lukis blob bundar radius r di (cy, cx)."""
    for row in range(max(0, cy - r), min(config.NROWS, cy + r + 1)):
        for col in range(max(0, cx - r), min(config.NCOLS, cx + r + 1)):
            if np.hypot(col - cx, row - cy) <= r:
                grid[row, col] = label


def _near_start_goal(cx: int, cy: int, margin: int) -> bool:
    """True jika (cy, cx) terlalu dekat start atau goal."""
    near_start = (abs(cx - config.START_COL) < margin and
                  abs(cy - config.START_ROW) < margin)
    near_goal  = (abs(cx - config.GOAL_COL) < margin and
                  abs(cy - config.GOAL_ROW) < margin)
    return near_start or near_goal


def _clear_area(grid: np.ndarray, row: int, col: int, radius: int) -> None:
    """Reset area persegi sekitar (row, col) menjadi grass."""
    r0 = max(0, row - radius); r1 = min(config.NROWS, row + radius + 1)
    c0 = max(0, col - radius); c1 = min(config.NCOLS, col + radius + 1)
    for r in range(r0, r1):
        for c in range(c0, c1):
            if grid[r, c] == "obstacle":
                grid[r, c] = "grass"


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap skor T untuk seluruh grid
# ══════════════════════════════════════════════════════════════════════════════

def build_T_grid(terrain_grid: np.ndarray, drivetrain: str,
                 T_table: dict = None) -> np.ndarray:
    """Hitung T untuk setiap sel, kembalikan array float 2D."""
    if T_table is None:
        T_table = build_T_table()
    T_grid = np.zeros((config.NROWS, config.NCOLS), dtype=float)
    for r in range(config.NROWS):
        for c in range(config.NCOLS):
            lbl        = str(terrain_grid[r, c])
            T_grid[r, c] = T_table[lbl][drivetrain]
    return T_grid


def terrain_color_image(terrain_grid: np.ndarray) -> np.ndarray:
    """Konversi label terrain → array RGB float [0,1] untuk imshow."""
    import matplotlib.colors as mc
    img = np.zeros((config.NROWS, config.NCOLS, 3), dtype=float)
    color_map = config.TERRAIN_COLORS
    for r in range(config.NROWS):
        for c in range(config.NCOLS):
            lbl = str(terrain_grid[r, c])
            hex_col = color_map.get(lbl, "#ffffff")
            img[r, c] = mc.to_rgb(hex_col)
    return img


def verify_terrain_coverage(terrain_grid: np.ndarray) -> bool:
    """Pastikan kelima kelas terrain + obstacle ada di peta."""
    labels = set(terrain_grid.flatten())
    ok = True
    for ter in config.TERRAIN_ORDER:
        present = ter in labels
        print(f"    {ter:<10}: {'OK' if present else 'MISSING'}")
        if not present:
            ok = False
    obs = "obstacle" in labels
    print(f"    {'obstacle':<10}: {'OK' if obs else 'MISSING'}")
    return ok
