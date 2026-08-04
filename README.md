# UGV Traversability-Aware Speed Control — Streamlit Dashboard

Interactive simulation of a UGV navigating a 2D off-road terrain grid (grass, sand,
puddle, rubble, mud) using an A*-planned reference path and one of three speed
controllers: P-Controller, MPC (no terrain awareness), and MPC+T
(traversability-aware MPC).

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard.py
```

Then open the URL Streamlit prints (default `http://localhost:8501`).

## Usage

- Pick `v_max`, drivetrain (2WD/4WD), terrain scale, and a control scenario in the
  sidebar.
- Click **Jalankan** to run all three scenarios (P / MPC / MPC+T) and compare
  speed profiles, Lyapunov stability, constraint violations, and trajectory replay.

## Files

| File | Purpose |
|---|---|
| `dashboard.py` | Streamlit UI — entry point |
| `config.py` | All tunable simulation parameters |
| `terrain.py` | Terrain grid generation + traversability (T) scoring |
| `pathplanning.py` | A* planning + path smoothing/resampling |
| `model.py` | Vehicle kinematic model |
| `controller.py` | P-Controller and MPC controllers |
| `simulate.py` | Simulation loop + metrics |

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, and
   set the main file to `dashboard.py`.
