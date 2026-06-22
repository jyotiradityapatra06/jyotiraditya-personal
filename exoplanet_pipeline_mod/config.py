import os

# Output directory
OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)
PLOT_DIR = os.path.join(OUT, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Known exoplanet systems for demo (TESS TIC IDs + sectors)
DEMO_TARGETS = [
    {"tic": 25155310,  "name": "WASP-18b",        "sector": 2,  "truth": "transit"},
    {"tic": 307210830, "name": "TOI-270b",       "sector": 3,  "truth": "transit"},
    {"tic": 441420236, "name": "TOI-1431b",      "sector": 15, "truth": "transit"},
    {"tic": 100100827, "name": "TIC-EB-blend",   "sector": 1,  "truth": "blend"},
    {"tic": 219114080, "name": "TIC-Eclipse",    "sector": 5,  "truth": "eclipse"},
]

