"""Central configuration: paths, column names, constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

DEFAULT_TRAIN = DATA_DIR / "UNSW_NB15_training-set.csv"
DEFAULT_TEST = DATA_DIR / "UNSW_NB15_testing-set.csv"

RANDOM_STATE = 42

# Columns in the partitioned UNSW-NB15 files
CATEGORICAL = ["proto", "service", "state"]
ID_COLS = ["id"]
LABEL_COL = "label"        # 0 = normal, 1 = attack
ATTACK_COL = "attack_cat"  # Normal + 9 attack families

NORMAL_NAME = "Normal"
BINARY_CLASS_NAMES = ["Normal", "Attack"]
