"""Central settings for the fraud detection study."""
from pathlib import Path

SEED = 42
DATA_PATH = Path("data/creditcard.csv")
FAST = False
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20
N_BOOTSTRAP = 1_000
ALERT_BUDGET = 100
