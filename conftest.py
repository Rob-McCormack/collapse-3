import sys
from pathlib import Path

# Ensure the repo root is importable so `collapse3` and `experiments` resolve
# when tests are run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
