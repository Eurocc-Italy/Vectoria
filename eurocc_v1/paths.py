from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR.parent / "data"
LOGS_DIR = ROOT_DIR.parent / "logs"

OUT_DIR = ROOT_DIR.parent / "output"
OUT_DIR.mkdir(exist_ok=True)