"""Shared locations for figure inputs and generated outputs."""
from pathlib import Path


FIGURE_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = FIGURE_ROOT / "inputs"
OUTPUT_DIR = FIGURE_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
