"""Rende importabile il package rtplots quando gli script sono lanciati per path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
