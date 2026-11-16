"""Hinge stress prediction package."""

from pathlib import Path

# Keep existing imports and python -m commands working after the file move.
__path__.append(str(Path(__file__).with_name("prediction")))
