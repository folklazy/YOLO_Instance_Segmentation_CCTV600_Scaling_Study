"""Command-line entry point for the CCTV600 study."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cctv600.cli import main

if __name__ == "__main__":
    main()
