"""
Print MPS availability diagnostic and the device get_device() would select.
Usage (from repo root): PYTHONPATH=. python scripts/diagnose_mps.py
"""
from src.utils import diagnose_mps, get_device

if __name__ == "__main__":
    diagnose_mps()
    device = get_device()
    print(f"[device] Selected device: {device}")
