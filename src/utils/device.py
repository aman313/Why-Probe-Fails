"""Device selection: prefer CUDA, then MPS (Apple Silicon), then CPU."""
import platform
from typing import Optional

import torch

_logged: Optional[str] = None
_diagnosed: bool = False


def diagnose_mps() -> None:
    """Print why MPS is or isn't available; on Darwin with MPS unavailable, explain root cause."""
    global _diagnosed
    _diagnosed = True
    print("[device] MPS diagnostic:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  platform: {platform.system()} {platform.machine()}")
    # macOS version (root cause for is_available() often depends on this)
    if platform.system() == "Darwin":
        try:
            import subprocess
            out = subprocess.run(
                ["sw_vers", "-productVersion"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                print(f"  macOS: {out.stdout.strip()} (MPS requires 12.3+)")
        except Exception:
            pass
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        print("  torch.backends.mps: not present")
        print("  Root cause: PyTorch build has no MPS backend.")
        return
    is_built = mps_backend.is_built()
    is_available = mps_backend.is_available()
    print(f"  MPS is_built(): {is_built}")
    print(f"  MPS is_available(): {is_available}")
    if platform.system() != "Darwin":
        return
    if is_built and is_available:
        print("  MPS is available; get_device() will use it when CUDA is not.")
        return
    print("  MPS not available on this Mac.")
    if not is_built:
        print("  Root cause: PyTorch was not built with MPS (CPU-only or wrong-arch wheel, e.g. x86 under Rosetta).")
        print("  Fix: Reinstall for Mac ARM: pip install torch (from a native arm64 shell).")
    else:
        print("  Root cause: is_available() is False — PyTorch checks Metal/runtime (macOS 12.3+, Metal device, Xcode CLI).")
        print("  Fix: Ensure macOS >= 12.3, run native arm64 (not Rosetta), and run: xcode-select --install")


def get_device() -> torch.device:
    """Return the best available device: cuda > mps > cpu. Logs choice on first call."""
    global _logged, _diagnosed
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        if platform.system() == "Darwin" and not _diagnosed:
            _diagnosed = True
            diagnose_mps()
    choice = str(device)
    if _logged != choice:
        print(f"[device] Using device: {choice}")
        _logged = choice
    return device
