"""Verify the toolchain: CUDA availability, sm_120 (Blackwell) kernel support, and that the
optional SB3 and brawl_vision stacks import cleanly. See README.md for the install trap this
guards against.
"""
import sys
import time

import torch


def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")

    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")

    if not cuda_available:
        print("CUDA is not available at all. Cannot check sm_120 support.")
        return 1

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    print(f"torch.cuda.get_device_name(0): {device_name}")
    print(f"torch.cuda.get_device_capability(0): {capability}")
    print(f"torch.cuda.get_arch_list(): {arch_list}")

    sm_120_present = "sm_120" in arch_list
    print(f"'sm_120' in arch_list: {sm_120_present}")

    if not sm_120_present:
        print(
            "\nFAIL: this torch build has no sm_120 kernels. On a Blackwell (RTX 50-series) "
            "GPU this will fail at runtime with 'no kernel image is available for execution "
            "on the device', even though is_available() returned True.\n"
            "Reinstall with: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu128 (or a newer cu12x/cu13x channel)."
        )
        return 1

    device = torch.device("cuda")

    # Time a 4096x4096 matmul.
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    matmul_s = time.perf_counter() - t0
    print(f"4096x4096 matmul: {matmul_s * 1000:.2f} ms")
    del a, b, c

    # Time a large index_put_(accumulate=True), representative of the sim's damage-scatter path.
    n = 20_000_000
    target = torch.zeros(1_000_000, device=device)
    idx = torch.randint(0, target.shape[0], (n,), device=device)
    src = torch.randn(n, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    target.index_put_((idx,), src, accumulate=True)
    torch.cuda.synchronize()
    scatter_s = time.perf_counter() - t0
    print(f"index_put_(accumulate=True) over {n:,} elements: {scatter_s * 1000:.2f} ms")
    del target, idx, src

    # SB3 stack is optional; report but don't fail the check on it.
    try:
        import stable_baselines3
        import sb3_contrib

        print(f"stable_baselines3: {stable_baselines3.__version__}")
        print(f"sb3_contrib: {sb3_contrib.__version__}")
    except ImportError as exc:
        print(f"SB3 stack not importable ({exc}). Install with: pip install .[sb3]")

    # brawl_vision's stack is optional in exactly the same sense as SB3's: report, don't fail.
    # The version numbers are the point. OpenCV 5.x is recent enough that an API this project
    # leans on could still move under it, and a run that half-works is much easier to diagnose
    # with the version printed here than by bisecting behavior.
    try:
        import cv2
        import mss

        print(f"opencv: {cv2.__version__}")
        print(f"mss: {mss.__version__}")
        missing = [name for name in
                   ("findHomography", "warpPerspective", "phaseCorrelate", "estimateAffine2D")
                   if not hasattr(cv2, name)]
        if missing:
            print(f"  WARNING: opencv is missing {missing} -- brawl_vision Phases C/D/F need them.")
    except ImportError as exc:
        print(f"vision stack not importable ({exc}). Install with: pip install .[vision]")

    print("\nOK: sm_120 kernels present, matmul and scatter ran successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
