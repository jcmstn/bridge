#!/usr/bin/env python3
"""
GPIB Self-Test Script for macOS with NI-488.2 and pyvisa-py.
Tries all NI library candidates until gpib_ctypes works, then lists resources.
"""

import sys
import platform
from pathlib import Path

try:
    from gpib_ctypes.gpib import gpib
    import pyvisa
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install gpib-ctypes pyvisa-py")
    sys.exit(1)

def is_macos():
    return platform.system() == "Darwin"

def gpib_library_candidates():
    """All possible NI-488.2 library locations."""
    base_paths = [
        Path("/Library/Frameworks/NI4882.framework"),
        Path("/Library/Frameworks/NI4882.framework/Versions/Current"),
        Path("/Library/Frameworks/NI4882.framework/Versions/Current/Libraries"),
        Path("/Library/Frameworks/NIGPIB.framework"),
        Path("/Library/Frameworks/NI4882.framework/Libraries"),
    ]

    candidates = []
    for base in base_paths:
        if base.exists():
            # Bare ni4882, dylibs, common names
            for name in ["ni4882", "NIGPIB", "libni4882.dylib", "ni4882.dylib"]:
                lib = base / name
                if lib.exists():
                    candidates.append(lib)

    # Also search recursively in NI4882
    ni4882 = Path("/Library/Frameworks/NI4882.framework")
    if ni4882.exists():
        for lib in ni4882.rglob("*ni4882*"):
            if lib.is_file() and (lib.suffix in {".dylib", ""} or "ni4882" in lib.name):
                candidates.append(lib)

    return list(set(candidates))  # Dedupe

def test_gpib_lib(lib_path):
    """Try loading lib and basic gpib_ctypes test."""
    try:
        gpib._load_lib(str(lib_path))
        ud = gpib.find(0)  # Test board 0
        if ud >= 0:
            status = gpib.status(ud)
            return True, f"✓ Works! handle={ud}, status=0x{status:02x}"
        return False, "Board find failed"
    except Exception:
        return False, "Load/test failed"

def main():
    print("=== GPIB Self-Test (pyvisa-py + NI-488.2) ===")

    if not is_macos():
        print("Not macOS—skipping.")
        sys.exit(1)

    candidates = gpib_library_candidates()
    if not candidates:
        print("✗ No NI GPIB libraries found. Install NI-488.2.")
        sys.exit(1)

    print(f"Found {len(candidates)} candidate libraries:")
    for i, lib in enumerate(candidates, 1):
        print(f"  {i}. {lib}")

    # Step 1: Try each until one works
    working_lib = None
    for lib in candidates:
        print(f"\nTrying {lib}...")
        ok, msg = test_gpib_lib(lib)
        print(f"  {msg}")
        if ok:
            working_lib = lib
            break

    if not working_lib:
        print("\n✗ No working NI library found.")
        sys.exit(1)

    print(f"\n🎯 Using: {working_lib}")

    # Step 2: Full gpib_ctypes test with working lib
    ud = gpib.find(0)
    listeners = [pad for pad in range(30) if gpib.listener(ud, pad)]
    print(f"Board 0 status: 0x{gpib.status(ud):02x}")
    print(f"GPIB listeners: {listeners}")

    # Step 3: pyvisa GPIB resources
    try:
        rm = pyvisa.ResourceManager('@py')
        all_resources = rm.list_resources()
        gpib_resources = [r for r in all_resources if 'GPIB' in r.upper()]

        print("\nAll resources:", all_resources)
        print("GPIB resources:", gpib_resources)

        status = "✓ GPIB ready!" if gpib_resources else "ℹ No devices (normal)"
        print(status)

    except Exception as e:
        print(f"✗ pyvisa failed: {e}")
        sys.exit(1)

    print("\n🎉 Full stack verified!")

if __name__ == "__main__":
    main()
