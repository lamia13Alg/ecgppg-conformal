"""Print the structure of Vilnius MAT files so that VILNIUS_KEYS (ecgppg/data.py) can be set.

    python scripts/inspect_vilnius.py 001_ECG.mat 001_PPG.mat
"""
import sys
import numpy as np


def show(name, v, depth=0):
    pad = "  " * depth
    if hasattr(v, "_fieldnames"):
        print(f"{pad}{name}: struct {v._fieldnames}")
        for f in v._fieldnames:
            show(f, getattr(v, f), depth + 1)
    elif isinstance(v, np.ndarray) and v.dtype == object:
        print(f"{pad}{name}: cell/object array shape {v.shape}")
        for j, el in enumerate(v.ravel()[:3]):
            show(f"{name}[{j}]", el, depth + 1)
    elif isinstance(v, np.ndarray):
        head = v.ravel()[:5]
        print(f"{pad}{name}: array {v.shape} {v.dtype} first={head}")
    else:
        print(f"{pad}{name}: {type(v).__name__} = {str(v)[:80]}")


for path in sys.argv[1:]:
    print("=" * 70, "\n", path)
    try:
        from scipy.io import loadmat
        M = loadmat(path, squeeze_me=True, struct_as_record=False)
        for k, v in M.items():
            if not k.startswith("__"):
                show(k, v)
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            f.visititems(lambda n, o: print(n, getattr(o, "shape", ""), getattr(o, "dtype", "")))
