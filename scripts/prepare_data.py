"""Download / read the raw data once and cache per-record features as .npz files.

VitalDB (development):
    wget -r -N -c -np https://physionet.org/files/vitaldb-arrhythmia/1.0.0/   # annotations (3.4 MB)
    python scripts/prepare_data.py vitaldb --ann physionet.org/files/vitaldb-arrhythmia/1.0.0/Annotation_Files
    (waveforms are streamed from VitalDB by the `vitaldb` package)

Vilnius (external test 1, open version v3, https://doi.org/10.5281/zenodo.23187681):
    python scripts/inspect_vilnius.py /path/to/001_ECG.mat /path/to/001_PPG.mat   # check field names
    edit VILNIUS_KEYS in ecgppg/data.py, then
    python scripts/prepare_data.py vilnius --root /path/to/vilnius
"""
import argparse
import glob
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ecgppg.data import load_vitaldb_case, list_vilnius, load_vilnius_days  # noqa: E402
from ecgppg.pipeline import save_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["vitaldb", "vilnius"])
    ap.add_argument("--ann", help="VitalDB Annotation_Files directory")
    ap.add_argument("--root", help="Vilnius directory with XX_ECG.mat / XX_PPG.mat")
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()
    out = os.path.join(args.cache, args.dataset)
    os.makedirs(out, exist_ok=True)
    exf = os.path.join(out, "excluded.txt")
    skip = set()
    if os.path.exists(exf):   # cases already found without PPG are not downloaded again
        skip = {ln.split("\t")[0] for ln in open(exf).read().splitlines() if "no PPG" in ln}
    log = open(exf, "a")
    if args.dataset == "vitaldb":
        files = sorted(glob.glob(os.path.join(args.ann, "Annotation_file_*.csv")))
        for k, f in enumerate(files):
            cid = int(os.path.basename(f).split("_")[-1].split(".")[0])
            dst = os.path.join(out, f"vdb{cid}.npz")
            if os.path.exists(dst) or str(cid) in skip:
                continue
            try:
                r = load_vitaldb_case(cid, args.ann)
                if r is None:
                    log.write(f"{cid}\tno PPG or too few beats\n")
                    continue
                save_record(r, dst)
                print(f"[{k + 1}/{len(files)}] case {cid}: {len(r['t'])} beats")
            except Exception as e:  # noqa: BLE001
                log.write(f"{cid}\terror {e!r}\n")
                traceback.print_exc()
    else:
        for ecg_path, ppg_paths in list_vilnius(args.root):
            for r in load_vilnius_days(ecg_path, ppg_paths[0] if ppg_paths else None):
                save_record(r, os.path.join(out, f"{r['id']}.npz"))
                print(r["id"], len(r["t"]), "beats")


if __name__ == "__main__":
    main()
