"""Download the Vilnius database (Zenodo 10.5281/zenodo.23187681) one patient at a time with
parallel HTTP range requests, extract features for every monitoring day, and delete the raw
files, so that the 63.7 GB never sit on disk together. Resumable: patients already processed
(listed in cache/vilnius/done.txt) are skipped, and partially downloaded files are completed.

    python scripts/prepare_vilnius_stream.py --cache cache --raw /tmp/raw_vilnius --hours 10
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RECORD = "23187681"
API = f"https://zenodo.org/api/records/{RECORD}"
UA = {"User-Agent": "ecgppg-conformal/1.0 (research; contact via GitHub)"}


def list_files():
    with urllib.request.urlopen(urllib.request.Request(API, headers=UA), timeout=60) as r:
        meta = json.load(r)
    files = meta.get("files", [])
    if isinstance(files, dict):
        files = files.get("entries", [])
        files = list(files.values()) if isinstance(files, dict) else files
    out = {}
    for f in files:
        key = f.get("key") or f.get("filename")
        out[key] = {"size": int(f.get("size") or 0),
                    "url": f"https://zenodo.org/records/{RECORD}/files/{key}?download=1"}
    return out


def _get_range(url, a, b, tries=12):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise IOError(f"range not supported (status {r.status})")
                return r.read()
        except Exception as e:  # noqa: BLE001
            if "range not supported" in str(e):
                raise
            time.sleep(min(60, 2 ** k))
    raise IOError(f"failed range {a}-{b}")


def download(url, size, dst, n_threads=6, chunk=32 * 2 ** 20):
    """Parallel ranged download with resume (completed chunks listed in dst + '.done')."""
    if os.path.exists(dst) and os.path.getsize(dst) == size and not os.path.exists(dst + ".done"):
        return
    done_path = dst + ".done"
    done = set()
    if os.path.exists(done_path):
        done = {int(x) for x in open(done_path).read().split()}
    if not os.path.exists(dst) or os.path.getsize(dst) != size:
        with open(dst, "wb") as fh:
            fh.truncate(size)
        done = set()
        open(done_path, "w").close()
    todo = [i for i in range(0, size, chunk) if i not in done]
    lock = threading.Lock()
    t0 = time.time()
    got = [0]
    errors = []

    def worker():
        fd = os.open(dst, os.O_WRONLY)
        try:
            while True:
                with lock:
                    if not todo or errors:
                        return
                    a = todo.pop()
                b = min(size, a + chunk) - 1
                try:
                    data = _get_range(url, a, b)
                except Exception as e:  # noqa: BLE001
                    errors.append(e)
                    return
                os.pwrite(fd, data, a)
                with lock:
                    with open(done_path, "a") as fh:
                        fh.write(f"{a}\n")
                    got[0] += len(data)
        finally:
            os.close(fd)

    th = [threading.Thread(target=worker) for _ in range(n_threads)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    if errors and "range not supported" in str(errors[0]):
        # fall back on a single resumable stream
        import subprocess
        os.remove(done_path)
        os.remove(dst)
        subprocess.run(["wget", "-q", "-c", "-O", dst, url], check=True)
        return
    if errors:
        raise errors[0]
    os.remove(done_path)
    dt = time.time() - t0
    print(f"  {os.path.basename(dst)}: {size / 1e6:.0f} MB in {dt / 60:.1f} min "
          f"({got[0] / 1e6 / max(dt, 1):.1f} MB/s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--raw", default="/tmp/raw_vilnius")
    ap.add_argument("--hours", type=float, default=10.0, help="stop starting new patients after this")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--only", default=None, help="comma-separated patient ids")
    args = ap.parse_args()
    t_start = time.time()
    from ecgppg.data import load_vilnius_days
    from ecgppg.pipeline import save_record
    files = list_files()
    pids = sorted({k.split("_")[0] for k in files if k.endswith(".mat")})
    if args.only:
        pids = [p for p in pids if p in args.only.split(",")]
    out = os.path.join(args.cache, "vilnius")
    os.makedirs(out, exist_ok=True)
    os.makedirs(args.raw, exist_ok=True)
    done_file = os.path.join(out, "done.txt")
    done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()
    print(f"{len(pids)} patients, {len(done)} already processed", flush=True)
    for pid in pids:
        if pid in done:
            continue
        if (time.time() - t_start) / 3600 > args.hours:
            print("time budget reached: rerun to continue", flush=True)
            break
        e, p = f"{pid}_ECG.mat", f"{pid}_PPG.mat"
        if e not in files:
            continue
        for attempt in range(4):
            try:
                for k in [e, p]:
                    if k in files:
                        download(files[k]["url"], files[k]["size"], os.path.join(args.raw, k),
                                 n_threads=args.threads)
                n = 0
                for r in load_vilnius_days(os.path.join(args.raw, e),
                                           os.path.join(args.raw, p) if p in files else None):
                    save_record(r, os.path.join(out, f"{r['id']}.npz"))
                    n += 1
                with open(done_file, "a") as fh:
                    fh.write(pid + "\n")
                print(f"patient {pid}: {n} day records  [{(time.time() - t_start) / 3600:.1f} h elapsed]",
                      flush=True)
                for k in [e, p]:
                    pth = os.path.join(args.raw, k)
                    if os.path.exists(pth):
                        os.remove(pth)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"patient {pid}: attempt {attempt + 1} failed ({exc!r}); retrying", flush=True)
                time.sleep(90 * (attempt + 1))
        else:
            print(f"patient {pid}: FAILED (will be retried on next run)", flush=True)


if __name__ == "__main__":
    main()
