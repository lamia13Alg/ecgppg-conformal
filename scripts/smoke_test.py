"""End-to-end smoke test on synthetic recordings (checks that the code runs; numbers are meaningless).

    python scripts/smoke_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ecgppg.synthetic import synth_record  # noqa: E402
from ecgppg.experiments import run_vitaldb_protocol, split_patients  # noqa: E402


def main():
    t0 = time.time()
    recs = {}
    for k in range(24):
        recs[f"syn{k:02d}"] = synth_record(f"syn{k:02d}", minutes=14, seed=k,
                                           p_v=0.01 + 0.004 * (k % 6), p_s=0.01 + 0.004 * (k % 4))
    tr, va, te = split_patients(list(recs), seed=42)
    ext = {"synext": ([synth_record(f"ext{k}", minutes=14, seed=100 + k, p_v=0.03) for k in range(4)], 3, False)}
    out, models, sel = run_vitaldb_protocol([recs[i] for i in tr], [recs[i] for i in va],
                                            [recs[i] for i in te], "results_smoke", tag="smoke",
                                            externals=ext)
    print("selected:", sel)
    print("macros written:", len(out.macros))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
