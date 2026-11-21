# -*- coding: utf-8 -*-
"""
P0 data source provenance supplement (closing the download -> extract provenance chain):
  - SHA-256 and byte counts of the three raw .7z archives
  - byte count and line count of sample_submission.csv (fills fields missing from the P0 audit log)
Writes logs/p0_source_provenance_{run_id}.json (versioned, refuses overwrite).
"""
import json
import os
import platform
import sys

import common as C


def main():
    prov = {
        "run_id": C.run_id("p0src"),
        "timestamp_utc": C.now_utc().isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
    }
    # raw .7z archives (located in the method2 root directory)
    for name in ["train.json.7z", "test.json.7z", "sample_submission.csv.7z"]:
        p = os.path.join(C.ROOT, name)
        prov[f"{name}_sha256"] = C.sha256_of_file(p)
        prov[f"{name}_bytes"] = os.path.getsize(p)
    # extracted sample_submission.csv (fills fields missing from the P0 audit log)
    prov["sample_submission_bytes"] = os.path.getsize(C.SAMPLE_SUB)
    with open(C.SAMPLE_SUB, "r", encoding="utf-8") as f:
        prov["sample_submission_n_lines"] = sum(1 for _ in f)

    out = os.path.join(C.LOGS, f"p0_source_provenance_{prov['run_id']}.json")
    C.save_json_no_overwrite(prov, out)

    print(json.dumps(prov, ensure_ascii=False, indent=2))
    print(f"\n[P0-src] -> {out}")


if __name__ == "__main__":
    main()
