#!/usr/bin/env python3

import argparse
import math
import os
import re

KNOWN_DATASETS = ["market1501", "msmt17", "dukemtmc", "occ_duke", "veri", "vehicleid"]


def _infer_dataset(path):
    lower = path.lower()
    for name in KNOWN_DATASETS:
        if name in lower:
            return name
    return "unknown"


def _infer_model(path):
    lower = path.lower()
    if "rn50" in lower or "resnet" in lower:
        return "RN50"
    if "vit" in lower:
        return "ViT-B-16"
    return "unknown"


def _parse_log(path):
    data = {
        "path": path,
        "dataset": _infer_dataset(path),
        "model": _infer_model(path),
        "sie": None,
        "train_pct": None,
        "train_pct_mode": None,
        "seed": None,
        "map": None,
        "rank1": None,
    }

    section = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.endswith(":") and " " not in s[:-1]:
                section = s[:-1]

            if data["map"] is None:
                m = re.search(r"\bmAP:\s*([0-9.]+)%", line)
                if m:
                    data["map"] = float(m.group(1))
            if data["rank1"] is None:
                m = re.search(r"Rank-1\s*:\s*([0-9.]+)%", line)
                if m:
                    data["rank1"] = float(m.group(1))

            if data["seed"] is None:
                m = re.search(r"\bSEED:\s*([0-9]+)", line)
                if m:
                    data["seed"] = int(m.group(1))

            if data["train_pct"] is None:
                m = re.search(r"\bTRAIN_PCT:\s*([0-9.]+)", line)
                if m:
                    data["train_pct"] = float(m.group(1))
            if data["train_pct_mode"] is None:
                m = re.search(r"\bTRAIN_PCT_MODE:\s*([A-Za-z_]+)", line)
                if m:
                    data["train_pct_mode"] = m.group(1)

            if data["sie"] is None:
                m = re.search(r"\bSIE_CAMERA:\s*(True|False)", line)
                if m:
                    data["sie"] = (m.group(1) == "True")

            if data["dataset"] == "unknown" and section == "DATASETS":
                m = re.search(r"\bNAMES:\s*([A-Za-z0-9_]+)", line)
                if m:
                    data["dataset"] = m.group(1)
            if data["model"] == "unknown" and section == "MODEL":
                m = re.search(r"\bNAME:\s*([A-Za-z0-9\-_]+)", line)
                if m:
                    data["model"] = m.group(1)

    if data["sie"] is None:
        data["sie"] = "sie_olp" in path.lower()
    if data["train_pct"] is None:
        data["train_pct"] = 1.0
    return data


def _mean_std(values):
    n = len(values)
    if n == 0:
        return None, None
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def _format_float(value, ndigits=2):
    if value is None:
        return ""
    return f"{value:.{ndigits}f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize CLIP-ReID test logs")
    parser.add_argument("--output-root", default="/root/autodl-tmp/CLIP_REID/OUTPUT")
    parser.add_argument("--per-run", action="store_true", help="print per-run table")
    parser.add_argument("--summary", action="store_true", help="print grouped summary")
    parser.add_argument("--compare", action="store_true", help="print baseline vs SIE+OLP comparison")
    args = parser.parse_args()

    any_flag = args.per_run or args.summary or args.compare
    per_run = args.per_run or not any_flag
    summary = args.summary or not any_flag
    compare = args.compare or not any_flag

    logs = []
    for root, _, files in os.walk(args.output_root):
        for name in files:
            if name.startswith("test") and name.endswith(".log"):
                logs.append(os.path.join(root, name))

    rows = []
    for path in sorted(logs):
        row = _parse_log(path)
        if row["map"] is None or row["rank1"] is None:
            continue
        rows.append(row)

    if per_run:
        print("dataset\tmodel\tsie\ttrain_pct\ttrain_pct_mode\tseed\tmAP\tRank-1\tlog")
        for r in rows:
            print(
                f"{r['dataset']}\t{r['model']}\t{r['sie']}\t{_format_float(r['train_pct'], 3)}\t"
                f"{r['train_pct_mode'] or ''}\t{r['seed'] or ''}\t{_format_float(r['map'])}\t"
                f"{_format_float(r['rank1'])}\t{r['path']}"
            )

    if summary:
        grouped = {}
        for r in rows:
            key = (r["dataset"], r["model"], r["sie"], r["train_pct"], r["train_pct_mode"])
            grouped.setdefault(key, []).append(r)

        print("\n# summary")
        print("dataset\tmodel\tsie\ttrain_pct\ttrain_pct_mode\tn\tmAP_mean\tmAP_std\tRank-1_mean\tRank-1_std")
        for key in sorted(grouped.keys()):
            entries = grouped[key]
            maps = [e["map"] for e in entries]
            r1s = [e["rank1"] for e in entries]
            m_mean, m_std = _mean_std(maps)
            r_mean, r_std = _mean_std(r1s)
            dataset, model, sie, train_pct, train_pct_mode = key
            print(
                f"{dataset}\t{model}\t{sie}\t{_format_float(train_pct, 3)}\t{train_pct_mode or ''}\t"
                f"{len(entries)}\t{_format_float(m_mean)}\t{_format_float(m_std)}\t"
                f"{_format_float(r_mean)}\t{_format_float(r_std)}"
            )

    if compare:
        grouped = {}
        for r in rows:
            key = (r["dataset"], r["model"], r["train_pct"], r["train_pct_mode"])
            grouped.setdefault((key, r["sie"]), []).append(r)

        print("\n# baseline vs sie_olp")
        print("dataset\tmodel\ttrain_pct\ttrain_pct_mode\tbaseline_n\tsie_n\tbaseline_mAP\tsie_mAP\tΔmAP\tbaseline_R1\tsie_R1\tΔR1")
        keys = sorted({k for k, _ in grouped.keys()})
        for key in keys:
            base = grouped.get((key, False), [])
            sie = grouped.get((key, True), [])
            if not base or not sie:
                continue
            base_maps = [e["map"] for e in base]
            sie_maps = [e["map"] for e in sie]
            base_r1s = [e["rank1"] for e in base]
            sie_r1s = [e["rank1"] for e in sie]
            base_m, _ = _mean_std(base_maps)
            sie_m, _ = _mean_std(sie_maps)
            base_r, _ = _mean_std(base_r1s)
            sie_r, _ = _mean_std(sie_r1s)
            dataset, model, train_pct, train_pct_mode = key
            print(
                f"{dataset}\t{model}\t{_format_float(train_pct, 3)}\t{train_pct_mode or ''}\t"
                f"{len(base)}\t{len(sie)}\t{_format_float(base_m)}\t{_format_float(sie_m)}\t"
                f"{_format_float(sie_m - base_m)}\t{_format_float(base_r)}\t{_format_float(sie_r)}\t"
                f"{_format_float(sie_r - base_r)}"
            )

if __name__ == "__main__":
    main()
