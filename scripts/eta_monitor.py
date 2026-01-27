#!/usr/bin/env python3
import argparse
import re
from datetime import datetime
from pathlib import Path

STAGE1_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Stage1 Epoch\[(\d+)\] Iteration\[(\d+)/(\d+)\]")
STAGE2_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Epoch\[(\d+)\] Iteration\[(\d+)/(\d+)\]")


def parse_max_epochs(text, stage_name):
    m = re.search(rf"{stage_name}:.*?MAX_EPOCHS:\s*(\d+)", text, re.S)
    return int(m.group(1)) if m else None


def parse_lines(lines, pattern):
    matches = []
    for line in lines:
        m = pattern.search(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
            epoch = int(m.group(2))
            it = int(m.group(3))
            total = int(m.group(4))
            matches.append((ts, epoch, it, total, line.strip()))
    return matches


def compute_sec_per_iter(matches):
    if len(matches) < 2:
        return None
    t2, e2, i2, tot2, _ = matches[-1]
    t1, e1, i1, tot1, _ = matches[-2]
    if i2 == i1:
        return None
    dt = (t2 - t1).total_seconds()
    di = (i2 - i1)
    if di <= 0:
        return None
    return dt / di


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        return

    lines = log_path.read_text(errors="ignore").splitlines()
    text = "\n".join(lines)

    stage1_max = parse_max_epochs(text, "STAGE1") or 60
    stage2_max = parse_max_epochs(text, "STAGE2") or 60

    stage1_lines = parse_lines(lines, STAGE1_RE)
    stage2_lines = [m for m in parse_lines(lines, STAGE2_RE) if "Stage1" not in m[4]]

    now = datetime.now()

    if stage2_lines:
        stage = "stage2"
        current = stage2_lines[-1]
        sec_per_iter = compute_sec_per_iter(stage2_lines) or compute_sec_per_iter(stage1_lines)
        stage_max = stage2_max
    elif stage1_lines:
        stage = "stage1"
        current = stage1_lines[-1]
        sec_per_iter = compute_sec_per_iter(stage1_lines)
        stage_max = stage1_max
    else:
        Path(args.out).write_text(f"{now:%Y-%m-%d %H:%M:%S} | ETA unavailable (no iteration logs yet)\n")
        return

    _, epoch, it, total, _ = current

    if sec_per_iter is None:
        msg = f"{now:%Y-%m-%d %H:%M:%S} | stage={stage} epoch={epoch} iter={it}/{total} | ETA unavailable (need more logs)"
        Path(args.out).open("a").write(msg + "\n")
        return

    if stage == "stage1":
        remain_stage1 = (total - it) + (stage_max - epoch) * total
        stage2_iters_per_epoch = stage2_lines[-1][3] if stage2_lines else total
        remain_stage2 = stage2_max * stage2_iters_per_epoch
        remain_iters = remain_stage1 + remain_stage2
    else:
        remain_iters = (total - it) + (stage_max - epoch) * total

    eta_seconds = remain_iters * sec_per_iter
    eta_time = now.timestamp() + eta_seconds
    eta_dt = datetime.fromtimestamp(eta_time)
    mins = eta_seconds / 60.0

    msg = (
        f"{now:%Y-%m-%d %H:%M:%S} | stage={stage} epoch={epoch} iter={it}/{total} "
        f"sec/iter={sec_per_iter:.3f} | ETA~{eta_dt:%Y-%m-%d %H:%M:%S} "
        f"(~{mins:.1f} min)"
    )
    Path(args.out).open("a").write(msg + "\n")


if __name__ == "__main__":
    main()
