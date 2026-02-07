"""Parse experiment output logs and update the Phase 2 tracking spreadsheet."""
import argparse
import os
import re
import glob
from datetime import datetime, timedelta
from openpyxl import load_workbook


def parse_metrics_from_log(output_dir):
    """Extract mAP, Rank-1/5/10 from the most recent log files in output_dir."""
    metrics = {}
    
    # Search for log files that contain evaluation results
    log_patterns = [
        os.path.join(output_dir, "*.log"),
        os.path.join(output_dir, "*.txt"),
    ]
    
    log_files = []
    for pattern in log_patterns:
        log_files.extend(glob.glob(pattern))
    
    # Also check if train_text_reid writes to a log file via logger
    # The logger typically writes to the output_dir with name "transreid"
    
    best_metrics = {}
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
        except:
            continue
        
        # Pattern: "mAP: XX.X%" or "mAP:  XX.X%"
        map_matches = re.findall(r'mAP:\s*([\d.]+)%', content)
        r1_matches = re.findall(r'(?:CMC curve|Rank-1\s*)\s*:\s*([\d.]+)%|Rank-1\s*[:,=]\s*([\d.]+)', content)
        r5_matches = re.findall(r'Rank-5\s*[:,=]\s*([\d.]+)', content)
        r10_matches = re.findall(r'Rank-10\s*[:,=]\s*([\d.]+)', content)
        
        # Also try the common transreid log format
        map_matches2 = re.findall(r'mAP\s*[:=]\s*([\d.]+)', content)
        
        if map_matches:
            best_metrics['mAP'] = float(map_matches[-1]) / 100.0
        elif map_matches2:
            val = float(map_matches2[-1])
            best_metrics['mAP'] = val / 100.0 if val > 1 else val
        
        for m in r1_matches:
            val = m[0] if m[0] else m[1]
            if val:
                v = float(val)
                best_metrics['R1'] = v / 100.0 if v > 1 else v
        
        if r5_matches:
            v = float(r5_matches[-1])
            best_metrics['R5'] = v / 100.0 if v > 1 else v
        
        if r10_matches:
            v = float(r10_matches[-1])
            best_metrics['R10'] = v / 100.0 if v > 1 else v
    
    return best_metrics


def parse_timestamps_from_log(output_dir):
    """Extract start and end timestamps from log files."""
    log_files = glob.glob(os.path.join(output_dir, "*.log"))
    
    all_times = []
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # Match common timestamp formats: "2026-01-28 14:30:05" or "01/28 14:30:05"
                    m = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                    if m:
                        try:
                            t = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                            all_times.append(t)
                        except:
                            pass
        except:
            continue
    
    if all_times:
        return min(all_times), max(all_times)
    
    # Fallback: use file modification times
    try:
        files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
        if files:
            mtimes = [os.path.getmtime(f) for f in files if os.path.isfile(f)]
            if mtimes:
                return (datetime.fromtimestamp(min(mtimes)), 
                        datetime.fromtimestamp(max(mtimes)))
    except:
        pass
    
    return None, None


# Experiment ID → row mapping in xlsx (row 4 = B0, row 5 = B2, etc.)
EXP_ROW_MAP = {
    "B0": 4,
    "B2": 5,
    "B3a": 6,
    "B3b": 7,
    "B3c": 8,
    "B3d": 9,
    "B1": 10,
    "B4": 11,
}


def update_xlsx(xlsx_path, exp_id, status, output_dir):
    wb = load_workbook(xlsx_path)
    ws = wb["Phase2 Ablation"]
    
    row = EXP_ROW_MAP.get(exp_id)
    if row is None:
        print(f"Unknown experiment ID: {exp_id}")
        return
    
    # Column mapping:
    # J=10: start time, K=11: end time, L=12: duration
    # M=13: mAP, N=14: R1, O=15: R5, P=16: R10
    # R=18: status
    
    if status == "running":
        ws.cell(row=row, column=10, value=datetime.now())
        ws.cell(row=row, column=10).number_format = 'YYYY-MM-DD HH:MM'
        ws.cell(row=row, column=18, value="运行中")
    
    elif status == "completed":
        # Parse metrics
        if output_dir and os.path.isdir(output_dir):
            metrics = parse_metrics_from_log(output_dir)
            start_time, end_time = parse_timestamps_from_log(output_dir)
            
            if end_time:
                ws.cell(row=row, column=11, value=end_time)
                ws.cell(row=row, column=11).number_format = 'YYYY-MM-DD HH:MM'
            else:
                ws.cell(row=row, column=11, value=datetime.now())
                ws.cell(row=row, column=11).number_format = 'YYYY-MM-DD HH:MM'
            
            if metrics.get('mAP') is not None:
                ws.cell(row=row, column=13, value=metrics['mAP'])
                ws.cell(row=row, column=13).number_format = '0.0%'
            if metrics.get('R1') is not None:
                ws.cell(row=row, column=14, value=metrics['R1'])
                ws.cell(row=row, column=14).number_format = '0.0%'
            if metrics.get('R5') is not None:
                ws.cell(row=row, column=15, value=metrics['R5'])
                ws.cell(row=row, column=15).number_format = '0.0%'
            if metrics.get('R10') is not None:
                ws.cell(row=row, column=16, value=metrics['R10'])
                ws.cell(row=row, column=16).number_format = '0.0%'
        
        ws.cell(row=row, column=18, value="已完成")
    
    elif status == "failed":
        ws.cell(row=row, column=18, value="失败")
    
    wb.save(xlsx_path)
    print(f"Updated {exp_id} → {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx_path", required=True)
    parser.add_argument("--exp_id", required=True)
    parser.add_argument("--status", required=True, choices=["running", "completed", "failed"])
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    
    update_xlsx(args.xlsx_path, args.exp_id, args.status, args.output_dir)
