"""Create Phase 2 experiment tracking spreadsheet."""
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = str(PROJECT_DIR / "DATASETS")

wb = openpyxl.Workbook()

# ====== Sheet 1: Experiment Log ======
ws = wb.active
ws.title = "Phase2 Ablation"

# Styles
header_font = Font(bold=True, size=11, color="FFFFFF")
header_fill = PatternFill("solid", fgColor="2F5496")
subheader_fill = PatternFill("solid", fgColor="D6E4F0")
subheader_font = Font(bold=True, size=10)
border = Border(
    left=Side(style='thin'), right=Side(style='thin'),
    top=Side(style='thin'), bottom=Side(style='thin')
)
center = Alignment(horizontal='center', vertical='center', wrap_text=True)
pct_fmt = '0.0%'
time_fmt = 'YYYY-MM-DD HH:MM'

# Column definitions
columns = [
    ("实验ID", 10), ("实验名称", 30), ("TEXT_ENCODER", 16), ("TEXT_LOSS_TYPE", 16),
    ("TEXT_LOSS_WEIGHT", 16), ("I2T_LOSS_WEIGHT", 16), ("Skip Stage1", 12),
    ("Stage1 Epochs", 13), ("Stage2 Epochs", 13),
    ("开始时间", 18), ("结束时间", 18), ("耗时(h)", 10),
    ("mAP", 10), ("Rank-1", 10), ("Rank-5", 10), ("Rank-10", 10),
    ("vs B0 mAP Δ", 12), ("状态", 10), ("备注", 35),
]

# Write title row
ws.merge_cells('A1:S1')
ws['A1'] = 'CLIP-ReID Text-Guided Phase 2 消融实验记录'
ws['A1'].font = Font(bold=True, size=14, color="2F5496")
ws['A1'].alignment = Alignment(horizontal='center')

# Category row
ws.merge_cells('A2:B2')
ws['A2'] = '基本信息'
ws['A2'].fill = PatternFill("solid", fgColor="BDD7EE")
ws['A2'].font = subheader_font
ws['A2'].alignment = center

ws.merge_cells('C2:F2')
ws['C2'] = '配置参数'
ws['C2'].fill = PatternFill("solid", fgColor="E2EFDA")
ws['C2'].font = subheader_font
ws['C2'].alignment = center

ws.merge_cells('G2:I2')
ws['G2'] = '训练设置'
ws['G2'].fill = PatternFill("solid", fgColor="FCE4D6")
ws['G2'].font = subheader_font
ws['G2'].alignment = center

ws.merge_cells('J2:L2')
ws['J2'] = '时间'
ws['J2'].fill = PatternFill("solid", fgColor="FFF2CC")
ws['J2'].font = subheader_font
ws['J2'].alignment = center

ws.merge_cells('M2:P2')
ws['M2'] = '评估指标'
ws['M2'].fill = PatternFill("solid", fgColor="D9E2F3")
ws['M2'].font = subheader_font
ws['M2'].alignment = center

ws.merge_cells('Q2:S2')
ws['Q2'] = '分析'
ws['Q2'].fill = PatternFill("solid", fgColor="EDEDED")
ws['Q2'].font = subheader_font
ws['Q2'].alignment = center

# Header row (row 3)
for col_idx, (name, width) in enumerate(columns, 1):
    cell = ws.cell(row=3, column=col_idx, value=name)
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = center
    cell.border = border
    ws.column_dimensions[get_column_letter(col_idx)].width = width

# Pre-fill experiment configs (rows 4-12)
experiments = [
    # (ID, Name, TEXT_ENCODER, TEXT_LOSS_TYPE, TEXT_LOSS_WEIGHT, I2T_LOSS_WEIGHT, skip_s1, s1_ep, s2_ep, notes)
    ("B0", "对照基线 (clip_native)", "clip_native", "none", 1.0, 1.0, "No", 120, 60, "Stage1+Stage2 全量训练"),
    ("B2", "+CrossModal contrastive", "clip_native", "contrastive", 0.5, 1.0, "Yes (B0)", 0, 60, "复用B0的Stage1 checkpoint"),
    ("B3a", "I2T weight=0.2", "clip_native", "none", 1.0, 0.2, "Yes (B0)", 0, 60, "复用B0的Stage1 checkpoint"),
    ("B3b", "I2T weight=0.5", "clip_native", "none", 1.0, 0.5, "Yes (B0)", 0, 60, "复用B0的Stage1 checkpoint"),
    ("B3c", "I2T weight=1.0 (default)", "clip_native", "none", 1.0, 1.0, "Yes (B0)", 0, 60, "复用B0的Stage1, 同B0但Stage2重训"),
    ("B3d", "I2T weight=2.0", "clip_native", "none", 1.0, 2.0, "Yes (B0)", 0, 60, "复用B0的Stage1 checkpoint"),
    ("B1", "TextQueryEncoder", "query_encoder", "none", 1.0, 1.0, "No", 120, 60, "Stage1+Stage2 全量训练 (query_encoder)"),
    ("B4", "最优组合 (TBD)", "—", "—", "—", "—", "—", "—", "—", "根据B0-B3结果确定"),
]

data_fill_even = PatternFill("solid", fgColor="F2F2F2")
for row_idx, exp in enumerate(experiments, 4):
    fill = data_fill_even if row_idx % 2 == 0 else None
    for col_idx, val in enumerate(exp, 1):
        cell = ws.cell(row=row_idx, column=col_idx, value=val)
        cell.alignment = center
        cell.border = border
        if fill:
            cell.fill = fill
    # mAP delta formula (column Q = col 17): =M{row}-M$4
    if row_idx > 4:
        ws.cell(row=row_idx, column=17, value=f'=M{row_idx}-M$4').border = border
        ws.cell(row=row_idx, column=17).alignment = center
        ws.cell(row=row_idx, column=17).number_format = '0.0%'
        if fill:
            ws.cell(row=row_idx, column=17).fill = fill
    
    # Duration formula (column L = col 12): =(K-J)*24
    ws.cell(row=row_idx, column=12, value=f'=IF(AND(K{row_idx}<>"",J{row_idx}<>""),(K{row_idx}-J{row_idx})*24,"")').border = border
    ws.cell(row=row_idx, column=12).alignment = center
    ws.cell(row=row_idx, column=12).number_format = '0.0'
    if fill:
        ws.cell(row=row_idx, column=12).fill = fill
    
    # Status column
    ws.cell(row=row_idx, column=18, value="待运行").border = border
    ws.cell(row=row_idx, column=18).alignment = center
    if fill:
        ws.cell(row=row_idx, column=18).fill = fill

# Format percentage columns
for row_idx in range(4, 12):
    for col in [13, 14, 15, 16]:
        cell = ws.cell(row=row_idx, column=col)
        cell.number_format = pct_fmt
        cell.alignment = center
        cell.border = border

# Freeze panes
ws.freeze_panes = 'A4'

# ====== Sheet 2: Historical Results ======
ws2 = wb.create_sheet("历史结果参考")

ws2['A1'] = '已有实验结果参考（Market-1501）'
ws2['A1'].font = Font(bold=True, size=13, color="2F5496")
ws2.merge_cells('A1:G1')

hist_headers = ["配置", "mAP", "Rank-1", "Rank-5", "Rank-10", "来源", "备注"]
hist_widths = [30, 10, 10, 10, 10, 25, 30]
for col_idx, (h, w) in enumerate(zip(hist_headers, hist_widths), 1):
    cell = ws2.cell(row=2, column=col_idx, value=h)
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = center
    cell.border = border
    ws2.column_dimensions[get_column_letter(col_idx)].width = w

hist_data = [
    ("Baseline CLIP-ReID (train eval)", 0.896, 0.952, 0.984, 0.990, "train.log 01-12", "SIZE_TRAIN=256×128"),
    ("Baseline CLIP-ReID (test)", 0.882, 0.945, 0.983, 0.990, "test_20260113", "SIZE_TEST=384×128"),
    ("SIE+OLP Baseline", 0.896, 0.953, 0.983, 0.989, "test_20260120", "SIE_COE=3.0, Stride=12"),
    ("Text-Guided (best, eval2)", 0.880, 0.943, 0.976, 0.985, "eval2.log 01-26", "Stage2 ep60, clip_native"),
    ("Text-Guided (eval_stage2)", 0.845, 0.934, 0.978, 0.989, "eval_stage2.log 01-27", "不同测试配置"),
    ("Text-Guided (baseline cfg)", 0.825, 0.922, 0.973, 0.985, "eval_stage2_baseline 01-27", "baseline测试配置"),
    ("Phase1 Dry-run (2ep, clip_native+contrastive)", 0.543, 0.754, None, None, "Phase1验证 dry-run", "仅2ep用于验证管线"),
    ("Phase1 Dry-run (1ep, query_encoder+combined)", 0.429, 0.673, None, None, "Phase1验证 dry-run", "仅1ep用于验证管线"),
    ("Phase1 Dry-run (1+1ep, query_encoder+contrastive)", 0.499, 0.717, None, None, "Phase1验证 dry-run", "仅1+1ep S1+S2"),
]

for row_idx, row_data in enumerate(hist_data, 3):
    for col_idx, val in enumerate(row_data, 1):
        cell = ws2.cell(row=row_idx, column=col_idx, value=val if val is not None else "—")
        cell.alignment = center
        cell.border = border
        if col_idx in [2, 3, 4, 5] and isinstance(val, float):
            cell.number_format = pct_fmt

ws2.freeze_panes = 'A3'

# ====== Sheet 3: Common Config ======
ws3 = wb.create_sheet("统一配置")
ws3['A1'] = 'Phase 2 统一实验配置'
ws3['A1'].font = Font(bold=True, size=13, color="2F5496")

config_items = [
    ("数据集", "Market-1501"),
    ("数据路径", DEFAULT_DATA_ROOT),
    ("标注文件", "annotations/market1501_train.json"),
    ("模型", "ViT-B-16"),
    ("SIZE_TRAIN", "[256, 128]"),
    ("SIZE_TEST", "[256, 128]"),
    ("SEED", "1234"),
    ("Stage1 Batch Size", "64"),
    ("Stage2 Batch Size", "64"),
    ("Stage1 LR", "0.00035"),
    ("Stage2 LR", "0.000005"),
    ("Stage1 Epochs", "120"),
    ("Stage2 Epochs", "60"),
    ("ID_LOSS_WEIGHT", "0.25"),
    ("TRIPLET_LOSS_WEIGHT", "1.0"),
    ("I2T_LOSS_WEIGHT", "1.0 (default, varies in B3)"),
    ("TEXT_TEMPERATURE", "0.07"),
    ("GPU", "本地 GPU"),
    ("Python 环境", "miniconda3 base, PyTorch 1.10+cu113"),
    ("创建时间", datetime.now().strftime("%Y-%m-%d %H:%M")),
]

for row_idx, (key, val) in enumerate(config_items, 3):
    ws3.cell(row=row_idx, column=1, value=key).font = Font(bold=True)
    ws3.cell(row=row_idx, column=2, value=val)
    ws3.column_dimensions['A'].width = 22
    ws3.column_dimensions['B'].width = 45

# Save
output_path = PROJECT_DIR / "docs/experiments/phase2_ablation_results.xlsx"
output_path.parent.mkdir(parents=True, exist_ok=True)
wb.save(output_path)
print(f"Saved to {output_path}")
