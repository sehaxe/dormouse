"""Build the unit-economics workbook: what a user costs us and what they pay.

Every input is a number we measured on this project (a 16 GB card, a run that
does N bytes per second, a model that ships as one file), so the model can be
argued with rather than admired. Every derived cell is a formula: change an
input and the margin moves.

    uv run --with openpyxl python scripts/build_business_model.py
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = Path(__file__).resolve().parents[2] / "site" / "business" / "model.xlsx"

INK = "1B1A17"
ACCENT = "2A66C2"
PAPER = "F6F3EC"
MUTED = "6D675D"
HEAD = PatternFill("solid", fgColor="E3DED2")
THIN = Side(style="thin", color="D3CCBC")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def style_header(ws, row: int, cols: int) -> None:
    for c in range(1, cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(bold=True, color=INK)
        cell.fill = HEAD
        cell.border = BOX


def title(ws, text: str) -> None:
    ws["A1"] = text
    ws["A1"].font = Font(size=14, bold=True, color=INK)


def note(ws, cell: str, text: str) -> None:
    ws[cell] = text
    ws[cell].font = Font(size=9, color=MUTED, italic=True)
    ws[cell].alignment = Alignment(wrap_text=True, vertical="top")


def build() -> None:
    wb = Workbook()

    # ---------------------------------------------------------------- inputs
    ws = wb.active
    ws.title = "Inputs"
    title(ws, "Measured inputs (change these, everything else follows)")
    rows = [
        ("Card price, USD", 1000, "one RTX 5070 Ti class card, 16 GB"),
        ("Card power, W", 300, "under load"),
        ("Electricity, USD/kWh", 0.15, "household rate, Minsk"),
        ("Card life, months", 36, "depreciation horizon"),
        ("Training throughput, kB/s", 87.2, "95M dense, b24 s1024, compile — measured"),
        ("Hours per month", 720, "always-on"),
        ("Bytes per training token", 1.0, "byte-level: one byte is one token"),
        ("Pretrain corpus, GB", 13.0, "built 2026-11-14 from 8.4 GB of parquet"),
        ("SFT corpus, MB", 75.9, "190,512 pairs"),
        ("Model file, MB (ternary export)", 19.0, "95M at 1.585 bits/weight"),
        ("Decode speed, bytes/s (batch 1)", 2800, "memory-bound: 190 MB per token at 896 GB/s"),
        ("Engineer cost, USD/month", 1500, "one person, part time"),
    ]
    ws.append([])
    ws.append(["input", "value", "where it comes from"])
    style_header(ws, ws.max_row, 3)
    first = ws.max_row + 1
    for name, value, why in rows:
        ws.append([name, value, why])
    for r in range(first, ws.max_row + 1):
        ws.cell(row=r, column=3).font = Font(size=9, color=MUTED)
        ws.cell(row=r, column=3).alignment = Alignment(wrap_text=True)
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 12
    ws.column_dimensions["C"].width = 62

    # ------------------------------------------------------------- unit cost
    ws = wb.create_sheet("Unit cost")
    title(ws, "What one training run and one served token cost us")
    ws.append([])
    ws.append(["line", "value", "formula / why"])
    style_header(ws, ws.max_row, 3)

    def add(name: str, value, why: str) -> None:
        ws.append([name, value, why])
        r = ws.max_row
        ws.cell(row=r, column=3).font = Font(size=9, color=MUTED)

    add("Card depreciation, USD/hour", "=Inputs!B2/Inputs!B5/Inputs!B6", "price / life / hours")
    add("Electricity, USD/hour", "=Inputs!B3/1000*Inputs!B4", "W/1000 x price")
    add("Machine, USD/hour", "=B4+B5", "depreciation + power")
    add("Pretrain, hours", "=Inputs!B8*1e9/Inputs!B7/3600", "GB / (kB/s) / 3600")
    add("Pretrain, USD", "=B6*B7", "hours x machine rate")
    add("SFT, hours", "=Inputs!B9*1e6/Inputs!B7/3600/10", "SFT runs ~10x faster per byte (no full corpus pass)")
    add("SFT, USD", "=B6*B9", "")
    add("One full model build, USD", "=B8+B10", "pretrain + SFT")
    add("Cost per shipped model file, USD", "=B11", "the file is free to copy; the build is not")
    add("Serving: bytes/s on one card", "=Inputs!B12", "decode is memory-bound, measured ceiling")
    add("Serving: model builds per card-year", "=Inputs!B6*12/B11", "how many builds that card can fund")

    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 54

    # --------------------------------------------------------------- pricing
    ws = wb.create_sheet("Pricing")
    title(ws, "Three ways to sell, priced against what they cost us")
    ws.append([])
    ws.append(["product", "unit", "price, USD", "our cost, USD", "margin"])
    style_header(ws, ws.max_row, 5)

    ws.append(["Weights: open (Apache-2.0), like Qwen and DeepSeek", "free", 0, 0, "=C5-D5"])
    ws.append(["Token API, cheap tier", "per 1M bytes", 0.30, 0, "=C6-D6"])
    ws.append(["Token API, private tier", "per 1M bytes", 1.20, 0, "=C7-D7"])
    ws.append(["On-prem deployment (hospital, ministry)", "per site, one-off", 25000, "=Unit cost!B11", "=C8-D8"])
    ws.append(["Fine-tune on the customer's own data", "per project", 12000, "=Unit cost!B11", "=C9-D9"])
    ws.append(["Support contract", "per site per year", 8000, 0, "=C10-D10"])
    note(ws, "A13", "Weights are free on purpose: Qwen and DeepSeek are Apache/MIT, so a paid licence for the file cannot be defended. What is not free is putting it inside someone's perimeter and making their data work.")
    ws.column_dimensions["A"].width = 44
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 14
    ws.column_dimensions["E"].width = 14

    # ------------------------------------------------------------- scenarios
    ws = wb.create_sheet("Scenarios")
    title(ws, "Break-even: how many of what, when")
    ws.append([])
    ws.append(["scenario", "units / year", "unit price", "revenue", "cost", "profit"])
    style_header(ws, ws.max_row, 6)

    ws.append(["Token API, 1e4 users x 20 MB/yr", 200000, "=Pricing!C6/1e6*20", "=B5*C5", "=B5*Pricing!D6", "=D5-E5"])
    ws.append(["On-prem sites", 4, "=Pricing!C8", "=B6*C6", "=B6*Pricing!D8", "=D6-E6"])
    ws.append(["Fine-tune projects", 6, "=Pricing!C9", "=B7*C7", "=B7*Pricing!D9", "=D7-E7"])
    ws.append(["Support contracts", 4, "=Pricing!C10", "=B8*C8", "=B8*Pricing!D10", "=D8-E8"])
    ws.append(["TOTAL", "", "", "=SUM(D5:D8)", "=SUM(E5:E8)", "=D9-E9"])
    ws.append([])
    ws.append(["Fixed cost: one engineer", "", "", "", "=Inputs!B14*12", ""])
    ws.append(["Result before tax", "", "", "", "", "=F9-F11"])

    for c in range(1, 7):
        ws.column_dimensions[get_column_letter(c)].width = 30 if c == 1 else 18
    for r in (9, 11, 12):
        for c in range(1, 7):
            ws.cell(row=r, column=c).font = Font(bold=True, color=ACCENT if r == 9 else INK)

    # ------------------------------------------------------------ positioning
    ws = wb.create_sheet("Positioning")
    title(ws, "What we are, in the words we actually use")
    lines = [
        ("Name", "dormouse"),
        ("Category", "a language model you own, not a service you rent"),
        ("One line", "Они продают доступ. Мы отдаём файл."),
        ("Who it is for", "organisations that cannot send data out: hospitals, government, factories, and one person who wants it on their laptop"),
        ("What we sell", "a licence for the file, tokens when someone wants an API, and deployment inside the perimeter"),
        ("Why we win", "smallest file, fastest start, works offline, and the training recipe is public"),
        ("Why we might lose", "a 1B-class model does not know as much as a 30B; we sell control, not knowledge"),
        ("Tone", "dry, specific, no adjectives that cannot be measured"),
        ("Never say", "революция, прорыв, лучший в мире, ИИ-платформа"),
    ]
    ws.append([])
    for k, v in lines:
        ws.append([k, v])
        r = ws.max_row
        ws.cell(row=r, column=1).font = Font(bold=True, color=INK)
        ws.cell(row=r, column=2).alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 96

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
