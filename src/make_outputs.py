"""make_outputs.py — assemble figures and tables in section order.

Output: output/Figures_Tables_by_section.docx and output/Tables_1-3.xlsx (with a manuscript-vs-reproduced comparison sheet).
"""
import os, pandas as pd, numpy as np
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from common import OUTPUT, FIG, TAB

CAP = {
 'f1': 'Figure 1. Taxonomy of geographic transfer failures and the corresponding gate steps (B2: covariate shift; '
       'B3–B6: level shift; B7: structural shift).',
 'f2': 'Figure 2. Conceptual framework of deployment-matched validation: the deployment unit fixes the validation unit.',
 'f3': 'Figure 3. Two-phase deployment-matched validation workflow (Phase A: development audit; Phase B: deployment '
       'gate). Dashed boxes are adjustable parameters.',
 'f4': 'Figure 4. Few-shot local calibration curves, Taiwan case (left: mean MAPE across 22 station areas; right: '
       'weighted retraining by station area).',
 't1': 'Table 1. LightGBM performance under validation designs that do and do not match the deployment unit.',
 't2': 'Table 2. Failure diagnostics by held-out station area (Taiwan).',
 't3': 'Table 3. Few-shot calibration, Taiwan (MAPE / R², mean of 22 station areas).'}


def fmt_t1(T):
    out = T.copy()
    for c in out.columns[1:]:
        if 'MAPE' in c:
            out[c] = out[c].map(lambda v: '—' if pd.isna(v) else f'{v:.1f}%')
        else:
            out[c] = out[c].map(lambda v: '—' if pd.isna(v) else f'{v:.2f}'.replace('-', '−'))
    out.columns = ['Validation design', 'Taiwan R²', 'Taiwan MAPE', 'California R²', 'California MAPE']
    return out


def add_table(doc, df, fs=8):
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = 'Table Grid'
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
    for _, r in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(fs)
    for c in t.rows[0].cells:
        for p in c.paragraphs:
            for run in p.runs:
                run.bold = True


def para(doc, text, size=10, bold=False, italic=False):
    p = doc.add_paragraph(); r = p.add_run(text); r.font.size = Pt(size); r.bold = bold; r.italic = italic
    return p


def main():
    t1 = fmt_t1(pd.read_csv(os.path.join(TAB, 'Table1.csv')))
    t2 = pd.read_csv(os.path.join(TAB, 'Table2.csv'))
    t3 = pd.read_csv(os.path.join(TAB, 'Table3.csv'))
    notes = {k: open(os.path.join(TAB, f'{k}_notes.txt')).read().strip() for k in ['Table1', 'Table2', 'Table3']}
    doc = Document()
    st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(10)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2)
    para(doc, 'Figures and tables in section order (reproduced from the analysis code)', 14, bold=True)
    blocks = [('3.2 無標籤診斷之適用界線', [('fig', 'Figure1_taxonomy.png', 'f1', 15)]),
              ('3.4 部署對應驗證與兩階段閘門', [('fig', 'Figure2_framework.png', 'f2', 14),
                                       ('fig', 'Figure3_workflow.png', 'f3', 15)]),
              ('4.1 驗證設計：保留單元決定效能估計', [('tab', t1, 't1', 'Table1')]),
              ('4.2 失效來源：價格水準錨定', [('tab', t2, 't2', 'Table2')]),
              ('4.5 少樣本校準與競爭方法', [('tab', t3, 't3', 'Table3'), ('fig', 'Figure4_calibration.png', 'f4', 16.5)])]
    for head, items in blocks:
        doc.add_heading(head, level=2)
        for it in items:
            if it[0] == 'fig':
                doc.add_picture(os.path.join(FIG, it[1]), width=Cm(it[3]))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                para(doc, CAP[it[2]], 9)
            else:
                para(doc, CAP[it[2]], 9, bold=True)
                add_table(doc, it[1], fs=7.5 if it[2] == 't2' else 8)
                para(doc, 'Reproduction notes. ' + notes[it[3]].replace('\n', ' '), 8, italic=True)
    doc.save(os.path.join(OUTPUT, 'Figures_Tables_by_section.docx'))
    with pd.ExcelWriter(os.path.join(OUTPUT, 'Tables_1-3.xlsx')) as xw:
        t1.to_excel(xw, sheet_name='Table1', index=False)
        t2.to_excel(xw, sheet_name='Table2', index=False)
        pd.read_csv(os.path.join(TAB, 'Table2_raw.csv')).to_excel(xw, sheet_name='Table2_raw', index=False)
        t3.to_excel(xw, sheet_name='Table3', index=False)
        pd.read_csv(os.path.join(TAB, 'Table3_by_station.csv')).to_excel(xw, sheet_name='Table3_by_station', index=False)
        if os.path.exists(os.path.join(TAB, 'Comparison.csv')):
            pd.read_csv(os.path.join(TAB, 'Comparison.csv')).to_excel(xw, sheet_name='Manuscript_vs_reproduced', index=False)
    print('written')


if __name__ == '__main__':
    main()
