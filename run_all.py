#!/usr/bin/env python3
"""Reproduce Figures 1–4 and Tables 1–3 in section order.

Usage:  python run_all.py            # run every step
        python run_all.py table2     # run from a given step onward (e.g. after an interruption)
"""
import os, subprocess, sys, time

STEPS = [
    ('prep',           '3.7  Taiwan sample (231,825 transactions)'),
    ('prep_ca',        '3.7  California block groups -> counties'),
    ('fig1_taxonomy',  '3.2  Figure 1'),
    ('fig2_framework', '3.4  Figure 2'),
    ('fig3_workflow',  '3.4  Figure 3'),
    ('core',           'predictions for all validation designs (~10 min)'),
    ('table1',         '4.1  Table 1'),
    ('table2',         '4.2  Table 2 (~4 min)'),
    ('calib',          '4.5  Table 3 + Figure 4 (~17 min)'),
    ('compare',        'manuscript vs reproduced'),
    ('make_outputs',   'output/Figures_Tables_by_section.docx, output/Tables_1-3.xlsx'),
]


REQUIRED = {'numpy': 'numpy', 'pandas': 'pandas', 'pyarrow': 'pyarrow', 'scipy': 'scipy', 'sklearn': 'scikit-learn',
            'lightgbm': 'lightgbm', 'gpboost': 'gpboost', 'matplotlib': 'matplotlib', 'docx': 'python-docx',
            'openpyxl': 'openpyxl'}


def check_packages():
    import importlib.util
    missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        sys.exit('Missing packages: ' + ', '.join(missing) +
                 f'\nInstall with:  {sys.executable} -m pip install ' + ' '.join(missing))


def main():
    check_packages()
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    names = [s for s, _ in STEPS]
    start = names.index(sys.argv[1]) if len(sys.argv) > 1 else 0
    t_all = time.time()
    for name, desc in STEPS[start:]:
        print(f'\n=== {name}.py  ({desc}) ===', flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, f'{name}.py'], cwd=src)
        if r.returncode != 0:
            sys.exit(f'{name}.py failed (exit code {r.returncode}); see the traceback above. '
                     f'After fixing it, resume with: python run_all.py {name}')
        print(f'--- {name}.py finished in {time.time() - t0:.0f} s', flush=True)
    print(f'\nAll steps finished in {(time.time() - t_all) / 60:.1f} min.')


if __name__ == '__main__':
    main()
