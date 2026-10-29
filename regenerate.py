"""Rebuild every current numerical report and presentation without fitting.

Reuse the study's saved-record builders and the harness renderer rather than
duplicating their arithmetic. Historical audit receipts remain historical.
Python module paths: https://docs.python.org/3/library/sys.html#sys.path
"""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
CYCLE = PROJECT / 'cycles/001'
sys.path.insert(0, str(CYCLE / 'reports'))

import build_pilot
import build_experiments
from audit_publish import render_final
from repair_presentation import main as repair_presentation
from auto_eureka_scientist.reporting import render
from auto_eureka_scientist.util import read_json


def main():
    """Verify saved checkpoints, refresh numerical sections, then compile once."""
    build_pilot.main()
    build_experiments.main()
    records = {path.stem: read_json(path) for path in (CYCLE / 'results').glob('*.json')}
    render_final(records, read_json(CYCLE / 'research/protocol.json'),
                 read_json(CYCLE / 'reports/comparison-summary.json'))
    render(PROJECT, CYCLE, backend='Codex', pdf=False)
    repair_presentation()


if __name__ == '__main__':
    main()
