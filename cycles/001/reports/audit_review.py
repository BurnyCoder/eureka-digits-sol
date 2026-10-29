"""Non-fitting review checks; preserve measurements and record actual output.

The frozen study defines the science. Reuse the existing checkpoint audit and
paired statistic, then check derived publication copies and local references.
Sources: src-python-subprocess, src-python-unittest, src-bootstrap.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from build_experiments import audit_records
from compare_artifacts import paired_summary
import study


def main():
    """Fail on scientific drift; report presentation defects without hiding them."""
    protected = list((ROOT / 'results').glob('*.json')) + list((ROOT / 'models').glob('*'))
    protected += [ROOT / 'study.py', ROOT / 'research/protocol.json', ROOT / 'research/registration.json']
    before = {str(p.relative_to(PROJECT)): study.sha(p) for p in protected if p.is_file()}
    commands = []
    for args in [('study.py', 'check'), ('-m', 'unittest', 'discover', '-s', 'tests', '-v'),
                 ('run_study.py', 'check')]:
        command = [sys.executable, *args]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        commands.append({'argv': command, 'returncode': run.returncode, 'output': run.stdout + run.stderr})
    protocol = study.load(ROOT / 'research/protocol.json')
    records, checked, _ = audit_records(protocol)
    assert checked == study.load(ROOT / 'research/experiments-audit.json')
    primary = paired_summary([records[f'regularized_s{s}'] for s in protocol['seeds']],
                             [records[f'unregularized_s{s}'] for s in protocol['seeds']],
                             resamples=protocol['bootstrap_resamples'], seed=protocol['bootstrap_seed'])
    assert primary == study.load(ROOT / 'reports/comparison-summary.json')
    site = study.load(PROJECT / 'site/data.json')
    assert site['claims'] == study.load(ROOT / 'research/claims.json')
    assert {r['id'] for r in site['records']} == set(records)
    for row in site['records']:
        for key in ('accuracy', 'y_true', 'y_pred', 'example_ids', 'history'):
            assert row[key] == records[row['id']][key]
    broken = []
    for path in (ROOT / 'reports/paper.md', PROJECT / 'site/paper.md', PROJECT / 'README.md'):
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            target = unquote(link.split('#')[0])
            if target and not re.match(r'https?://|mailto:', target) and not (path.parent / target).exists():
                broken.append({'document': str(path.relative_to(PROJECT)), 'target': link})
    # Claims cite existing experiment/source/audit IDs, never invented result IDs.
    known = set(records) | {r['id'] for r in study.load(ROOT / 'research/sources.json')}
    known |= {r['id'] for r in study.load(ROOT / 'research/evidence.json')}
    for claim in study.load(ROOT / 'research/claims.json'):
        assert set(claim['evidence']) <= known
    assert before == {str(p.relative_to(PROJECT)): study.sha(p) for p in protected if p.is_file()}
    destination = ROOT / 'research/review-audit.json'
    result = {'analysis_kind': 'exploratory', 'at': datetime.now(timezone.utc).isoformat(),
              'commands': commands, 'experiment_audit_reproduced': True,
              'primary_summary_reproduced': True, 'site_records_match': True,
              'protected_sha256': before, 'broken_local_links': broken,
              'paper_copies_match': (ROOT / 'reports/paper.pdf').read_bytes() == (PROJECT / 'site/paper.pdf').read_bytes(),
              'primary_prediction_disagreements': [sum(a != b for a, b in zip(records[f'regularized_s{s}']['y_pred'], records[f'unregularized_s{s}']['y_pred'])) for s in protocol['seeds']],
              'no_fitting': True}
    study.write(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('commands', 'protected_sha256')}))
    assert all(c['returncode'] == 0 for c in commands)
    assert not broken, 'Repair local publication links before handoff'


if __name__ == '__main__':
    main()
