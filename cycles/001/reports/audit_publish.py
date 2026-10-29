"""Reconcile final documents with preserved measurements; never fit a model.

Global context: reuse the reviewed scientific bytes and registered statistic.
Only the final report's marked numerical table is regenerated. Sources for
argv execution and the existing statistic: src-python-subprocess, src-bootstrap.
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
import study
from compare_artifacts import paired_summary


def render_final(records, protocol, primary):
    """Generate empirical values from saved records rather than typing scores."""
    lines = ['| Test arm | Recorded seeds | Mean accuracy |', '| --- | --- | --- |']
    for method in protocol['methods']:
        rows = [r for r in records.values() if r['method'] == method and r['split'] == 'test']
        assert rows
        average = sum(r['accuracy'] for r in rows) / len(rows)
        lines.append(f'| `{method}` | {len(rows)} | {average:.6%} |')
    lines += ['', '| Primary quantity | Recorded value |', '| --- | --- |',
              f"| Mean paired accuracy difference | {primary['mean_difference']:.6f} |",
              f"| Conditional percentile interval | {json.dumps(primary['conditional_interval'])} |",
              f"| Seed effect sample SD | {primary['seed_difference_sample_sd']:.6f} |",
              f"| Registered decision | {primary['interpretation']} |"]
    path = ROOT / 'reports/final.md'
    begin = '<!-- BEGIN GENERATED FINAL MEASUREMENTS -->'
    end = '<!-- END GENERATED FINAL MEASUREMENTS -->'
    text = path.read_text(encoding='utf-8')
    prefix, remainder = text.split(begin)
    _, suffix = remainder.split(end)
    path.write_text(prefix + begin + '\n\n' + '\n'.join(lines) + '\n\n' + end + suffix, encoding='utf-8')


def main():
    """Check drift, evidence and local packaging, preserving the first audit."""
    reviewed = study.load(ROOT / 'research/review-audit.json')
    protected = reviewed['protected_sha256']
    for path, expected in protected.items():
        assert study.sha(PROJECT / path) == expected, f'Protected artifact drift: {path}'
    records = {row['id']: row for path in sorted((ROOT / 'results').glob('*.json'))
               for row in [study.load(path)]}
    for row in records.values():
        assert row['analysis_kind'] == 'exploratory'
        correct = sum(a == b for a, b in zip(row['y_true'], row['y_pred']))
        assert len(row['y_true']) == len(row['y_pred']) == len(row['example_ids'])
        assert correct / len(row['y_true']) == row['accuracy']
    protocol = study.load(ROOT / 'research/protocol.json')
    primary = paired_summary([records[f'regularized_s{s}'] for s in protocol['seeds']],
                             [records[f'unregularized_s{s}'] for s in protocol['seeds']],
                             resamples=protocol['bootstrap_resamples'], seed=protocol['bootstrap_seed'])
    assert primary == study.load(ROOT / 'reports/comparison-summary.json')
    render_final(records, protocol, primary)
    sources = study.load(ROOT / 'research/sources.json')
    evidence = study.load(ROOT / 'research/evidence.json')
    assert len({r['id'] for r in evidence}) == len(evidence)
    for item in evidence:
        assert study.sha(PROJECT / item['path']) == item['sha256'], item['id']
    known = set(records) | {r['id'] for r in sources} | {r['id'] for r in evidence}
    claims = study.load(ROOT / 'research/claims.json')
    for claim in claims:
        assert claim['analysis_kind'] == 'exploratory'
        assert set(claim['evidence']) <= known
    for requirement in study.load(ROOT / 'research/requirements.json'):
        assert requirement['reason']
        assert all((PROJECT / path).exists() for path in requirement['evidence'])
    site = study.load(PROJECT / 'site/data.json')
    assert site['claims'] == claims
    assert {r['id'] for r in site['records']} == set(records)
    for row in site['records']:
        assert all(row[key] == records[row['id']][key]
                   for key in ('accuracy', 'y_true', 'y_pred', 'example_ids', 'history'))
    visual = study.load(ROOT / 'research/review-presentation-audit.json')
    assert study.sha(ROOT / 'reports/paper.pdf') == visual['pdf_sha256']
    assert study.sha(PROJECT / 'site/paper.pdf') == visual['site_pdf_sha256']
    documents = [PROJECT / 'README.md', ROOT / 'docs/reproduction.md',
                 ROOT / 'reports/final.md', ROOT / 'reports/interpretation.md',
                 ROOT / 'reports/paper.md', PROJECT / 'site/paper.md']
    pending_audit = ROOT / 'research/publish-audit.json'
    for path in documents:
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            target = unquote(link.split('#')[0])
            if target and not re.match(r'https?://|mailto:', target):
                resolved = (path.parent / target).resolve()
                assert resolved.exists() or resolved == pending_audit, f'{path}: {link}'
    # A shell-free canonical check exercises the user-facing non-fitting path.
    command = [sys.executable, 'study.py', 'check']
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    print(run.stdout + run.stderr, end='')
    assert run.returncode == 0
    assert all(study.sha(PROJECT / path) == expected for path, expected in protected.items())
    destination = ROOT / 'research/publish-audit.json'
    result = {'analysis_kind': 'exploratory', 'at': datetime.now(timezone.utc).isoformat(),
              'protected_review_hashes_match': True, 'primary_summary_reproduced': True,
              'record_accuracy_reproduced': True, 'claim_references_resolve': True,
              'evidence_hashes_match': True, 'requirement_paths_exist': True,
              'site_records_match': True, 'reviewed_pdf_hashes_match': True,
              'local_document_links_exist': True, 'final_table_generated_from_results': True,
              'no_fitting': True, 'browser_visual_check_performed': False,
              'remote_publication_checked': False,
              'canonical_check': {'argv': command, 'returncode': run.returncode,
                                  'output': run.stdout + run.stderr}}
    if not destination.exists():
        study.write(destination, result)
    assert destination.exists()
    print(json.dumps({k: v for k, v in result.items() if k != 'canonical_check'}))


if __name__ == '__main__':
    main()
