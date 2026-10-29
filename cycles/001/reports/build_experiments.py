"""Check saved comparisons and regenerate displays without optimizer steps.

Global context: study.py owns scientific definitions; this post-hoc exploratory
audit reuses its dataset/network constructors and independently verifies artifacts.
Sources: src-bootstrap, src-torch-adamw, src-matplotlib-plot, audit-fixture-source.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
import study
from compare_artifacts import paired_summary


def audit_records(protocol):
    """Require the entire registered matrix, reload checkpoints and verify logs."""
    expected = ['pilot_regularized_s0'] + [f'{method}_s{seed}'
        for method in protocol['methods']
        for seed in (protocol['seeds'] if method in protocol['weight_decay'] else [0])]
    assert sorted(path.stem for path in (ROOT / 'results').glob('*.json')) == sorted(expected)
    records = {key: study.load(ROOT / f'results/{key}.json') for key in expected}
    x, y, splits = study.dataset(protocol)
    assert study.load(ROOT / 'research/registration.json')['protocol_sha256'] == study.sha(ROOT / 'research/protocol.json')
    publication = study.load(ROOT / 'research/pilot-publication.json')
    pilot = records['pilot_regularized_s0']
    assert publication['models']['verified'] is True
    assert publication['models']['files']['regularized-s0.safetensors'] == pilot['checkpoint_sha256']
    logs = sorted((ROOT / 'reports/logs').glob('experiments-*.log'))
    assert len(logs) == 1, 'Review extra attempts explicitly rather than silently excluding them'
    text = logs[0].read_text(encoding='utf-8')
    assert text.rstrip().endswith('exit=0')
    assert not any(token in text for token in ('Traceback', 'Warning:', 'Error:'))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    checked = []
    for key, row in records.items():
        method, seed = row['method'], row['seed']
        assert key == row['id'] and row['analysis_kind'] == 'exploratory'
        assert key == ('pilot_' if key.startswith('pilot_') else '') + f'{method}_s{seed}'
        assert row['protocol_sha256'] == study.sha(ROOT / 'research/protocol.json')
        assert row['code_sha256'] == study.sha(ROOT / 'study.py')
        assert row['dataset_sha256'] == study.hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
        assert row['checkpoint_reload_verified'] is True
        checkpoint = ROOT / row['checkpoint']
        assert study.sha(checkpoint) == row['checkpoint_sha256']
        for name, indices in splits.items():
            np.testing.assert_array_equal(indices, row['split_ids'][name])
        indices = splits['validation' if key.startswith('pilot_') else 'test']
        assert row['split'] == ('validation' if key.startswith('pilot_') else 'test')
        np.testing.assert_array_equal(indices, row['example_ids'])
        np.testing.assert_array_equal(y[indices], row['y_true'])
        predicted = np.asarray(row['y_pred'])
        assert predicted.shape == indices.shape
        assert row['accuracy'] == float(np.mean(predicted == y[indices]))
        assert set(predicted).issubset(set(np.unique(y)))
        if method in protocol['weight_decay']:
            history = row['history']
            assert [h['epoch'] for h in history] == list(range(1, protocol['epochs'] + 1))
            assert np.isfinite([[h['train_loss'], h['validation_loss']] for h in history]).all()
            model = study.network(protocol).to(row['device'])
            model.load_state_dict(load_file(str(checkpoint), device=row['device']))
            model.eval()
            with torch.no_grad():
                restored = model(torch.as_tensor(x[indices], device=row['device'])).argmax(1).cpu().numpy()
                val_loss = float(torch.nn.functional.cross_entropy(
                    model(torch.as_tensor(x[splits['validation']], device=row['device'])),
                    torch.as_tensor(y[splits['validation']], device=row['device'])))
            assert val_loss == history[-1]['validation_loss']
            if not key.startswith('pilot_') and key != 'regularized_s0':
                logged = [json.loads(line[line.index('{'):]) for line in text.splitlines()
                          if f' {method} seed={seed} {{' in line]
                assert logged == history
        else:
            assert row['history'] == []
            with np.load(checkpoint, allow_pickle=False) as saved:
                if method == 'logistic':
                    restored = saved['classes'][(x[indices] @ saved['coef'].T + saved['intercept']).argmax(1)]
                else:
                    classes, counts = np.unique(y[splits['train']], return_counts=True)
                    np.testing.assert_array_equal(saved['classes'], classes)
                    np.testing.assert_array_equal(saved['probabilities'], counts / counts.sum())
                    restored = np.full(len(indices), classes[counts.argmax()])
        np.testing.assert_array_equal(restored, predicted)
        if not key.startswith('pilot_'):
            saved_lines = [json.loads(line.split(' SAVED ', 1)[1]) for line in text.splitlines()
                           if ' SAVED ' in line]
            assert [s for s in saved_lines if s['id'] == key] == [
                {k: row[k] for k in ('id', 'accuracy', 'seconds', 'device')}]
        checked.append({'experiment_ref': key, 'result_sha256': study.sha(ROOT / f'results/{key}.json'),
                        'correct': int(np.sum(predicted == y[indices])), 'total': len(indices),
                        'checkpoint_sha256': row['checkpoint_sha256']})
    final = records['regularized_s0']
    assert final['checkpoint_sha256'] == pilot['checkpoint_sha256']
    assert final['history'] == pilot['history']
    assert not any(' regularized seed=0 {' in line for line in text.splitlines())
    # Source-level reuse plus identical checkpoint/history and no fit log support reuse.
    weight_checks = []
    for seed in protocol['seeds']:
        left = load_file(str(ROOT / records[f'regularized_s{seed}']['checkpoint']))
        right = load_file(str(ROOT / records[f'unregularized_s{seed}']['checkpoint']))
        weight_checks.append({'seed': seed, 'equal_state_dicts': all(torch.equal(left[k], right[k]) for k in left),
                              'maximum_parameter_absolute_difference': max(float((left[k] - right[k]).abs().max()) for k in left)})
    audit = {'analysis_kind': 'exploratory', 'checks': checked,
             'split_sizes': {k: len(v) for k, v in splits.items()},
             'training_logs': [{'path': str(p.relative_to(ROOT)), 'sha256': study.sha(p)} for p in logs],
             'completed_comparison_count': len(records) - 1, 'new_neural_fits': sum(' seed=' in l and '"epoch": 1,' in l for l in text.splitlines()),
             'all_prediction_metrics_hashes_and_reload_checks_passed': True,
             'histories_match_logs': True, 'published_pilot_reused': True,
             'paired_parameter_checks_post_hoc': weight_checks,
             'initial_synthetic_test_failure': 'Before implementation, test_comparisons could not import compare_artifacts (ModuleNotFoundError); expected test-first failure. No fitting failure occurred.'}
    return records, audit, x


def table(headers, rows):
    """Render generated Markdown tables without manually entered empirical values."""
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + '\n'.join(
        '| ' + ' | '.join(map(str, row)) + ' |' for row in rows)


def render(records, audit, primary, protocol, x):
    """Produce complete descriptive tables, full trajectories and fixed example views."""
    rows = []
    for check in audit['checks']:
        row = records[check['experiment_ref']]
        history = row['history']
        rows.append([f"[{row['id']}](../results/{row['id']}.json)", row['split'],
                     f"{check['correct']}/{check['total']}", f"{100 * row['accuracy']:.4f}%",
                     f"{history[-1]['train_loss']:.6f}" if history else 'n/a',
                     f"{history[-1]['validation_loss']:.6f}" if history else 'n/a', row['device']])
        if row['id'].startswith('pilot_'):
            continue  # Preserve controller-generated historical pilot report.
        purpose = {'regularized': 'Primary decay intervention', 'unregularized': 'Zero-decay ablation',
                   'sensitivity': 'Prespecified weaker-decay sensitivity', 'shuffled_labels': 'Negative label control',
                   'logistic': 'Penalized linear baseline', 'class_prior': 'Training-majority constant baseline'}[row['method']]
        content = f"# {row['id']}\n\nExploratory execution. {purpose}.\n\n"
        content += table(['Result reference', 'Split', 'Correct / total', 'Accuracy', 'Final online train loss', 'Final validation loss', 'Device'], [rows[-1]])
        content += '\n\nConfiguration, checkpoint SHA, timestamps and predictions are in the linked raw record and frozen protocol. No best epoch or test-selected tuning. '
        content += 'Checkpoint reload, label/split membership, metric arithmetic and provenance passed audit-experiments-artifacts. '
        content += 'This fitted-instance result alone cannot establish a decay benefit; see [the comparison report](experiments.md) for predictions, learning and limitations.\n'
        (ROOT / f"reports/experiment-{row['id']}.md").write_text(content, encoding='utf-8')
    generated = table(['Experiment', 'Scored split', 'Correct / total', 'Accuracy', 'Final online train loss', 'Final validation loss', 'Device'], rows)
    generated += '\n\n' + table(['Primary quantity', 'Value'], [
        ['Paired seed differences (accuracy fraction)', str(primary['paired_seed_differences'])],
        ['Mean difference (accuracy fraction)', f"{primary['mean_difference']:.10f}"],
        ['Mean difference (percentage points)', f"{100 * primary['mean_difference']:.6f}"],
        ['Sample SD of seed differences (ddof=1)', f"{primary['seed_difference_sample_sd']:.10f}"],
        ['Conditional percentile interval (accuracy fraction)', str(primary['conditional_interval'])],
        ['Bootstrap generator seed / resamples', f"{primary['bootstrap_seed']} / {primary['bootstrap_resamples']}"],
        ['Correctness disagreements per paired seed', str(primary['correctness_disagreements_by_seed'])],
        ['Interpretation', primary['interpretation']],
        ['Registered comparison records / newly fitted neural models', f"{audit['completed_comparison_count']} / {audit['new_neural_fits']}"],
        ['Train / validation / test size', str(list(audit['split_sizes'].values()))]])
    grouped = []
    for method in protocol['methods']:
        values = [r['accuracy'] for r in records.values() if r['method'] == method and r['split'] == 'test']
        grouped.append([method, str(len(values)), f'{100 * np.mean(values):.4f}%',
                        f'{100 * np.std(values, ddof=1):.4f} pp' if len(values) > 1 else 'not estimated',
                        f'{100 * min(values):.4f}% to {100 * max(values):.4f}%'])
    generated += '\n\n' + table(['Arm', 'Fitted seeds', 'Mean accuracy', 'Seed sample SD', 'Observed seed range'], grouped)
    generated += '\n\n' + table(['Paired seed', 'State dictionaries equal?', 'Maximum absolute parameter difference'],
        [[r['seed'], r['equal_state_dicts'], f"{r['maximum_parameter_absolute_difference']:.10f}"] for r in audit['paired_parameter_checks_post_hoc']])
    report = ROOT / 'reports/experiments.md'
    before, tail = report.read_text(encoding='utf-8').split('<!-- BEGIN GENERATED COMPARISONS -->')
    _, after = tail.split('<!-- END GENERATED COMPARISONS -->')
    report.write_text(before + '<!-- BEGIN GENERATED COMPARISONS -->\n\n' + generated + '\n\n<!-- END GENERATED COMPARISONS -->' + after, encoding='utf-8')
    # Show every seed separately, avoiding misleading uncertainty bars or clipped axes.
    fig, ax = plt.subplots(figsize=(9, 4.2), layout='constrained')
    for i, method in enumerate(protocol['methods']):
        values = [r['accuracy'] * 100 for r in records.values() if r['method'] == method and r['split'] == 'test']
        ax.scatter(np.linspace(i - .12, i + .12, len(values)) if len(values) > 1 else [i], values)
    ax.set(xticks=range(len(protocol['methods'])), xticklabels=protocol['methods'], ylim=(0, 100), ylabel='Test accuracy (%)', title='All registered fitted seeds; points are not confidence intervals')
    ax.tick_params(axis='x', labelrotation=15)
    fig.savefig(ROOT / 'figures/comparison-accuracy.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout='constrained', sharex=True)
    for column, method in enumerate(protocol['weight_decay']):
        for row in records.values():
            if row['method'] != method or row['split'] != 'test':
                continue
            for axis, key in zip(axes[:, column], ('train_loss', 'validation_loss')):
                axis.plot([h['epoch'] for h in row['history']], [h[key] for h in row['history']], label=f"seed {row['seed']}")
                axis.set(ylim=(0, 2.6), xlabel='Epoch', ylabel=key.replace('_', ' '))
        axes[0, column].set_title(method)
        axes[0, column].legend(fontsize=8)
    fig.savefig(ROOT / 'figures/comparison-loss.png', dpi=160)
    plt.close(fig)
    # Fixed first examples were selected by saved test order, never by correctness.
    example_ids = records['regularized_s0']['example_ids'][:8]
    fig, axes = plt.subplots(len(protocol['methods']), len(example_ids), figsize=(12, 9), layout='constrained')
    for i, method in enumerate(protocol['methods']):
        row = records[f'{method}_s0']
        for j, idx in enumerate(example_ids):
            ax = axes[i, j]
            ax.imshow(x[idx].reshape(8, 8), cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"true {row['y_true'][j]}, pred {row['y_pred'][j]}", fontsize=8)
            if j == 0:
                ax.set_ylabel(method, fontsize=8)
    fig.savefig(ROOT / 'figures/comparison-examples.png', dpi=160)
    plt.close(fig)


def main():
    """Preserve the first successful audit; regenerate only derived displays."""
    protocol = study.load(ROOT / 'research/protocol.json')
    records, checked, x = audit_records(protocol)
    path = ROOT / 'research/experiments-audit.json'
    if path.exists():
        assert checked == study.load(path), 'Preserve old evidence and investigate drift'
    else:
        study.write(path, checked)
    primary = paired_summary([records[f'regularized_s{s}'] for s in protocol['seeds']],
                             [records[f'unregularized_s{s}'] for s in protocol['seeds']],
                             resamples=protocol['bootstrap_resamples'], seed=protocol['bootstrap_seed'])
    study.write(ROOT / 'reports/comparison-summary.json', primary)
    render(records, checked, primary, protocol, x)
    print(json.dumps({'audit_passed': True, 'primary': primary, 'no_fitting': True}))


if __name__ == '__main__':
    main()
