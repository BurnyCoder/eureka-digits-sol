"""Audit existing pilot artifacts and regenerate displays; never train or score test data.

Global context: study.py owns the science. This post-hoc exploratory audit reuses
its constructors but independently checks saved labels, arithmetic and provenance.
Sources: study.py; src-torch-adamw; src-reproducibility; src-matplotlib-plot.
"""
from importlib.metadata import version
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


def audit(record, protocol):
    """Fail on inconsistent artifacts, without fitting or calculating test accuracy."""
    assert record['id'] == 'pilot_regularized_s0'
    assert (record['method'], record['seed'], record['split']) == ('regularized', 0, 'validation')
    assert record['analysis_kind'] == protocol['analysis_kind'] == 'exploratory'
    assert study.sha(ROOT / 'study.py') == record['code_sha256']
    assert study.sha(ROOT / 'research/protocol.json') == record['protocol_sha256']
    assert study.load(ROOT / 'research/registration.json')['protocol_sha256'] == record['protocol_sha256']
    checkpoint = ROOT / record['checkpoint']
    assert study.sha(checkpoint) == record['checkpoint_sha256']
    assert record['checkpoint_reload_verified'] is True
    x, y, splits = study.dataset(protocol)
    assert study.hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest() == record['dataset_sha256']
    for name, indices in splits.items():
        np.testing.assert_array_equal(indices, record['split_ids'][name])
        assert len(np.unique(indices)) == len(indices)
    assert len(np.unique(np.concatenate(list(splits.values())))) == len(y)
    indices = splits['validation']
    np.testing.assert_array_equal(indices, record['example_ids'])
    np.testing.assert_array_equal(y[indices], record['y_true'])
    truth, predicted = np.asarray(record['y_true']), np.asarray(record['y_pred'])
    assert truth.shape == predicted.shape == indices.shape
    assert set(predicted).issubset(set(np.unique(y[splits['train']])))
    assert float(np.mean(truth == predicted)) == record['accuracy']
    history = record['history']
    assert [row['epoch'] for row in history] == list(range(1, protocol['epochs'] + 1))
    assert np.isfinite([[row['train_loss'], row['validation_loss']] for row in history]).all()

    # Match recorded training device; this audit forwards validation features only.
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device(record['device'])
    model = study.network(protocol).to(device)
    model.load_state_dict(load_file(str(checkpoint), device=str(device)))
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x[indices], device=device))
        actual = logits.argmax(1).cpu().numpy()
        final_loss = float(torch.nn.functional.cross_entropy(logits, torch.as_tensor(truth, device=device)))
    np.testing.assert_array_equal(actual, predicted)
    assert final_loss == history[-1]['validation_loss']

    # Reconstruct effective defaults from the installed optimizer, without a step.
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'],
                                 weight_decay=protocol['weight_decay'][record['method']])
    defaults = {key: value for key, value in optimizer.param_groups[0].items() if key != 'params'}
    # Canonical JSON types make a repeated audit comparable to its saved bytes.
    defaults = json.loads(json.dumps(defaults))
    logs = []
    for path in sorted((ROOT / 'reports/logs').glob('pilot-*.log')):
        text = path.read_text(encoding='utf-8')
        if f'"id": "{record["id"]}"' not in text:
            continue
        logged = [json.loads(line[line.index('{'):]) for line in text.splitlines()
                  if ' regularized seed=0 {' in line]
        if logged:  # Cache-only invocations are not new training evidence.
            assert logged == history
            assert text.rstrip().endswith('exit=0')
            logs.append({'path': str(path.relative_to(ROOT)), 'sha256': study.sha(path)})
    assert len(logs) == 1, 'Expected exactly one complete pilot training log'
    result_files = sorted(path.name for path in (ROOT / 'results').glob('*.json'))
    assert 'pilot_regularized_s0.json' in result_files
    return {
        'analysis_kind': 'exploratory', 'experiment_ref': record['id'],
        'result_sha256': study.sha(ROOT / 'results/pilot_regularized_s0.json'),
        'protocol_snapshot': protocol, 'effective_adamw_group_reconstructed': defaults,
        'configuration_basis': 'Frozen source/protocol and installed defaults; no optimizer state was saved.',
        'runtime_reconstructed_from_source': {'threads': 2, 'deterministic_algorithms': True,
                                             'dtype': str(next(model.parameters()).dtype),
                                             'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']},
        'environment_at_audit': {'python': sys.version, 'executable': sys.executable,
                                 'packages': {name: version(name) for name in
                                              ('torch', 'numpy', 'scikit-learn', 'safetensors', 'matplotlib')},
                                 'cuda_runtime': torch.version.cuda},
        'split_sizes': {key: len(value) for key, value in splits.items()},
        'correct': int(np.sum(truth == predicted)), 'errors': int(np.sum(truth != predicted)),
        'validation_accuracy_recomputed': float(np.mean(truth == predicted)),
        'checkpoint_sha256': study.sha(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
        'validation_reload_predictions_equal': True, 'validation_reload_loss': final_loss,
        'all_recorded_hashes_match': True, 'split_ids_and_labels_match': True,
        'history_and_log_match': True, 'training_logs': logs,
        'result_files_at_first_audit': result_files,
    }


def render(record, protocol, checked):
    """Build numeric table and unsmoothed linear loss plot from raw result rows."""
    history = record['history']
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout='constrained')
    for key, label in [('train_loss', 'Training (online minibatch average)'),
                       ('validation_loss', 'Validation (end of epoch)')]:
        ax.plot([row['epoch'] for row in history], [row[key] for row in history], label=label)
    ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title=record['id'],
           xlim=(1, protocol['epochs']), ylim=(0, None))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(ROOT / 'figures/pilot-loss.png', dpi=160)
    plt.close(fig)
    rows = [
        ('Experiment', record['id']), ('Analysis / scored split', f"{record['analysis_kind']} / {record['split']}"),
        ('Architecture / activation', f"{' → '.join(map(str, protocol['architecture']))} / {protocol['activation']}"),
        ('Seed / split seed', f"{record['seed']} / {protocol['split_seed']}"),
        ('Optimizer / learning rate / decay', f"{protocol['optimizer']} / {protocol['learning_rate']} / {protocol['weight_decay'][record['method']]}"),
        ('Epochs / batch size', f"{protocol['epochs']} / {protocol['batch_size']}"),
        ('Train / validation / test examples', ' / '.join(str(checked['split_sizes'][key]) for key in ('train', 'validation', 'test'))),
        ('Validation correct / total (errors)', f"{checked['correct']} / {len(record['y_true'])} ({checked['errors']})"),
        ('Validation accuracy', f"{record['accuracy']:.10f} ({100 * record['accuracy']:.4f}%)"),
        ('Training loss: first → final epoch', f"{history[0]['train_loss']:.8f} → {history[-1]['train_loss']:.8f}"),
        ('Validation loss: first → final epoch', f"{history[0]['validation_loss']:.8f} → {history[-1]['validation_loss']:.8f}"),
        ('Local validation-loss increases', str(int(np.sum(np.diff([row['validation_loss'] for row in history]) > 0)))),
        ('Experiment elapsed seconds', f"{record['seconds']:.6f} (fixture timer; not end-to-end runtime)"),
        ('Training device / hardware', f"{record['device']} / {record['hardware']}"),
        ('Torch / NumPy', f"{record['torch_version']} / {record['numpy_version']}"),
        ('Checkpoint bytes / SHA-256', f"{checked['checkpoint_bytes']} / `{record['checkpoint_sha256']}`"),
    ]
    table = '| Quantity | Value |\n| --- | --- |\n' + '\n'.join(f'| {key} | {value} |' for key, value in rows)
    report = ROOT / 'reports/pilot.md'
    before, remaining = report.read_text(encoding='utf-8').split('<!-- BEGIN GENERATED PILOT -->')
    _, after = remaining.split('<!-- END GENERATED PILOT -->')
    report.write_text(before + '<!-- BEGIN GENERATED PILOT -->\n\n' + table +
                      '\n\n<!-- END GENERATED PILOT -->' + after, encoding='utf-8')


def main():
    """Preserve the first audit and regenerate only presentation on later calls."""
    record = study.load(ROOT / 'results/pilot_regularized_s0.json')
    protocol = study.load(ROOT / 'research/protocol.json')
    checked = audit(record, protocol)
    destination = ROOT / 'research/pilot-audit.json'
    if destination.exists():
        # Later phases can add results without changing this historical pilot audit.
        prior = study.load(destination)
        checked['result_files_at_first_audit'] = prior['result_files_at_first_audit']
        assert checked == prior, 'Preserve old audit and investigate any changed provenance'
    else:
        assert checked['result_files_at_first_audit'] == ['pilot_regularized_s0.json']
        study.write(destination, checked)
    render(record, protocol, checked)
    print(f'[{study.timestamp()}] Pilot audit passed; report and figure regenerated; no fitting performed.')


if __name__ == '__main__':
    main()
