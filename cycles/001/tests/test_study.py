"""Non-training contract probes for the unchanged supplied fixture.

Global context: mocks exercise orchestration, never model quality. Real data probes
use the registered partition. API: https://docs.python.org/3.14/library/unittest.html
Scientific API references are in study.py and research/sources.json.
"""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import study


ROOT = Path(study.__file__).resolve().parent


class FixtureContracts(unittest.TestCase):
    """Probe data boundaries, serialization and fail-closed execution guards."""

    @classmethod
    def setUpClass(cls):
        """Load configuration once; no fitting occurs anywhere in this suite."""
        cls.protocol = study.load(ROOT / 'research/protocol.json')

    def test_frozen_bytes_and_registration(self):
        """A silent edit must not invalidate the registered experiment contract."""
        registration = study.load(ROOT / 'research/registration.json')
        self.assertEqual(registration['protocol_sha256'], study.sha(ROOT / 'research/protocol.json'))
        audits = study.load(ROOT / 'research/evidence.json')
        archived = next(item for item in audits if item['id'] == 'audit-fixture-source')
        self.assertEqual(archived['sha256'], study.sha(ROOT / 'study.py'))

    def test_data_boundaries_and_repeatability(self):
        """Check index isolation and scaling independently against the raw loader."""
        x, y, splits = study.dataset(self.protocol)
        x_again, y_again, again = study.dataset(self.protocol)
        raw = study.load_digits()
        np.testing.assert_array_equal(x, raw.data.astype(np.float32) / 16)
        np.testing.assert_array_equal(y, raw.target)
        np.testing.assert_array_equal(x, x_again)
        np.testing.assert_array_equal(y, y_again)
        joined = np.concatenate(list(splits.values()))
        np.testing.assert_array_equal(np.sort(joined), np.arange(len(y)))
        self.assertEqual(len(np.unique(joined)), len(y))
        for name, indices in splits.items():
            np.testing.assert_array_equal(indices, again[name])
            np.testing.assert_array_equal(np.unique(y[indices]), np.unique(y))

    def test_wrong_input_width_rejected(self):
        """Detect a configuration/data mismatch before creating a network."""
        changed = dict(self.protocol, architecture=[1, 128, 10])
        with self.assertRaises(AssertionError):
            study.dataset(changed)

    def test_untrained_checkpoint_round_trip(self):
        """Exercise the real serializer on synthetic features without optimization."""
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(0)
            model = study.network(self.protocol).eval()
            features = torch.zeros(2, self.protocol['architecture'][0])
            checkpoint = Path(directory) / 'untrained.safetensors'
            study.save_file(model.state_dict(), str(checkpoint))
            restored = study.network(self.protocol).eval()
            restored.load_state_dict(study.load_file(str(checkpoint)))
            with torch.no_grad():
                reference, actual = model(features), restored(features)
            self.assertEqual(list(actual.shape), [2, self.protocol['architecture'][-1]])
            torch.testing.assert_close(reference, actual, rtol=0, atol=0)

    def temporary_root(self, directory):
        """Prepare isolated synthetic orchestration artifacts, never real results."""
        root = Path(directory)
        study.write(root / 'research/protocol.json', self.protocol)
        study.write(root / 'research/registration.json', {
            'protocol_sha256': study.sha(root / 'research/protocol.json')})
        return root

    def test_stale_registration_refuses_fitting(self):
        """An invalid registration must stop execution before the model is fit."""
        with tempfile.TemporaryDirectory() as directory, patch.object(study, 'fit_mlp') as fit:
            root = self.temporary_root(directory)
            study.write(root / 'research/registration.json', {'protocol_sha256': 'stale'})
            with self.assertRaisesRegex(AssertionError, 'Unregistered protocol change'):
                study.experiment(root, self.protocol, 'regularized', 0, 'validation', torch.device('cpu'), pilot=True)
            fit.assert_not_called()

    def test_stale_cached_result_refuses_reuse(self):
        """Both code and protocol drift are rejected without overwriting a record."""
        for field, message in [('protocol_sha256', 'Stale protocol'), ('code_sha256', 'Code changed')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = self.temporary_root(directory)
                record = {'protocol_sha256': study.sha(root / 'research/protocol.json'),
                          'code_sha256': study.sha(ROOT / 'study.py')}
                record[field] = 'stale'
                destination = root / 'results/pilot_regularized_s0.json'
                study.write(destination, record)
                before = destination.read_bytes()
                with self.assertRaisesRegex(AssertionError, message):
                    study.experiment(root, self.protocol, 'regularized', 0, 'validation', torch.device('cpu'), pilot=True)
                self.assertEqual(before, destination.read_bytes())

    def test_valid_cached_record_is_reused_without_fitting(self):
        """A repair must not trigger fitting merely because a record is requested again."""
        with tempfile.TemporaryDirectory() as directory, patch.object(study, 'fit_mlp') as fit:
            root = self.temporary_root(directory)
            record = {'protocol_sha256': study.sha(root / 'research/protocol.json'),
                      'code_sha256': study.sha(ROOT / 'study.py'), 'synthetic': True}
            study.write(root / 'results/pilot_regularized_s0.json', record)
            with contextlib.redirect_stdout(io.StringIO()):
                reused = study.experiment(root, self.protocol, 'regularized', 0, 'validation', torch.device('cpu'), pilot=True)
            self.assertEqual(record, reused)
            fit.assert_not_called()

    def test_mock_pilot_scores_validation_only(self):
        """Deliberately different synthetic test predictions expose a wrong split."""
        with tempfile.TemporaryDirectory() as directory:
            root = self.temporary_root(directory)
            checkpoint = root / 'synthetic-checkpoint'
            checkpoint.write_bytes(b'mock checkpoint; not a trained model')
            x, y = np.zeros((6, 64), dtype=np.float32), np.arange(6)
            splits = {'train': np.array([0, 1]), 'validation': np.array([2, 3]), 'test': np.array([4, 5])}
            prediction = np.array([9, 9, 2, 3, 9, 9])
            with patch.object(study, 'dataset', return_value=(x, y, splits)), \
                 patch.object(study, 'fit_mlp', return_value=(prediction, [], checkpoint)) as fit, \
                 contextlib.redirect_stdout(io.StringIO()):
                record = study.experiment(root, self.protocol, 'regularized', 0, 'validation', torch.device('cpu'), pilot=True)
            fit.assert_called_once()
            self.assertEqual(record['example_ids'], splits['validation'].tolist())
            self.assertEqual(record['y_true'], [2, 3])
            self.assertEqual(record['y_pred'], [2, 3])
            self.assertEqual(record['accuracy'], 1.0)
            self.assertEqual(record['analysis_kind'], 'exploratory')
            self.assertEqual(record, study.load(root / 'results/pilot_regularized_s0.json'))

    def test_comparisons_require_publication(self):
        """The fixture refuses comparisons before the controller's publication gate."""
        with tempfile.TemporaryDirectory() as directory:
            root = self.temporary_root(directory)
            with patch.object(study, '__file__', str(root / 'study.py')), \
                 patch.object(study.sys, 'argv', ['study.py', 'experiments']), \
                 patch.object(study, 'experiment') as experiment:
                with self.assertRaisesRegex(RuntimeError, 'Publish the first pilot'):
                    study.main()
                experiment.assert_not_called()


if __name__ == '__main__':
    unittest.main()
