"""Test console preservation and child-failure propagation, without training.

Sources: https://docs.python.org/3.14/library/subprocess.html and
https://docs.python.org/3.14/library/unittest.html .
"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

from run_study import stream_command


class RunnerContracts(unittest.TestCase):
    """Real child processes exercise the log path rather than mocking the pipe."""

    def test_full_stdout_stderr_and_failure_status(self):
        """A long line and stderr must survive; nonzero exit must stay nonzero."""
        with tempfile.TemporaryDirectory() as directory:
            root, terminal = Path(directory), io.StringIO()
            log = root / 'console.log'
            command = [sys.executable, '-u', '-c',
                       "import sys; print('x'*20000); print('diagnostic',file=sys.stderr); sys.exit(7)"]
            with contextlib.redirect_stdout(terminal):
                status = stream_command(command, root, log)
            self.assertEqual(status, 7)
            saved = log.read_text(encoding='utf-8')
            self.assertIn('x' * 20000, saved)
            self.assertIn('diagnostic', saved)
            self.assertIn('exit=7', saved)
            self.assertEqual(saved, terminal.getvalue())

    def test_success_and_working_directory(self):
        """An argument array must preserve a path containing spaces and report success."""
        with tempfile.TemporaryDirectory(prefix='digits core ') as directory:
            root = Path(directory).resolve()
            log = root / 'console.log'
            with contextlib.redirect_stdout(io.StringIO()):
                status = stream_command([sys.executable, '-u', '-c',
                                         'from pathlib import Path; print(Path.cwd())'], root, log)
            self.assertEqual(status, 0)
            self.assertIn(str(root), log.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
