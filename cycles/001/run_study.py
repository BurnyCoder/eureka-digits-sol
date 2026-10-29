"""Run the unchanged fixture with a complete timestamped console log.

Global context: this is execution plumbing only; study.py and protocol.json own
the science. The controller owns publication and permission to enter each phase.
Source: https://docs.python.org/3.14/library/subprocess.html (src-python-subprocess).
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def timestamp():
    """Use UTC in log records to make the execution sequence unambiguous."""
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def stream_command(command, root, log_path):
    """Tee merged child output as it arrives and preserve the child's exit code."""
    # Exclusive creation preserves earlier evidence if a caller repeats a log path.
    with log_path.open('x', encoding='utf-8', buffering=1) as log:
        def emit(line):
            """Write the same complete line to both destinations without clipping."""
            print(line, end='', flush=True)
            log.write(line)
            log.flush()

        emit(f'[{timestamp()}] command={json.dumps(command)} cwd={root}\n')
        # No shell is involved; argv arrays handle spaces without command injection.
        with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding='utf-8', bufsize=1) as process:
            for line in process.stdout:
                emit(line)
            status = process.wait()
        emit(f'[{timestamp()}] exit={status}\n')
    return status


def main():
    """Prepare fresh-checkout output paths; delegate every scientific operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('check', 'pilot', 'experiments'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    # The supplied serializer requires its destination directory to exist.
    for name in ('models', 'results', 'reports/logs'):
        (root / name).mkdir(parents=True, exist_ok=True)
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log = root / 'reports/logs' / f'{args.operation}-{suffix}.log'
    return stream_command([sys.executable, '-u', str(root / 'study.py'), args.operation], root, log)


if __name__ == '__main__':
    raise SystemExit(main())
