"""Read a CASA training queue over SSH without uploading files."""
import argparse
from pathlib import Path
import shlex
import sys

from ssh import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='CASA-Pyramid')
    parser.add_argument('--run', default='Run2')
    parser.add_argument('--json', action='store_true', help='Return structured JSON')
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    source = Path(__file__).with_name('server_status.py').read_text(encoding='utf-8')
    arguments = ['python3', '-', '--tag', args.tag, '--run', args.run]
    if args.json:
        arguments.append('--json')
    command = ' '.join(shlex.quote(part) for part in arguments)
    command += " <<'CASA_STATUS_PY'\n" + source + '\nCASA_STATUS_PY\n'
    code, output, error = run(command, args.timeout)
    print(output, end='')
    print(error, end='', file=sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
