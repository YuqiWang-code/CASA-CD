"""SSH access using the project's local SFTP configuration."""
import argparse
import json
from pathlib import Path
import sys

import paramiko


ROOT = Path(__file__).resolve().parent.parent


def _conn():
    config = json.loads((ROOT / '.vscode' / 'sftp.json').read_text(encoding='utf-8'))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(config['host'], port=config.get('port', 22),
                   username=config['username'], password=config['password'], timeout=30)
    return client


def run(command, timeout=120):
    client = _conn()
    try:
        _, stdout, stderr = client.exec_command(command, timeout=timeout)
        output = stdout.read().decode('utf-8', errors='replace')
        error = stderr.read().decode('utf-8', errors='replace')
        return stdout.channel.recv_exit_status(), output, error
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', help='Shell command or local UTF-8 command file')
    parser.add_argument('timeout', nargs='?', type=int, default=120)
    args = parser.parse_args()
    candidate = Path(args.command)
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    command = candidate.read_text(encoding='utf-8') if is_file else args.command
    code, output, error = run(command, args.timeout)
    print(output, end='')
    print(error, end='', file=sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
