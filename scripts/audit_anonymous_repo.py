"""Inspect publication candidates without staging files or running experiments.

Uses Git's current index plus untracked, nonignored files. This is a preflight,
not a guarantee that metadata, bibliography or Git history cannot identify authors.
"""
from pathlib import Path
import json
import hashlib
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    names = sorted(set(git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')) - {''})
    findings = []
    third_party_metadata = []
    size = 0
    count = 0
    for name in names:
        path = ROOT / name
        if path.is_symlink():
            findings.append({'path': name, 'reason': 'symlink requires manual review'})
            continue
        if not path.is_file():
            continue
        count += 1
        size += path.stat().st_size
        if name.startswith(('results/', '.tools/', '.venv/', '.vscode/')):
            findings.append({'path': name, 'reason': 'private or generated directory is tracked'})
        try:
            content = path.read_text()
        except (UnicodeError, OSError):
            continue
        for number, line in enumerate(content.splitlines(), 1):
            if re.search(r'/(?:home|Users)/[^\s/<>]+/', line):
                if path.suffix == '.svg' and hashlib.sha256(path.read_bytes()).hexdigest() == 'f393530d0589c3cba9c596263ccd807acdd86c1446ecd301c1c06bc747a85008':
                    third_party_metadata.append({'path': name, 'line': number, 'reason': 'unchanged upstream Openclipart metadata; reviewed'})
                else:
                    findings.append({'path': name, 'line': number, 'reason': 'personal absolute path'})
        if name == 'paper/paper-preamble.tex':
            for author in re.findall(r'(?m)^\\author\{([^}]+)\}', content):
                if author != 'Anonymous Authors':
                    findings.append({'path': name, 'reason': 'nonanonymous author declaration'})
    tracked_ignored = git('ls-files', '--cached', '--ignored', '--exclude-standard').decode().splitlines()
    report = {'files': count, 'bytes': size, 'findings': findings,
              'tracked_ignored_files': tracked_ignored, 'reviewed_upstream_metadata': third_party_metadata,
              'scope': 'Current publication candidates only; review Git history and binary metadata separately.'}
    print(json.dumps(report, indent=2))
    return int(bool(findings or tracked_ignored))


if __name__ == '__main__':
    raise SystemExit(main())
