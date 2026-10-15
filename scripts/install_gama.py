#!/usr/bin/env python3
"""Download the pinned official Linux/x86_64 GAMA release locally, without system install."""
from pathlib import Path
import hashlib
import platform
import urllib.request
import zipfile
import stat

VERSION = '2025.06.4'
ARCHIVE = f'GAMA_{VERSION}_Linux_with_JDK.zip'
URL = f'https://github.com/gama-platform/gama/releases/download/{VERSION}/{ARCHIVE}'
# Official release checksum.txt SHA256, recorded with the evaluation artifact.
SHA256 = '043a81de03a79e0a71055c20fc1a1d8d6d290045e8c7ea7162aa5e2be622834f'
ROOT = Path(__file__).resolve().parents[1]


def main():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise SystemExit('This installer pins the Linux x86_64 release; use a matching official distribution on other platforms.')
    tools = ROOT / '.tools'; tools.mkdir(exist_ok=True)
    archive = tools / 'gama.zip'
    if not archive.exists():
        print(f'Downloading {URL}', flush=True)
        temporary = archive.with_suffix('.part')
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(archive)
    with archive.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
        digest = digest.hexdigest()
    if digest != SHA256:
        raise SystemExit(f'Archive checksum mismatch: {digest}; remove the incomplete archive before retrying.')
    destination = tools / 'gama'
    with zipfile.ZipFile(archive) as file:
        for info in file.infolist():
            target = destination / info.filename
            if not target.resolve().is_relative_to(destination.resolve()):
                raise SystemExit('Unsafe archive entry')
            file.extract(info, destination)
            mode = info.external_attr >> 16
            if stat.S_ISREG(mode): target.chmod(stat.S_IMODE(mode))
    print(destination / 'headless/gama-headless.sh')


if __name__ == '__main__': main()
