"""Validate fresh coupled evidence before permitting performance measurement."""
from pathlib import Path
import hashlib
import json
from gamax.compiler import implementation_identity


def verified_modules(root, cases, seeds, steps, optimization, corpus):
    root, corpus = Path(root), Path(corpus)
    protocol = json.loads((root / 'protocol.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    if not summary.get('complete') or not summary.get('passed'):
        raise ValueError('Incomplete or failed coupled validation')
    if (protocol['implementation_sha256'] != implementation_identity()[1]
            or protocol['steps'] != steps or protocol.get('optimization', 'none') != optimization
            or protocol['parameters'] or protocol['atol'] != 1e-12 or protocol['rtol'] != 1e-10
            or protocol['corpus_sha256'] != hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest()):
        raise ValueError('Validation implementation, corpus, parameters or numeric profile differs')
    for name, digest in protocol['files'].items():
        path = Path(__file__).resolve().parents[1] / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Validated source changed: ' + name)
    registry=json.loads((corpus/'manifest.json').read_text())
    for name,digest in registry['assets'].items():
        if hashlib.sha256((corpus/'official'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Corpus asset changed: '+name)
    result = {}
    for case in cases:
        rows = [r for r in summary['reports'] if r['id'] == case]
        for seed in seeds:
            records = [r for r in rows if r.get('seed') == seed]
            if len(records) != 1:
                raise ValueError('Missing or duplicate seed coverage')
            row = records[0]
            if (row['status'] != 'coupled_passed' or not row['comparison']['passed']
                    or not row['identities']['passed'] or row['comparison']['snapshots'] != steps+1
                    or row['identities']['snapshots'] != steps+1
                    or row['comparison']['float_atol'] != 1e-12
                    or row['comparison']['float_rtol'] != 1e-10):
                raise ValueError('Failed or incomplete trajectory')
        folder = root / case / 'generated'
        manifest = json.loads((folder/'manifest.json').read_text())
        if hashlib.sha256((folder/'model_jax.py').read_bytes()).hexdigest() != manifest['generated_sha256']:
            raise ValueError('Validated module changed')
        if hashlib.sha256(Path(manifest['source']).read_bytes()).hexdigest() != manifest['source_sha256']:
            raise ValueError('Validated GAML source changed')
        result[case] = manifest
    return result
