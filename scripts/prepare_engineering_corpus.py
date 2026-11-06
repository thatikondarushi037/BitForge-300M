"""Generate a small, offline CC0 procedural engineering corpus; no downloads."""
import hashlib
import json
from pathlib import Path
import random


def main():
    root = Path(__file__).resolve().parents[1] / 'data'
    path = root / 'engineering_corpus.txt'
    rng = random.Random(606)
    subjects = ['bearing', 'sensor', 'pump', 'relay', 'filter', 'valve', 'rotor', 'bracket', 'cable', 'housing']
    actions = ['inspect', 'measure', 'replace', 'align', 'clean', 'test', 'record', 'adjust']
    paragraphs = []
    for i in range(700):
        subject, action = rng.choice(subjects), rng.choice(actions)
        paragraphs.append(f'Workshop record {i:04d}. The technician will {action} the {subject} before the next trial. '
                          f'The reference reading is {rng.randrange(10, 990)} units at {rng.randrange(15, 45)} degrees. '
                          'Record the initial condition, isolate the power supply, and compare the final reading with the reference. '
                          f'If the {subject} moves during inspection, stop the trial and check its mounting. '
                          'A second operator reviews the notes before work continues.')
    raw = ('\n\n'.join(paragraphs)+'\n').encode()
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError('refusing to replace different corpus')
    path.write_bytes(raw)
    manifest = {'name': 'Procedural engineering notebook', 'source': 'Original templates written for BitForge; generated offline by scripts/prepare_engineering_corpus.py, seed 606; no external passages',
                'license': 'CC0-1.0', 'license_url': 'https://creativecommons.org/publicdomain/zero/1.0/',
                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'paragraphs': len(paragraphs),
                'limitations': 'Synthetic template text, repeated phrasing; not a natural-language benchmark. Split before windowing; no exact duplicate paragraphs.'}
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
