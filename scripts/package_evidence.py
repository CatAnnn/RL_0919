import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import json_dump, sha256


if __name__ == '__main__':
    files = []
    for folder in ['agents', 'envs', 'federated', 'tests', 'scripts', 'configs', 'data', 'prompts', 'reference_digitized']:
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend(p for p in ROOT.iterdir() if p.suffix in ['.py', '.md', '.yaml', '.txt'])
    archive = ROOT / 'evidence/source_snapshot.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(files):
            zipped.write(path, str(path.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        hashes_ok = all(__import__('hashlib').sha256(zipped.read(str(p.relative_to(ROOT)).replace('\\', '/'))).hexdigest() == sha256(p) for p in files)
    links = []
    for document in [ROOT / 'README.md', ROOT / 'reproduction_report.md', ROOT / 'paper_spec.md']:
        content = document.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)', content):
            if not target.startswith(('http:', 'https:', '#')) and not (document.parent / target.split('#')[0]).exists():
                links.append({'document': document.name, 'target': target})
    original = [ROOT / 'AGENTS.md', ROOT / 'prompt.md', next(ROOT.glob('*.pdf'))]
    original += list((ROOT / 'prompts').rglob('*.md'))
    json_dump(ROOT / 'evidence/package_audit.json', {'source_archive_crc_passed': True,
        'source_archive_hashes_match': hashes_ok, 'source_file_count': len(files), 'broken_links': links,
        'original_input_current_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in original},
        'original_inputs_edit_policy': 'Original files were read only throughout this run; these are final hashes, not a separately captured pre-run baseline.'})
    artifacts = [p for folder in ['checkpoints', 'logs', 'figures', 'data', 'configs'] for p in (ROOT / folder).rglob('*') if p.is_file()]
    artifacts += [ROOT / name for name in ['reproduction_report.md', 'reproduction_state.md', 'README.md', 'assumptions.yaml']]
    json_dump(ROOT / 'evidence/artifact_sha256.json', {str(p.relative_to(ROOT)): sha256(p) for p in artifacts})
    print(json.dumps({'source_file_count': len(files), 'hashes_match': hashes_ok, 'broken_links': links}, indent=2))
    if links or not hashes_ok:
        raise SystemExit(1)
