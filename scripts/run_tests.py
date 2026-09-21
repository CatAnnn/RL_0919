import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    with (ROOT / 'evidence/unit_tests.txt').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print((ROOT / 'evidence/unit_tests.txt').read_text(encoding='utf-8'))
    sys.exit(0 if result.wasSuccessful() else 1)
