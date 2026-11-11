from __future__ import annotations

import argparse
import shutil
import traceback

from redteam.common import AUDIT, ROOT, evaluate, install, read_rows, ref_rows, restore, snapshot, write_json
from redteam.shortcuts import brute_force_object_vector, prepare_shortcut


def run(case):
    snapshot()
    restore()
    folder = AUDIT / case
    folder.mkdir(parents=True, exist_ok=True)
    try:
        settings = {}
        if case == 'reference':
            pass
        elif case.startswith('reference_seed') or case.startswith('better_') or case == 'weaker_undertrained':
            from redteam.train_variant import train
            trained_outputs = train(case, folder)
            for path in trained_outputs.glob('*_predictions.jsonl'):
                shutil.copy2(path, ROOT / 'outputs' / path.name)
        elif case == 'weaker_no_alignment':
            install(ref_rows('baseline'))
        elif case == 'bruteforce_object_vector':
            brute_force_object_vector()
        elif case.startswith('shortcut_'):
            settings = prepare_shortcut(case)
        else:
            raise ValueError(case)
        evaluate(folder, **settings)
    except Exception as exc:
        write_json(folder / 'error.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        restore()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    run(parser.parse_args().case)
