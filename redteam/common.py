from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
STATE = AUDIT / '_initial'
BASE_COMMIT = 'e53c7dee90d376c376ab3ce0fc735594da49e5cd'
EXPERIMENT = 'synthetic_object_halbench'
OBJECTS = [f'{c} {s}' for c in ['red', 'blue', 'green', 'yellow']
           for s in ['circle', 'square', 'triangle']]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def snapshot():
    """Save exactly the files the audit temporarily changes, including ignored outputs."""
    if (STATE / 'ready.json').exists():
        return
    STATE.mkdir(parents=True, exist_ok=True)
    names = ['data/eval_scenes.jsonl', 'scoring/scores_train.json', 'scoring/scores_test.json']
    for name in names:
        src = ROOT / name
        dst = STATE / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copytree(ROOT / 'outputs', STATE / 'outputs', dirs_exist_ok=True)
    write_json(STATE / 'ready.json', {'files': names, 'base_commit': BASE_COMMIT})


def restore():
    state = json.loads((STATE / 'ready.json').read_text())
    for name in state['files']:
        shutil.copy2(STATE / name, ROOT / name)
    # Only our default prediction artifacts are temporarily replaced.
    for path in (ROOT / 'outputs').glob('*_predictions.jsonl'):
        path.unlink()
    for path in (STATE / 'outputs').iterdir():
        if path.is_file():
            shutil.copy2(path, ROOT / 'outputs' / path.name)


def ref_rows(name='sentinel'):
    # Reusing the trained reference's outputs is explicitly authorized for checks
    # that change only submitted predictions. No hidden score or test label is read.
    return read_rows(STATE / 'outputs' / f'{name}_predictions.jsonl')


def install(rows, name='sentinel'):
    write_rows(ROOT / 'outputs' / f'{name}_predictions.jsonl', rows)


def original_test_is_installed():
    for name in ['scripts/evaluate_test.sh', 'eval/test/evaluate.py', 'eval/train/evaluate.py']:
        original = subprocess.check_output(['git', 'show', f'{BASE_COMMIT}:{name}'], cwd=ROOT)
        if (ROOT / name).read_bytes() != original:
            raise RuntimeError(f'Audit requires the original evaluator: {name}')


def evaluate(folder, *, manifest_names=None, forge_scores=False, primary='sentinel'):
    """Run the real, no-argument scripts in independent processes, then save evidence."""
    original_test_is_installed()
    statuses = {}
    for phase in ['train', 'test']:
        if phase == 'test':
            if manifest_names is not None:
                write_json(ROOT / 'scoring/scores_train.json', {
                    'experiments': {EXPERIMENT: {'results': {name: {} for name in manifest_names}}}})
            if forge_scores:
                write_json(ROOT / 'scoring/scores_test.json', {
                    'experiments': {EXPERIMENT: {'results': {'sentinel': {
                        'response_hallucination_rate': -1e9,
                        'mention_hallucination_rate': -1e9,
                        'object_recall': 1e9, 'avg_object_mentions': 1e9, 'qa_accuracy': 1e9}}}}})
        with (folder / f'{phase}.log').open('w') as log:
            completed = subprocess.run(['bash', f'scripts/evaluate_{phase}.sh'],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        statuses[phase] = completed.returncode
        shutil.copy2(ROOT / f'scoring/scores_{phase}.json', folder / f'scores_{phase}.json')
    scored = json.loads((folder / 'scores_test.json').read_text())
    results = scored.get('experiments', {}).get(EXPERIMENT, {}).get('results', {})
    summary = {'exit_codes': statuses, 'primary_method': primary,
               'methods_on_disk': sorted(p.name.removesuffix('_predictions.jsonl')
                                         for p in (ROOT / 'outputs').glob('*_predictions.jsonl')),
               'methods_scored': sorted(results),
               'score_is_fresh': statuses['test'] == 0,
               'metrics': results.get(primary) if statuses['test'] == 0 else None}
    write_json(folder / 'result.json', summary)
    artifacts = folder / 'artifacts' / 'submitted_outputs'
    artifacts.mkdir(parents=True, exist_ok=True)
    for src in (ROOT / 'outputs').glob('*_predictions.jsonl'):
        shutil.copy2(src, artifacts / src.name)
    shutil.copy2(ROOT / 'data/eval_scenes.jsonl', folder / 'artifacts' / 'submitted_gold.jsonl')
    print(json.dumps({'case': folder.name, **summary}), flush=True)
    return summary
