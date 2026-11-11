"""Quality interventions use only method/, public data, and public training options."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from redteam.common import ROOT, write_json


def train(case, folder):
    seed = int(case.removeprefix('reference_seed')) if case.startswith('reference_seed') else 7
    epochs = 90 if case == 'better_more_training' else 60 if case == 'better_clean_training' else 30
    probability = 0.0 if case == 'better_clean_training' else 0.5
    config = {'seed': seed, 'hidden_size': 192, 'sft_epochs': epochs,
              'dpo_epochs': 1, 'dpo_lr': 5e-5, 'dpo_beta': 0.1,
              'preference_images': 700, 'samples_per_context': 16,
              'icb_steps': 2, 'sample_temperature': 1.4,
              'caption_hallucination_prob': probability}
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'training_config.json', config)
    artifacts = folder / 'artifacts'
    outputs = artifacts / 'trained_outputs'
    checkpoint = artifacts / 'checkpoints'
    # No installation or GPU computation on the login node. Called only in the job.
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    if case == 'weaker_undertrained':
        command = [sys.executable, '-m', 'redteam.train_variant', '--weak-worker', str(folder)]
    else:
        command = [sys.executable, '-m', 'method.train', '--outputs', str(outputs),
                   '--out-dir', str(checkpoint)]
        for key, value in config.items():
            command += ['--' + key.replace('_', '-'), str(value)]
    with (folder / 'training.log').open('w') as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'{case} failed: see {folder}/training.log')
    return outputs


def weak_worker(folder):
    import random
    import torch
    from method.sentinel.model import TinyVLM
    from method.sentinel.training import read_jsonl, train_sft, predict
    from method.sentinel.vocab import OBJECTS, WORDS
    torch.set_num_threads(1)
    random.seed(7)
    torch.manual_seed(7)
    args = argparse.Namespace(seed=7, sft_epochs=1, sft_lr=3e-3, batch_size=64,
                              caption_hallucination_prob=0.5)
    model = TinyVLM(len(OBJECTS), len(WORDS), hidden=192)
    rows = read_jsonl(ROOT / 'data/train_scenes.jsonl')
    train_sft(model, rows, folder / 'artifacts/checkpoints', args)
    predict(model, read_jsonl(ROOT / 'data/eval_scenes.jsonl'),
            folder / 'artifacts/trained_outputs/sentinel_predictions.jsonl', args)
    write_json(folder / 'training_config.json', {
        'seed': 7, 'sft_epochs': 1, 'dpo_epochs': 0, 'hidden_size': 192,
        'note': 'One SFT epoch, preference alignment removed.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weak-worker', type=Path, required=True)
    weak_worker(parser.parse_args().weak_worker)
