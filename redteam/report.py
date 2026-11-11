"""Audit analysis only. Never called by a submitted shortcut method."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from redteam.common import AUDIT, EXPERIMENT, ROOT, write_json


def main():
    reference_config = json.loads((ROOT / 'scoring/reference.json').read_text())
    experiment = reference_config['experiments'][EXPERIMENT]
    config = experiment['metrics']
    runs = {}
    for path in sorted(AUDIT.glob('*/result.json')):
        result = json.loads(path.read_text())
        runs[path.parent.name] = result
    reference = runs['reference']['metrics']
    valid = {name: result['metrics'] for name, result in runs.items()
             if result['score_is_fresh'] and result['metrics'] is not None}
    # Use a disclosed empirical cohort; the eventual ranked-attempt distribution
    # does not exist yet and its standard deviations cannot be known in advance.
    spreads = {metric: statistics.pstdev(row[metric] for row in valid.values())
               for metric in config}

    def rank(row, std):
        return experiment['weight'] * sum(
            abs(conf['coefficient']) * (1 if conf['higher_is_better'] else -1) *
            (row[metric] - reference[metric]) / std[metric]
            for metric, conf in config.items() if std[metric] > 0)

    for name, row in valid.items():
        runs[name]['audit_cohort_rank_vs_reference'] = rank(row, spreads)
        # Also show a reference-plus-this-attempt cohort, which is a valid future
        # cohort and exposes whether trading off total task failure can pay.
        pair_std = {metric: statistics.pstdev([reference[metric], row[metric]])
                    for metric in config}
        runs[name]['pairwise_cohort_rank_vs_reference'] = rank(row, pair_std)
    seed_names = ['reference_seed7', 'reference_seed17', 'reference_seed27']
    seeds = {name: valid[name] for name in seed_names if name in valid}
    seed_summary = {}
    for metric in config:
        values = [row[metric] for row in seeds.values()]
        if values:
            seed_summary[metric] = {'min': min(values), 'max': max(values),
                                    'range': max(values) - min(values)}
    seed_ranks = [runs[name]['audit_cohort_rank_vs_reference'] for name in seeds]
    summary = {'reference': reference, 'runs': runs,
               'rank_formula': 'weight * sum(abs(coefficient) * direction * (value-reference)/population_std)',
               'rank_warning': 'Ranks are conditional on the named audit cohort, not unknowable future attempt variances.',
               'audit_cohort': sorted(valid), 'audit_cohort_std': spreads,
               'seed_metrics': seeds, 'seed_metric_spread': seed_summary,
               'seed_rank_range': max(seed_ranks) - min(seed_ranks) if seed_ranks else None,
               'missing_test_experiments': sorted(set(reference_config['experiments']) - {EXPERIMENT})}
    # A second representative cohort isolates honest method variation + silence.
    honest_names = [n for n in valid if n == 'reference' or n.startswith(('reference_seed', 'better_', 'weaker_'))]
    if 'shortcut_empty_caption' in valid:
        names = honest_names + ['shortcut_empty_caption']
        std = {m: statistics.pstdev(valid[n][m] for n in names) for m in config}
        summary['honest_plus_empty_cohort'] = {
            'members': names, 'std': std,
            'ranks': {n: rank(valid[n], std) for n in names}}
    write_json(AUDIT / 'summary.json', summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
