"""Проверки формул и независимый пересчёт сохранённого аудита."""
import csv
import json
import math
from pathlib import Path
import unittest

import numpy as np

from stage1 import evaluate_log

ROOT = Path(__file__).resolve().parents[1]


def records(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


class Stage1Tests(unittest.TestCase):
    def test_known_wis_and_full_naive_denominator(self):
        reward = np.zeros((1, 20))
        reward[0, 0] = 1
        propensity = np.full(20, 0.025)
        propensity[15:] += 0.1
        result, _, _ = evaluate_log(reward, propensity, np.array([[0, 1, 15, 15]]))
        self.assertEqual(result['wis'], 0.5)
        self.assertEqual(result['naive'], 0.25)
        self.assertEqual(result['ess'], 2)

    def test_absent_target_is_undefined_not_zero(self):
        result, _, _ = evaluate_log(np.zeros((2, 20)), np.full(20, .05),
                                    np.full((2, 4), 15))
        self.assertIsNone(result['wis'])
        self.assertIsNone(result['wis_ci_low'])
        self.assertEqual(result['ess'], 0)
        self.assertEqual(result['users_without_target'], 2)

    def test_saved_log_independent_formulas(self):
        rows = records(ROOT/'results/stage1/example_log.csv')
        run = records(ROOT/'results/stage1/runs.csv')[0]
        w = [float(r['target_probability'])/float(r['behavior_probability']) for r in rows]
        reward = [float(r['reward']) for r in rows]
        total = math.fsum(w)
        wis = math.fsum(a*b for a, b in zip(w, reward))/total
        naive = math.fsum(reward)/len(rows)
        ess = total**2/math.fsum(a*a for a in w)
        self.assertEqual(len(rows), 15000)
        self.assertAlmostEqual(wis, float(run['wis']), places=12)
        self.assertAlmostEqual(naive, float(run['naive']), places=12)
        self.assertAlmostEqual(ess, float(run['ess']), places=9)

    def test_metrics_recomputed_from_all_runs(self):
        rows = records(ROOT/'results/stage1/runs.csv')
        self.assertEqual(len(rows), 500)
        for metric in records(ROOT/'results/stage1/metrics.csv'):
            values = [float(r[metric['method']]) for r in rows]
            oracle = float(metric['oracle'])
            rmse = math.sqrt(math.fsum((v-oracle)**2 for v in values)/len(values))
            self.assertAlmostEqual(rmse, float(metric['rmse']), places=12)

    def test_oracle_without_estimator_or_prepare_tasks(self):
        # Независимый от stage1.py пересчёт сортировки и эталона стандартной библиотекой.
        main = records(ROOT/'data/main_slice.csv')
        control = records(ROOT/'data/control_slice.csv')
        heldout = {r['UserID'] for r in control}
        popularity = {}
        for r in main:
            if r['UserID'] not in heldout:
                popularity[r['MovieID']] = popularity.get(r['MovieID'], 0)+1
        groups = {}
        for r in control:
            groups.setdefault(r['UserID'], []).append(r)
        successes = 0
        for group in groups.values():
            ordered = sorted(group, key=lambda r: (popularity.get(r['MovieID'], 0), int(r['MovieID'])))
            successes += sum(int(r['Rating']) >= 4 for r in ordered[:5])
        self.assertEqual(successes, 357)
        self.assertEqual(successes / 750, .476)
        tasks = [json.loads(line) for line in (ROOT/'results/stage1/audit_tasks.jsonl').read_text().splitlines()]
        self.assertEqual(len(tasks), 150)
        self.assertEqual(len({t['user_id'] for t in tasks}), 150)


if __name__ == '__main__':
    unittest.main()
