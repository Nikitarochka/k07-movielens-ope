"""Этап 1: воспроизводимый аудит WIS и наивной средней на фиксированном срезе."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import time

import numpy as np

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def prepare_tasks(data_dir):
    """Политики строятся без контрольных оценок; награды раскрываются для аудита."""
    manifest = json.loads((data_dir / 'slice_manifest.json').read_text(encoding='utf-8'))
    for name in ['main_slice.csv', 'control_slice.csv']:
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != manifest['output_sha256'][name]:
            raise ValueError(f'Контрольная сумма {name} не совпала')
    main = read_csv(data_dir / 'main_slice.csv')
    control = read_csv(data_dir / 'control_slice.csv')
    heldout = sorted({int(row['UserID']) for row in control})
    heldout_set = set(heldout)
    popularity = {}
    development = set()
    for row in main:
        uid, movie = int(row['UserID']), int(row['MovieID'])
        if uid not in heldout_set:
            development.add(uid)
            popularity[movie] = popularity.get(movie, 0) + 1
    assert len(main) == 383928 and len(control) == 3000
    assert len(heldout) == 150 and len(development) == 1546
    main_rows = {int(row['source_row']) for row in main}
    assert all(int(row['source_row']) in main_rows for row in control)
    grouped = {uid: [] for uid in heldout}
    for row in control:
        grouped[int(row['UserID'])].append(row)
    rewards, tasks = [], []
    for task_id, uid in enumerate(heldout):
        rows = sorted(grouped[uid], key=lambda r: (popularity.get(int(r['MovieID']), 0), int(r['MovieID'])))
        assert len(rows) == len({r['MovieID'] for r in rows}) == 20
        reward = [int(int(row['Rating']) >= 4) for row in rows]
        rewards.append(reward)
        tasks.append({
            'task_id': task_id, 'user_id': uid,
            'source_rows': [int(row['source_row']) for row in rows],
            'target_action_indices': list(range(5)),
            'popular_action_indices': list(range(15, 20)),
            'target_value': sum(reward[:5]) / 5,
        })
    reward = np.asarray(rewards, dtype=float)
    assert reward[:, :5].sum() == 357
    return reward, tasks


def evaluate_log(reward, propensity, actions, confidence=0.95):
    """WIS и naive; нормальные интервалы с учётом фиксированного числа событий/user."""
    users, per_user = actions.shape
    observed = np.take_along_axis(reward, actions, axis=1)
    weights = (actions < 5) * 0.2 / propensity[actions]
    total_weight = weights.sum()
    naive = float(observed.mean())
    # Пользователи фиксированы, случайны только независимые действия внутри каждого.
    naive_se = float(np.sqrt(per_user * observed.var(axis=1, ddof=1).sum()) / actions.size)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    result = {'naive': naive, 'naive_se': naive_se,
              'naive_ci_low': naive - z * naive_se, 'naive_ci_high': naive + z * naive_se,
              'target_events': int((actions < 5).sum()),
              'users_without_target': int((~(actions < 5).any(axis=1)).sum())}
    if total_weight == 0:
        result.update(wis=None, wis_se=None, wis_ci_low=None, wis_ci_high=None, ess=0.0)
    else:
        wis = float((weights * observed).sum() / total_weight)
        residual = weights * (observed - wis)
        # Стратифицированный delta-method для отношения двух сумм.
        se = float(np.sqrt(per_user * residual.var(axis=1, ddof=1).sum()) / total_weight)
        result.update(wis=wis, wis_se=se, wis_ci_low=wis-z*se, wis_ci_high=wis+z*se,
                      ess=float(total_weight**2 / np.square(weights).sum()))
    return result, observed, weights


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run(config, reward, tasks, output):
    users = len(tasks)
    n = config['events_per_user']
    epsilon = config['epsilon']
    if not (0 <= epsilon <= 1) or n < 2:
        raise ValueError('Нужны epsilon в [0,1] и не менее двух событий на пользователя')
    propensity = np.full(20, epsilon / 20)
    propensity[15:] += (1 - epsilon) / 5
    np.testing.assert_allclose(propensity.sum(), 1)
    oracle = float(reward[:, :5].mean())
    sequences = np.random.SeedSequence(config['seed']).spawn(2)
    pilot_rng = np.random.default_rng(sequences[0])
    for _ in range(config['pilot_repetitions']):
        actions = pilot_rng.choice(20, (users, n), p=propensity)
        result, observed, weights = evaluate_log(reward, propensity, actions, config['confidence_level'])
        selected = actions < 5
        if selected.any():
            # В этом протоколе все ненулевые веса одинаковы: WIS = среднее целевых наград.
            np.testing.assert_allclose(result['wis'], observed[selected].mean())
            np.testing.assert_allclose(result['ess'], selected.sum())
    rows = []
    main_sequences = sequences[1].spawn(config['repetitions'])
    for repetition, seq in enumerate(main_sequences):
        rng = np.random.default_rng(seq)
        actions = rng.choice(20, (users, n), p=propensity)
        result, observed, weights = evaluate_log(reward, propensity, actions, config['confidence_level'])
        rows.append({'repetition': repetition, 'spawn_key': '.'.join(map(str, seq.spawn_key)),
                     'epsilon': epsilon, **result})
        if repetition == 0:
            example = []
            for uid in range(users):
                for event, action in enumerate(actions[uid]):
                    example.append({'task_id': uid, 'event': event, 'action_index': int(action),
                                    'reward': int(observed[uid, event]),
                                    'behavior_probability': float(propensity[action]),
                                    'target_probability': 0.2 if action < 5 else 0.0})
            write_csv(output / 'example_log.csv', example)
    write_csv(output / 'runs.csv', rows)
    with (output / 'audit_tasks.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for task in tasks:
            stream.write(json.dumps(task, ensure_ascii=False) + '\n')
    bootstrap_rng = np.random.default_rng(20261125)
    indices = bootstrap_rng.integers(0, len(rows), (config['summary_bootstrap_repetitions'], len(rows)))
    metrics = []
    alpha = (1 - config['confidence_level']) / 2
    for method in ['wis', 'naive']:
        valid = np.array([r[method] is not None for r in rows])
        if not valid.all():
            raise ValueError('Неопределённые запуски нельзя молча исключать из основного аудита')
        values = np.array([r[method] for r in rows])
        errors = values - oracle
        rmse_boot = np.sqrt(np.square(errors)[indices].mean(axis=1))
        bias_boot = errors[indices].mean(axis=1)
        metrics.append({
            'method': method, 'oracle': oracle, 'mean_estimate': float(values.mean()),
            'bias': float(errors.mean()), 'bias_ci_low': float(np.quantile(bias_boot, alpha)),
            'bias_ci_high': float(np.quantile(bias_boot, 1-alpha)),
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'rmse_ci_low': float(np.quantile(rmse_boot, alpha)),
            'rmse_ci_high': float(np.quantile(rmse_boot, 1-alpha)),
            'mae': float(np.abs(errors).mean()), 'sd': float(values.std(ddof=1)),
            'ci_coverage': float(np.mean([r[f'{method}_ci_low'] <= oracle <= r[f'{method}_ci_high'] for r in rows])),
            'undefined_fraction': float(1 - valid.mean()),
        })
    write_csv(output / 'metrics.csv', metrics)
    worst = sorted(rows, key=lambda row: abs(row['wis'] - oracle), reverse=True)[:5]
    write_csv(output / 'error_cases.csv', [{**row, 'oracle': oracle, 'wis_error': row['wis']-oracle} for row in worst])
    summary = {'config': config, 'oracle': oracle, 'users': users,
               'development_users': 1546, 'events_per_run': users*n,
               'mean_ess': float(np.mean([r['ess'] for r in rows])),
               'metrics': metrics}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/stage1.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/stage1')
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    reward, tasks = prepare_tasks(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = run(config, reward, tasks, args.output)
    elapsed = time.perf_counter()-started
    print(json.dumps({'seconds': round(elapsed, 2), **summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
