"""Воспроизвести зафиксированный срез К07. Только стандартная библиотека Python."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

SOURCE_URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
SLICE_ID = 'K07-ML1M-25to34-v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def order_key(kind, *identifiers):
    key = ':'.join(map(str, ['K07', 'v1', 42, kind, *identifiers]))
    return hashlib.sha256(key.encode('ascii')).digest()


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Официальный ml-1m.zip; без аргумента скачивается с GroupLens')
    parser.add_argument('--out', type=Path, default=Path('local_slice'))
    args = parser.parse_args()
    if args.source:
        archive = args.source.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=90) as response:
            archive = response.read()
    with ZipFile(io.BytesIO(archive)) as source:
        ratings_bytes = source.read('ml-1m/ratings.dat')
        users_bytes = source.read('ml-1m/users.dat')
    # Если архив обновится, не выдаём другой набор за тот же срез.
    reference_path = Path(__file__).with_name('slice_manifest.json')
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding='utf-8'))
        for name, raw in [('ratings.dat', ratings_bytes), ('users.dat', users_bytes)]:
            if digest(raw) != reference['source_sha256'][name]:
                raise ValueError(f'Исходный {name} отличается от зафиксированной версии')
    ages = {}
    for line in users_bytes.decode('latin-1').splitlines():
        uid, _, age, _, _ = line.split('::')
        ages[int(uid)] = int(age)
    ratings = [tuple(map(int, line.split('::')))
               for line in ratings_bytes.decode('ascii').splitlines()]
    assert len(ratings) == 1000209
    assert len({(uid, mid) for uid, mid, _, _ in ratings}) == len(ratings)
    counts = Counter(uid for uid, _, _, _ in ratings)
    eligible = sorted(uid for uid, age in ages.items() if age == 25 and counts[uid] >= 40)
    eligible_set = set(eligible)
    heldout = sorted(sorted(eligible, key=lambda uid: (order_key('user', uid), uid))[:150])
    heldout_set = set(heldout)
    candidate_rows = defaultdict(list)
    main_rows = []
    for source_row, record in enumerate(ratings, start=1):
        uid, mid, rating, timestamp = record
        if uid in eligible_set:
            main_rows.append((source_row, *record))
        if uid in heldout_set:
            candidate_rows[uid].append((source_row, *record))
    control_rows = []
    for uid in heldout:
        selected = sorted(candidate_rows[uid],
                          key=lambda row: (order_key('movie', uid, row[2]), row[2]))[:20]
        control_rows.extend(sorted(selected, key=lambda row: row[2]))
    assert len(heldout) == 150 and len(control_rows) == 3000
    assert all(ages[uid] == 25 and counts[uid] >= 40 for uid in heldout)
    assert all(n == 20 for n in Counter(row[1] for row in control_rows).values())
    assert not (heldout_set & (eligible_set - heldout_set))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    # Файлы с оценками создаются локально; в пакет заявки они не включаются.
    header = ['source_row', 'UserID', 'MovieID', 'Rating', 'Timestamp']
    write_csv(out / 'main_slice.csv', header, main_rows)
    write_csv(out / 'control_slice.csv', header, control_rows)
    # Индексы исходных строк однозначно определяют срез без пересылки оценок.
    (out / 'main_source_rows.txt').write_text(
        ''.join(f'{row[0]}\n' for row in main_rows), encoding='ascii')
    (out / 'control_source_rows.txt').write_text(
        ''.join(f'{row[0]}\n' for row in control_rows), encoding='ascii')
    (out / 'heldout_user_ids.txt').write_text(
        ''.join(f'{uid}\n' for uid in heldout), encoding='ascii')
    manifest = {
        'slice_id': SLICE_ID,
        'source_url': SOURCE_URL,
        'source_sha256': {'archive': digest(archive), 'ratings.dat': digest(ratings_bytes),
                          'users.dat': digest(users_bytes)},
        'main_slice': {'age_code': 25, 'age_label': '25–34', 'minimum_user_ratings': 40,
                       'users': len(eligible), 'rows': len(main_rows),
                       'movies': len({row[2] for row in main_rows})},
        'control_slice': {'users': len(heldout), 'rows': len(control_rows),
                          'movies': len({row[2] for row in control_rows}), 'movies_per_user': 20},
        'development_users_within_main_slice': len(eligible_set - heldout_set),
        'selection': 'Rank SHA256 ASCII K07:v1:42:user:UserID; take first 150. '
                     'For each user rank K07:v1:42:movie:UserID:MovieID; take first 20. '
                     'Break ties by numeric ID. No rating values used for selection.',
        'source_row_numbering': '1-based line numbers in the original ml-1m/ratings.dat, no header',
        'output_sha256': {name: digest((out / name).read_bytes()) for name in [
            'main_slice.csv', 'control_slice.csv', 'main_source_rows.txt',
            'control_source_rows.txt', 'heldout_user_ids.txt']},
    }
    if reference_path.is_file() and manifest != reference:
        raise ValueError('Результат не совпал с зафиксированным манифестом')
    (out / 'slice_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
