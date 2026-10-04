"""Portable, audited manifests for burglary / normal / vandalism videos."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import random

CLASSES = ['burglary', 'normal', 'vandalism']


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative_video(value):
    value = value.strip().replace('\\', '/')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or ':' in value or len(path.parts) < 2:
        raise ValueError(f'Expected a relative class/video path: {value}')
    if path.suffix.lower() not in {'.mp4', '.avi', '.mov', '.mkv'}:
        raise ValueError(f'Unsupported video: {value}')
    return path.as_posix()


def video_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative_video(relative)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing video or path outside video root: {relative}')
    return path


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def validate_rows(rows):
    seen_paths, seen_hashes, groups = set(), {}, {}
    for row in rows:
        row['path'] = relative_video(row['path'])
        if row['label'] not in {'0', '1', '2'} or row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Manifest needs labels 0/1/2 and train/val/test splits.')
        if row['path'].split('/')[0].lower() != CLASSES[int(row['label'])]:
            raise ValueError(f'Folder and label disagree: {row["path"]}')
        if row['path'] in seen_paths or row['sha256'] in seen_hashes:
            raise ValueError('Duplicate path or video hash in manifest.')
        if len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256']):
            raise ValueError('Invalid video SHA-256.')
        seen_paths.add(row['path'])
        seen_hashes[row['sha256']] = row['split']
        if not row['group']:
            raise ValueError('Empty recording group.')
        if row['group'] in groups and groups[row['group']] != row['split']:
            raise ValueError(f'Recording group crosses splits: {row["group"]}')
        groups[row['group']] = row['split']
    for split in ('train', 'val', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != {'0', '1', '2'}:
            raise ValueError(f'{split} must contain all three classes.')
    return rows


def read_manifest(path):
    return validate_rows(read_csv(path))


def prepare(train_csv, test_csv, root, output, groups_csv=None, val_fraction=.2, seed=42):
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists. Use a new output folder to preserve the audited split.')
    if not 0 < val_fraction < 1:
        raise ValueError('Validation fraction must be between 0 and 1.')
    group_map = {}
    if groups_csv:
        for row in read_csv(groups_csv):
            path = relative_video(row['video_path'])
            if path in group_map or not row['group'].strip():
                raise ValueError('Groups CSV needs unique video_path values and nonempty groups.')
            group_map[path] = row['group'].strip()
    rows, hashes, paths, duplicates = [], {}, {}, []
    for source, split in ((train_csv, 'train'), (test_csv, 'test')):
        for row in read_csv(source):
            relative = relative_video(row['video_path'])
            label = row['label'].strip()
            if label not in {'0', '1', '2'} or relative.split('/')[0].lower() != CLASSES[int(label)]:
                raise ValueError(f'Folder and label disagree: {relative}')
            path = video_path(root, relative)
            digest = file_hash(path)
            old = hashes.get(digest) or paths.get(relative)
            if old:
                if old['label'] != label or old['split'] != split:
                    raise ValueError(f'Duplicate has conflicting labels or crosses train/test: {relative}')
                duplicates.append({'path': relative, 'same_as': old['path']})
                continue
            if groups_csv and relative not in group_map:
                raise ValueError(f'Groups CSV has no entry for {relative}')
            record = dict(path=relative, label=label, split=split, sha256=digest,
                          group=group_map.get(relative, relative))
            rows.append(record)
            hashes[digest] = paths[relative] = record
            if len(rows) % 50 == 0:
                print(f'Hashed {len(rows)} videos', flush=True)
    # A group may span labels, but cannot span the supplied train/test boundary.
    owners = {}
    for row in rows:
        if row['group'] in owners and owners[row['group']] != row['split']:
            raise ValueError(f'Recording group crosses supplied train/test: {row["group"]}')
        owners[row['group']] = row['split']
    # Stratify groups by the set of classes they contain. Never split a recording.
    by_group = defaultdict(list)
    for row in rows:
        if row['split'] == 'train':
            by_group[row['group']].append(row)
    strata = defaultdict(list)
    for group, members in by_group.items():
        strata[tuple(sorted({r['label'] for r in members}))].append(group)
    rng = random.Random(seed)
    for signature, groups in sorted(strata.items()):
        groups.sort()
        rng.shuffle(groups)
        if len(groups) < 2:
            raise ValueError(f'Need at least two independent training groups for class set {signature}.')
        count = max(1, min(len(groups) - 1, round(len(groups) * val_fraction)))
        for group in groups[:count]:
            for row in by_group[group]:
                row['split'] = 'val'
    validate_rows(rows)
    output.mkdir(parents=True)
    with (output / 'manifest.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'label', 'split', 'sha256', 'group'])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r['split'], r['path'])))
    report = dict(classes=CLASSES, seed=seed, validation_fraction=val_fraction,
                  counts=dict(Counter(r['split'] + '/' + CLASSES[int(r['label'])] for r in rows)),
                  duplicates_removed=duplicates,
                  input_sha256={'train': file_hash(train_csv), 'test': file_hash(test_csv),
                                'groups': file_hash(groups_csv) if groups_csv else None},
                  grouping='user recording groups' if groups_csv else 'individual files; related recordings not verified',
                  limitations=['Whole-video labels have no event timestamps.',
                               'Exact hashes cannot detect re-encoded copies or related scenes.'])
    (output / 'dataset_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--groups-csv', help='Optional CSV: video_path,group; must cover every retained video')
    parser.add_argument('--val-fraction', type=float, default=.2)
    args = parser.parse_args()
    prepare(args.train_csv, args.test_csv, args.video_root, args.output, args.groups_csv, args.val_fraction)
