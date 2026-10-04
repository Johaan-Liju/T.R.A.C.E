"""Colab dataset staging: version-pinned downloads, storage checks and 7z extraction."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

HANDLE = 'kizitoryan/burglary-and-vandalism-detection-dataset/versions/1'
PREFIX = 'Burglary-and-Vandalism-dataset'
PART_SIZES = [20401094656] * 4 + [362741883]
GIB = 1024 ** 3


def inspect_listing(listing):
    """Read file records, not the archive header, from `7z l -slt`."""
    listing = listing.replace('\r\n', '\n')
    if '\n----------\n' not in listing:
        raise ValueError('Cannot read 7z file listing; extraction has not started.')
    records = listing.split('\n----------\n', 1)[1]
    total, count = 0, 0
    for block in records.split('\n\n'):
        fields = dict(line.split(' = ', 1) for line in block.splitlines() if ' = ' in line)
        if 'Path' not in fields:
            continue
        path = fields['Path'].replace('\\', '/')
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or '..' in parsed.parts or ':' in path:
            raise ValueError(f'Unsafe archive path: {path}')
        if fields.get('Symbolic Link') or fields.get('Hard Link') or 'l' in fields.get('Mode', '')[:1]:
            raise ValueError('Archive links are not supported.')
        total += int(fields.get('Size') or 0)
        count += 1
    if not count or total <= 0:
        raise ValueError('Archive has no usable file inventory.')
    return total


def on_drive(path):
    return str(Path(path).resolve()).startswith('/content/drive/')


def check_space(path, required):
    free = shutil.disk_usage(path).free
    if on_drive(path):
        print(f'Google Drive needs at least {required / GIB:.1f} GiB of available quota at {path}. '
              'The mounted filesystem cannot reliably report your account quota.', flush=True)
    elif free < required:
        raise ValueError(f'Insufficient space at {path}: {free / GIB:.1f} GiB free, '
                         f'{required / GIB:.1f} GiB required. Put archives on a Drive with enough '
                         'quota, use a larger runtime, or point to already extracted videos.')


def fetch(relative, output, size=None):
    import kagglehub
    target = Path(output) / PREFIX / relative
    if target.is_file() and (size is None or target.stat().st_size == size):
        return target
    if target.exists():
        raise ValueError(f'Incomplete or unexpected file: {target}. Remove this file before retrying.')
    kagglehub.dataset_download(HANDLE, path=f'{PREFIX}/{relative}', output_dir=str(output))
    if not target.is_file() or (size is not None and target.stat().st_size != size):
        raise ValueError(f'Download size or output path differs from dataset version 1: {target}')
    return target


def find_video_root(extracted):
    extracted = Path(extracted)
    candidates = [extracted, extracted / 'dataset', extracted / PREFIX / 'dataset']
    for root in candidates:
        if all((root / name).is_dir() for name in ('burglary', 'normal', 'vandalism')):
            return root.resolve()
    raise ValueError(f'Cannot find burglary/normal/vandalism directories under {extracted}. '
                     'Set VIDEO_ROOT_OVERRIDE to their parent directory.')


def stage(metadata, archives, extracted, existing=None):
    metadata, archives, extracted = map(Path, (metadata, archives, extracted))
    metadata.mkdir(parents=True, exist_ok=True)
    train = fetch('train.csv', metadata)
    test = fetch('test.csv', metadata)
    if existing:
        root = find_video_root(existing)
    else:
        archives.mkdir(parents=True, exist_ok=True)
        extracted.mkdir(parents=True, exist_ok=True)
        parts = [archives / PREFIX / f'dataset.7z.{i:03d}' for i in range(1, 6)]
        remaining = sum(size for path, size in zip(parts, PART_SIZES)
                        if not path.is_file() or path.stat().st_size != size)
        # Conservative initial allowance; the actual unpacked size is checked after listing.
        required = remaining + 5 * GIB
        marker = extracted / '.trace-extracted-v1.json'
        if archives.stat().st_dev == extracted.stat().st_dev and not marker.exists():
            required += sum(PART_SIZES)
        check_space(archives, required)
        for i, size in enumerate(PART_SIZES, 1):
            print(f'Archive part {i}/5', flush=True)
            fetch(f'dataset.7z.{i:03d}', archives, size)
        seven_zip = shutil.which('7z') or shutil.which('7zz')
        if not seven_zip:
            raise ValueError('Install 7-Zip first (Colab: apt-get install -y p7zip-full).')
        if not marker.exists():
            result = subprocess.run([seven_zip, 'l', '-slt', str(parts[0])],
                                    check=True, capture_output=True, text=True)
            size = inspect_listing(result.stdout)
            check_space(extracted, size + 5 * GIB)
            # Overwrite incomplete files on an interrupted extraction; never skip partial videos.
            subprocess.run([seven_zip, 'x', str(parts[0]), f'-o{extracted}', '-aoa', '-y'], check=True)
            marker.write_text(json.dumps({'dataset': HANDLE, 'archive_sizes': PART_SIZES}), encoding='utf-8')
        elif json.loads(marker.read_text()) != {'dataset': HANDLE, 'archive_sizes': PART_SIZES}:
            raise ValueError('Extraction marker belongs to a different dataset.')
        root = find_video_root(extracted)
    state = dict(video_root=str(root), train_csv=str(train.resolve()), test_csv=str(test.resolve()))
    (metadata / 'staged.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    print(json.dumps(state, indent=2))
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--archives', required=True)
    parser.add_argument('--extract-to', required=True)
    parser.add_argument('--existing', help='Existing parent of burglary, normal and vandalism folders')
    args = parser.parse_args()
    stage(args.metadata, args.archives, args.extract_to, args.existing)
