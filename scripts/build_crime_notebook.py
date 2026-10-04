"""Build a self-contained Colab notebook from the current TRACE source."""
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build():
    sources = [ROOT / 'trace' / name for name in (
        '__init__.py', 'violence.py', 'crime.py', 'crime_data.py', 'crime_download.py',
        'train_crime.py', 'predict_crime.py')]
    sources += [ROOT / 'tests/test_crime.py', ROOT / 'docs/CRIME_TRAINING.md']
    archive = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sources:
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            hashes[relative] = hashlib.sha256(data).hexdigest()
            # Fixed metadata makes regeneration deterministic.
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
    payload = base64.b64encode(archive.getvalue()).decode('ascii')
    cells = []

    def markdown(source):
        cells.append(dict(cell_type='markdown', metadata={}, source=textwrap.dedent(source).strip() + '\n'))

    def code(source):
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                          source=textwrap.dedent(source).strip() + '\n'))

    markdown('''
        # TRACE: train burglary / normal / vandalism in Colab

        This notebook trains a **new three-class video classifier**. It uses a frozen R3D-18
        encoder and learns attention over 32 sampled windows per video. Labels apply to
        entire videos; attention weights are not event timestamps. The existing violence
        classifier remains a separate model.

        **Start:** Runtime → Change runtime type → GPU. Run cells in order.
        This notebook embeds the project source, so no GitHub push or clone is necessary.

        **Storage first:** the dataset's five archive parts alone total **82 GB (76 GiB)**,
        plus extracted-video space. Free Colab may not fit both. You can keep archives on
        a Google Drive with enough quota, extract to the runtime, or use existing extracted
        videos. The notebook stops before a large local download if space is insufficient.
        Drive quotas must be checked in your account; the mount's reported capacity is unreliable.

        Dataset: [Kizito Ryan, version 1](https://www.kaggle.com/datasets/kizitoryan/burglary-and-vandalism-detection-dataset/data).
        Published labels: burglary=100, normal=950, vandalism=50. Actual files are audited below.
        If Kaggle authentication is needed, add `KAGGLE_API_TOKEN` to Colab Secrets and allow
        notebook access. Never paste credentials in a cell.
    ''')
    markdown('''
        ## 1. Install the download helper and check the GPU

        Keep Colab's paired PyTorch/torchvision installation to preserve CUDA support.
        The training pipeline also needs NumPy and OpenCV, normally present in Colab.
    ''')
    code('''
        import importlib.util
        import subprocess
        import sys

        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'], check=True)
        if importlib.util.find_spec('cv2') is None:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'opencv-python-headless'], check=True)
        import torch
        import torchvision
        import cv2
        import numpy as np
        print('Python:', sys.version.split()[0])
        print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__)
        print('OpenCV:', cv2.__version__, 'NumPy:', np.__version__)
        if not torch.cuda.is_available():
            raise RuntimeError('Enable a GPU under Runtime > Change runtime type, then rerun.')
        print('GPU:', torch.cuda.get_device_name(0))
    ''')
    markdown('''
        ## 2. Restore the embedded source and run a small self-check

        This writes only inside `/content/TRACE_crime_code`. The self-check uses tiny
        synthetic videos and a test encoder; passing it does not measure model accuracy.
    ''')
    code(f'''
        import base64
        import hashlib
        import io
        import json
        import os
        from pathlib import Path
        import zipfile

        CODE_ROOT = Path('/content/TRACE_crime_code')
        CODE_ROOT.mkdir(parents=True, exist_ok=True)
        SOURCE_HASHES = {hashes!r}
        SOURCE_ZIP = base64.b64decode({payload!r})
        with zipfile.ZipFile(io.BytesIO(SOURCE_ZIP)) as archive:
            for member in archive.infolist():
                target = (CODE_ROOT / member.filename).resolve()
                if not target.is_relative_to(CODE_ROOT.resolve()):
                    raise ValueError('Unexpected source archive path')
                data = archive.read(member)
                if hashlib.sha256(data).hexdigest() != SOURCE_HASHES[member.filename]:
                    raise ValueError('Source checksum mismatch')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        os.chdir(CODE_ROOT)
        # Separate Python processes avoid Jupyter's standard-library `trace` module collision.
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
    ''')
    markdown('''
        ## 3. Mount Drive and configure storage

        Edit these paths before continuing. Set `VIDEO_ROOT_OVERRIDE` if you already have
        the extracted `burglary/`, `normal/`, `vandalism/` directories.

        Otherwise, download archives either to `/content/trace_archives` (requires a large
        runtime disk) or `DRIVE_ROOT / 'archives'` (requires at least 82 GB of Drive quota).
        Extracting to local `/content` usually gives faster video reads. The extraction
        cell checks the actual uncompressed size before starting.

        Use a new `RUN_NAME` when changing data, source or hyperparameters. Keep it unchanged
        to resume an interrupted run. The test set should be evaluated only after you have
        settled on settings: set `EVALUATE_TEST=False` during experimentation.
    ''')
    code('''
        from google.colab import drive
        drive.mount('/content/drive')

        DRIVE_ROOT = Path('/content/drive/MyDrive/TRACE_crime')
        ARCHIVE_DIR = Path('/content/trace_archives')
        # Alternative when Drive has enough quota:
        # ARCHIVE_DIR = DRIVE_ROOT / 'archives'
        EXTRACT_DIR = Path('/content/trace_videos')
        VIDEO_ROOT_OVERRIDE = ''  # e.g. '/content/drive/MyDrive/dataset'
        GROUPS_CSV = ''  # optional complete video_path,group CSV for related recordings

        RUN_NAME = 'crime_baseline'
        CLIPS = 32
        EPOCHS = 200
        EVALUATE_TEST = True
        if not RUN_NAME or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in RUN_NAME):
            raise ValueError('Use letters, digits, underscores or hyphens for RUN_NAME.')
        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
        METADATA_DIR = DRIVE_ROOT / 'dataset_metadata'
        AUDIT_DIR = DRIVE_ROOT / 'audit' / RUN_NAME
        RUN_DIR = DRIVE_ROOT / 'runs' / RUN_NAME
        CACHE_DIR = DRIVE_ROOT / 'feature_cache'
        print('Persistent outputs:', RUN_DIR)
    ''')
    markdown('''
        ## 4. Download or locate the dataset

        The dataset is version-pinned. All five archive parts are needed for extraction.
        Completed downloads and extraction are reused. No video archives are downloaded
        when `VIDEO_ROOT_OVERRIDE` is set. If space is insufficient, change the storage
        paths above and rerun this cell. Downloads can take considerable time.
    ''')
    code('''
        import shutil
        if not VIDEO_ROOT_OVERRIDE and not (shutil.which('7z') or shutil.which('7zz')):
            subprocess.run(['apt-get', 'update', '-qq'], check=True)
            subprocess.run(['apt-get', 'install', '-y', '-qq', 'p7zip-full'], check=True)
        command = [sys.executable, '-m', 'trace.crime_download',
                   '--metadata', str(METADATA_DIR), '--archives', str(ARCHIVE_DIR),
                   '--extract-to', str(EXTRACT_DIR)]
        if VIDEO_ROOT_OVERRIDE:
            command += ['--existing', VIDEO_ROOT_OVERRIDE]
        subprocess.run(command, check=True)
        STAGED = json.loads((METADATA_DIR / 'staged.json').read_text())
        VIDEO_ROOT = Path(STAGED['video_root'])
    ''')
    markdown('''
        ## 5. Audit the labels, files and splits

        This hashes every video, preserves the publisher's test assignment and reserves
        20% of training groups for validation. Exact cross-split duplicates or conflicting
        labels stop the audit. Do not bypass these errors: review the data and create a
        corrected split. Related recordings require `GROUPS_CSV`; otherwise each file is
        treated as one group. The publisher's random split is not verified camera-independent.

        The supplied labels have no event intervals. Review examples from each class before
        committing to a long run. Hashing 82 GB of video can take time, especially from Drive.
    ''')
    code('''
        def sha256_file(path):
            digest = hashlib.sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest()

        manifest = AUDIT_DIR / 'manifest.csv'
        if manifest.exists():
            audit = json.loads((AUDIT_DIR / 'dataset_report.json').read_text())
            expected = {'train': sha256_file(STAGED['train_csv']),
                        'test': sha256_file(STAGED['test_csv']),
                        'groups': sha256_file(GROUPS_CSV) if GROUPS_CSV else None}
            if audit['input_sha256'] != expected:
                raise ValueError('Split inputs changed. Choose a new RUN_NAME.')
            print('Reusing audited manifest; training rechecks video hashes.')
        else:
            command = [sys.executable, '-m', 'trace.crime_data',
                       '--train-csv', STAGED['train_csv'], '--test-csv', STAGED['test_csv'],
                       '--video-root', str(VIDEO_ROOT), '--output', str(AUDIT_DIR)]
            if GROUPS_CSV:
                command += ['--groups-csv', GROUPS_CSV]
            subprocess.run(command, check=True)
        print((AUDIT_DIR / 'dataset_report.json').read_text())
    ''')
    markdown('''
        ## 6. Download the frozen video encoder

        The classifier checkpoint depends on these exact encoder weights. They will be
        included in the final export. This is the same R3D-18 encoder family as the existing
        violence pipeline, but the new three-class classifier is trained independently.
    ''')
    code('''
        from torchvision.models.video import R3D_18_Weights

        ENCODER = DRIVE_ROOT / 'models' / 'r3d_18-b3b3357e.pth'
        ENCODER.parent.mkdir(parents=True, exist_ok=True)
        if ENCODER.exists() and not sha256_file(ENCODER).startswith('b3b3357e'):
            raise ValueError('Encoder checksum mismatch. Remove the incomplete file and rerun this cell.')
        if not ENCODER.exists():
            temporary = ENCODER.with_suffix('.download')
            torch.hub.download_url_to_file(R3D_18_Weights.KINETICS400_V1.url, str(temporary),
                                          hash_prefix='b3b3357e')
            temporary.replace(ENCODER)
        print('Encoder:', ENCODER)
    ''')
    markdown('''
        ## 7. Train (or resume)

        Feature extraction uses the GPU and saves each completed video to Drive. The small
        attention/classification head trains on CPU. Checkpoints are saved every epoch.
        After a disconnect, rerun setup/storage/audit cells with the same settings, then
        this cell. Completed feature files and optimizer progress are reused.

        Class weights counter the large normal class. The best checkpoint is selected by
        **validation macro F1**, with weighted validation loss breaking ties. Test data does
        not choose the model. A decode error stops the run instead of silently omitting data.
    ''')
    code('''
        command = [sys.executable, '-m', 'trace.train_crime',
                   '--manifest', str(manifest), '--video-root', str(VIDEO_ROOT),
                   '--encoder', str(ENCODER), '--cache', str(CACHE_DIR),
                   '--output', str(RUN_DIR), '--device', 'cuda:0',
                   '--clips', str(CLIPS), '--epochs', str(EPOCHS)]
        if RUN_DIR.exists():
            command.append('--resume')
        if EVALUATE_TEST:
            command.append('--evaluate-test')
        subprocess.run(command, check=True)
    ''')
    markdown('''
        ## 8. Review class-specific results

        Predicting normal for every original dataset entry would score about **86.4% accuracy**.
        Read burglary and vandalism precision/recall, macro F1 and the confusion matrix.
        These are whole-video results, not event detection rates. Small test-class counts
        make results uncertain; evaluate new-camera footage before live integration.
    ''')
    code('''
        import pandas as pd
        import matplotlib.pyplot as plt

        report = json.loads((RUN_DIR / 'metrics.json').read_text())
        split = 'test' if report['test_evaluated'] else 'val'
        values = report['metrics'][split]
        print('Split:', split, '| Best epoch:', report['best_epoch'])
        print('Macro F1:', round(values['macro_f1'], 4),
              '| Balanced accuracy:', round(values['balanced_accuracy'], 4))
        display(pd.DataFrame(values['per_class']).T)
        matrix = np.array(values['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(matrix, cmap='Blues')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=report['classes'],
               yticklabels=report['classes'], xlabel='Predicted', ylabel='Actual',
               title=f'{split.title()} confusion matrix')
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha='center', va='center')
        fig.tight_layout()
        fig.savefig(RUN_DIR / 'confusion_matrix.png', dpi=160)
        plt.show()
    ''')
    markdown('''
        ## 9. Export for your Windows TRACE folder

        The ZIP contains `models/`, `runs/<RUN_NAME>/`, audit files and the source snapshot.
        Download it from Drive and extract into your local `T.R.A.C.E` folder. For the
        default run name, double-click `run_crime.cmd`. Custom run names need the explicit
        prediction command printed below. This does not connect the model to live CCTV yet.
    ''')
    code('''
        EXPORT_DIR = DRIVE_ROOT / 'exports'
        EXPORT_DIR.mkdir(exist_ok=True)
        (RUN_DIR / 'source_snapshot.zip').write_bytes(SOURCE_ZIP)
        (RUN_DIR / 'source_hashes.json').write_text(json.dumps(SOURCE_HASHES, indent=2))
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                                capture_output=True, text=True, check=True)
        (RUN_DIR / 'environment.txt').write_text(freeze.stdout)
        export = EXPORT_DIR / f'{RUN_NAME}.zip'
        temporary = export.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(ENCODER, 'models/' + ENCODER.name)
            for name in ('best.pt', 'metrics.json', 'predictions.csv', 'history.json', 'config.json',
                         'confusion_matrix.png', 'environment.txt', 'source_snapshot.zip', 'source_hashes.json'):
                bundle.write(RUN_DIR / name, f'runs/{RUN_NAME}/{name}')
            for name in ('manifest.csv', 'dataset_report.json'):
                bundle.write(AUDIT_DIR / name, f'runs/{RUN_NAME}/audit/{name}')
        temporary.replace(export)
        print('Download from Google Drive:', export)
        print('Windows prediction command:')
        print(f'.\\\\.venv\\\\Scripts\\\\python.exe -m trace.predict_crime --source "C:\\\\path\\\\to\\\\clip.mp4" --checkpoint runs/{RUN_NAME}/best.pt')
        print('last.pt and feature caches stay on Drive for resuming.')
    ''')
    markdown('''
        ## Limitations and references

        No live-alert speed or false-alerts-per-hour result is established by this notebook.
        Class scores are uncalibrated. A normal prediction does not establish absence of an
        event, and a burglary prediction does not establish unlawful intent or forced entry.
        Sparse samples can miss events. The model uses whole-video labels and cannot provide
        validated event timestamps. Review mistakes and test independent camera recordings.

        [Dataset](https://www.kaggle.com/datasets/kizitoryan/burglary-and-vandalism-detection-dataset/data)
        · [KaggleHub](https://github.com/Kaggle/kagglehub)
        · [Colab limits](https://research.google.com/colaboratory/faq.html)
        · [R3D-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html)
    ''')
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.12'),
        colab=dict(name='Train_Crime_Colab.ipynb', provenance=[]), accelerator='GPU'),
        nbformat=4, nbformat_minor=5)
    for index, cell in enumerate(cells):
        cell['id'] = f'trace-crime-{index:02d}'
    destination = ROOT / 'notebooks/Train_Crime_Colab.ipynb'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Created {destination} ({len(cells)} cells)')


if __name__ == '__main__':
    build()
