"""Train a class-balanced, attention-pooled video classifier with resumable caches."""
import argparse
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torchvision

from trace.crime import CrimeHead, FORMAT, LIMITATIONS, atomic_save, metrics
from trace.crime_data import CLASSES, file_hash, read_manifest, video_path
from trace.violence import RECIPE, encode, load_encoder


def extract_features(rows, video_root, encoder, cache, clips, device):
    weight_hash = file_hash(encoder)
    cache = Path(cache) / f'{RECIPE}-{weight_hash}-clips{clips}'
    cache.mkdir(parents=True, exist_ok=True)
    features, model = [], None
    for index, row in enumerate(rows):
        source = video_path(video_root, row['path'])
        if file_hash(source) != row['sha256']:
            raise ValueError(f'Video changed since dataset audit: {source}')
        target = cache / (row['sha256'] + '.pt')
        if target.exists():
            saved = torch.load(target, map_location='cpu', weights_only=True)
        else:
            if model is None:
                model = load_encoder(encoder, device)
            windows, metadata = encode(model, source, clips, device)
            saved = dict(features=windows, metadata=metadata, sha256=row['sha256'],
                         encoder_sha256=weight_hash, recipe=RECIPE)
            atomic_save(saved, target)
        windows = saved['features']
        if (saved.get('sha256') != row['sha256'] or saved.get('encoder_sha256') != weight_hash
                or saved.get('recipe') != RECIPE or tuple(windows.shape) != (clips, 512)
                or not torch.isfinite(windows).all()):
            raise ValueError(f'Invalid cached features; remove this cache file and retry: {target}')
        features.append(windows.float())
        print(f'Features {index + 1}/{len(rows)}: {row["path"]}', flush=True)
    return torch.stack(features)


def fit(features, labels, splits, output, metadata, epochs=200, patience=25, learning_rate=.001,
        resume=False, evaluate_test=False):
    """Train only on train bags, select by validation macro F1, optionally evaluate test."""
    output = Path(output)
    masks = {split: torch.tensor([s == split for s in splits]) for split in ('train', 'val', 'test')}
    for split, mask in masks.items():
        if set(labels[mask].tolist()) != {0, 1, 2}:
            raise ValueError(f'{split} must contain all three classes.')
    training = features[masks['train']]
    mean = training.mean(dim=(0, 1))
    std = training.std(dim=(0, 1)).clamp_min(.01)
    normalized = (features - mean) / std
    counts = torch.bincount(labels[masks['train']], minlength=3).float()
    weights = counts.sum() / (3 * counts)
    # Reset after encoder construction so cache hits cannot change initialization.
    torch.manual_seed(42)
    head = CrimeHead()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=.01)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    start, stale, best_f1, best_loss, history = 1, 0, -1., float('inf'), []
    last_path = output / 'last.pt'
    if resume and last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=True)
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        start, stale = saved['epoch'] + 1, saved['stale']
        best_f1, best_loss, history = saved['best_f1'], saved['best_loss'], saved['history']
        # last.pt includes the selected checkpoint so interrupted writes are recoverable.
        atomic_save(saved['best_checkpoint'], output / 'best.pt')
    best_checkpoint = (torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
                       if resume and last_path.exists() else None)
    for epoch in range(start, epochs + 1):
        if stale >= patience:
            break
        head.train()
        optimizer.zero_grad()
        logits, _ = head(normalized[masks['train']])
        loss = criterion(logits, labels[masks['train']])
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            val_logits, _ = head(normalized[masks['val']])
            val_loss = criterion(val_logits, labels[masks['val']]).item()
            val_f1 = metrics(labels[masks['val']].tolist(), val_logits.argmax(-1).tolist())['macro_f1']
        if not np.isfinite(loss.item()) or not np.isfinite(val_loss):
            raise ValueError('Non-finite training loss; last complete epoch remains available.')
        history.append(dict(epoch=epoch, train_loss=loss.item(), val_loss=val_loss, val_macro_f1=val_f1))
        if val_f1 > best_f1 + 1e-8 or (abs(val_f1 - best_f1) <= 1e-8 and val_loss < best_loss - 1e-5):
            best_f1, best_loss, stale = val_f1, val_loss, 0
            best_checkpoint = dict(format=FORMAT, classes=CLASSES, recipe=RECIPE,
                                   head={k: v.detach().clone() for k, v in head.state_dict().items()},
                                   mean=mean, std=std, best_epoch=epoch, **metadata)
            atomic_save(best_checkpoint, output / 'best.pt')
        else:
            stale += 1
        atomic_save(dict(head=head.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                         stale=stale, best_f1=best_f1, best_loss=best_loss, history=history,
                         best_checkpoint=best_checkpoint), last_path)
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}: loss={loss.item():.4f}, val macro F1={val_f1:.4f}', flush=True)
    if best_checkpoint is None:
        raise ValueError('No trained checkpoint is available.')
    head.load_state_dict(best_checkpoint['head'])
    head.eval()
    evaluated = ('train', 'val', 'test') if evaluate_test else ('train', 'val')
    reports, scores = {}, {}
    with torch.inference_mode():
        for split in evaluated:
            logits, _ = head(normalized[masks[split]])
            scores[split] = logits.softmax(-1).numpy()
            reports[split] = metrics(labels[masks[split]].tolist(), logits.argmax(-1).tolist())
    report = dict(model=FORMAT, classes=CLASSES, best_epoch=best_checkpoint['best_epoch'],
                  epochs_completed=history[-1]['epoch'], class_weights=weights.tolist(),
                  selection='validation macro F1; ties resolved by weighted validation loss',
                  metrics=reports, limitations=LIMITATIONS,
                  test_evaluated=evaluate_test, **metadata)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return report, scores


def train(args):
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Enable a Colab GPU or use --device cpu.')
    rows = read_manifest(args.manifest)
    config = dict(manifest_sha256=file_hash(args.manifest), encoder_sha256=file_hash(args.encoder),
                  clips=args.clips, learning_rate=args.learning_rate, patience=args.patience,
                  recipe=RECIPE, format=FORMAT, seed=42,
                  source_sha256={name: file_hash(Path(__file__).parent / name)
                                 for name in ('crime.py', 'crime_data.py', 'train_crime.py', 'violence.py')})
    output = Path(args.output)
    if output.exists():
        if not args.resume:
            raise ValueError('Output exists. Use --resume or a new output directory.')
        if not (output / 'config.json').is_file():
            raise ValueError('Cannot resume: config.json is missing.')
        if json.loads((output / 'config.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Dataset, source code, encoder or training settings changed; use a new output.')
        if (output / 'last.pt').exists():
            last = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
            if args.epochs < last['epoch']:
                raise ValueError('--epochs cannot be less than the completed epoch count.')
    else:
        output.mkdir(parents=True)
        (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    started = time.perf_counter()
    features = extract_features(rows, args.video_root, args.encoder, args.cache, args.clips, args.device)
    metadata = dict(encoder_sha256=config['encoder_sha256'], clips=args.clips,
                    manifest_sha256=config['manifest_sha256'], source_sha256=config['source_sha256'],
                    software=dict(python=platform.python_version(), torch=str(torch.__version__),
                                  torchvision=str(torchvision.__version__)))
    report, scores = fit(features, torch.tensor([int(r['label']) for r in rows]),
                         [r['split'] for r in rows], output, metadata, args.epochs,
                         args.patience, args.learning_rate, args.resume, args.evaluate_test)
    with (output / 'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'split', 'actual', 'predicted', *CLASSES])
        for split, values in scores.items():
            selected = [r for r in rows if r['split'] == split]
            for row, score in zip(selected, values):
                writer.writerow([row['path'], split, CLASSES[int(row['label'])],
                                 CLASSES[int(score.argmax())], *score.tolist()])
    print(json.dumps(report, indent=2))
    print(f'Completed in {(time.perf_counter() - started) / 60:.1f} minutes. Output: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--encoder', default='models/r3d_18-b3b3357e.pth')
    parser.add_argument('--cache', default='data/crime_features')
    parser.add_argument('--output', default='runs/crime_baseline')
    parser.add_argument('--device', default='cpu', help='Only frozen encoder extraction uses the GPU')
    parser.add_argument('--clips', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true', help='Evaluate held-out test after model selection')
    args = parser.parse_args()
    if min(args.clips, args.epochs, args.patience, args.threads) < 1 or not 0 < args.learning_rate < 1:
        parser.error('Counts must be positive and learning rate between 0 and 1.')
    train(args)


if __name__ == '__main__':
    main()
